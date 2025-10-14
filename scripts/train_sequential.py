import torch
# import warp
import hydra
import numpy as np
import einops
import wandb
import logging
import os
import sys
import time
import datetime
import shutil
import inspect

from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle
from hydra import compose

import active_adaptation as aa
from isaaclab.app import AppLauncher
# from active_adaptation.utils.torchrl import SyncDataCollector
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictModuleBase
from tensordict import TensorDict

# local import
from scripts.helpers import make_env_policy, EpisodeStats, evaluate
import multiprocessing

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")


def run_training_stage(cfg: DictConfig, return_queue: multiprocessing.Queue = None) -> str:
    OmegaConf.set_struct(cfg, False)
    
    print(f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}")
    app_launcher = AppLauncher(
        OmegaConf.to_container(cfg.app),
        distributed=aa.is_distributed(),
        device=f"cuda:{aa.get_local_rank()}"
    )
    simulation_app = app_launcher.app

    run = wandb.init(
        job_type=cfg.wandb.job_type,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
    )
    run.config.update(OmegaConf.to_container(cfg))
    
    default_run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    run_idx = run.name.split("-")[-1]
    run.name = f"{run_idx}-{default_run_name}"
    setproctitle(run.name)

    cfg_save_path = os.path.join(run.dir, "cfg.yaml")
    OmegaConf.save(cfg, cfg_save_path)
    run.save(cfg_save_path, policy="now")
    run.save(os.path.join(run.dir, "config.yaml"), policy="now")

    # 2. --- Environment and Policy Creation ---
    # `make_env_policy` will handle loading the checkpoint if `cfg.checkpoint_path` is set
    env, policy, vecnorm = make_env_policy(cfg)

    # Save policy source code for reproducibility
    source_path = inspect.getfile(policy.__class__)
    target_path = os.path.join(run.dir, source_path.split("/")[-1])
    shutil.copy(source_path, target_path)
    wandb.save(target_path, policy="now")

    # 3. --- Training Setup ---
    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
    total_frames = total_frames // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    save_interval = cfg.get("save_interval", -1)

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    def save(policy, checkpoint_name: str, artifact: bool=False):
        ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg
        if "vecnorm" in locals():
            state_dict["vecnorm"] = vecnorm.state_dict()
        torch.save(state_dict, ckpt_path)
        if artifact:
            artifact = wandb.Artifact(
                f"{type(env).__name__}-{type(policy).__name__}",
                type="model"
            )
            artifact.add_file(ckpt_path)
            run.log_artifact(artifact)
        run.save(ckpt_path, policy="now", base_path=run.dir)
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")

    assert env.training
    def should_save(i):
        if not aa.is_main_process():
            return False
        return i > 0 and save_interval > 0 and i % save_interval == 0

    # 4. --- Training Loop ---
    carry = env.reset()
    rollout_policy: TensorDictModuleBase = policy.get_rollout_policy("train")

    with torch.inference_mode():
        tmp_carry = rollout_policy(carry.clone(False))
        tmp_td, _ = env.step_and_maybe_reset(tmp_carry.clone(False))
        tmp_td["next"] = tmp_td["next"].select("done", "terminated", "discount", "reward", "stats", "is_init", "adapt_hx", strict=False)

    N = env.num_envs
    T = cfg.algo.train_every
    device = env.device

    data_buf = TensorDict({}, batch_size=[N, T], device=device)
    for key, value in tmp_td.items(include_nested=True, leaves_only=True):
        shape_tail = value.shape[1:]
        buf = torch.empty((N, T, *shape_tail), dtype=value.dtype, device=device)
        data_buf.set(key, buf)
    logging.info(f"Data buffer size: {data_buf.bytes() / 1e6:.2f} MB")

    if aa.is_main_process():
        progress = tqdm(range(total_iters))
    else:
        progress = range(total_iters)

    env_frames = 0
    start_iter = env.current_iter
    for i in progress:
        rollout_start = time.perf_counter()
        with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
            torch.compiler.cudagraph_mark_step_begin() # for compiled policy
            env.set_progress(start_iter + i)
            for step in range(cfg.algo.train_every):
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                td["next"] = td["next"].select("done", "terminated", "discount", "reward", "stats", "is_init", "adapt_hx", strict=False)
                data_buf[:, step] = td
            policy.critic(data_buf)
            values = data_buf["state_value"]
            data_buf["next", "state_value"] = torch.where(
                data_buf["next", "done"],
                values, # a walkaround to avoid storing the next states
                torch.cat([values[:, 1:], policy.critic(carry.copy())["state_value"].unsqueeze(1)], dim=1)
            )
        rollout_time = time.perf_counter() - rollout_start

        episode_stats.add(data_buf)
        env_frames += data_buf.numel()

        info = {}
        if i % log_interval == 0 and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
        training_start = time.perf_counter()
        info.update(policy.train_op(data_buf))
        training_time = time.perf_counter() - training_start
        info.update(env.extra)
        info.update(env.stats_ema)

        if hasattr(policy, "step_schedule"):
            policy.step_schedule(i / total_iters)

        info["env_frames"] = env_frames
        info["rollout_fps"] = data_buf.numel() / rollout_time
        info["training_time"] = training_time

        if should_save(i):
            save(policy, f"checkpoint_{i}")

        if aa.is_main_process():
            # print(OmegaConf.to_yaml({k: v for k, v in info.items() if (isinstance(v, (float, int)) and not k.startswith("performance_reward"))}))
            run.log(info)

    # 5. --- Finalization and Cleanup ---
    if aa.is_main_process():
        save(policy, "checkpoint_final", artifact=True)

    # Calculate the run path to return
    run_id = run.id
    project = run.project
    entity = run.entity
    run_path = f"{entity}/{project}/{run_id}"

    # IMPORTANT: Put the result in the queue BEFORE exiting.
    if return_queue:
        print(f"Child process sending run_path to parent: {run_path}")
        return_queue.put(run_path)

    # Clean up wandb and Isaac Sim, then force exit
    # The original 'return' statement is now unreachable but kept for clarity.
    print("Finishing W&B run and preparing to exit process.")
    wandb.finish()
    
    # Due to an Isaac Sim bug, simulation_app.close() hangs.
    # We must use os._exit(0) to forcefully terminate this child process.
    # The parent process will not be affected.
    print(f"Child process for stage '{cfg.algo.name}' is now exiting.")
    os._exit(0)


@hydra.main(config_path="../cfg", config_name="train_sequential", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to orchestrate the sequence of training stages.
    This version runs each stage in a separate process to handle the
    os._exit(0) required by Isaac Sim cleanup.
    """
    # 1. --- Capture all command-line overrides ---
    cli_overrides = []
    for arg in sys.argv[1:]:
        if arg.startswith("hydra.") or os.path.basename(__file__) in arg:
            continue
        if not arg.startswith("stages="):
            cli_overrides.append(arg)

    print("="*80)
    print("Detected Command-Line Overrides to be applied to ALL stages:")
    if cli_overrides:
        for ov in cli_overrides:
            print(f"  - {ov}")
    else:
        print("  - None")
    print("="*80)

    previous_run_path = None
    
    # 2. --- Loop through each defined stage ---
    for i, stage_name in enumerate(cfg.stages):
        print("\n" + "="*80)
        print(f"🚀 Preparing to launch STAGE {i+1}/{len(cfg.stages)}: {stage_name}")
        print("="*80)

        # 3. --- Dynamically compose the configuration for THIS stage ---
        stage_overrides = cli_overrides.copy()
        stage_overrides.append(f"algo={stage_name}")
        if previous_run_path:
            stage_overrides.append(f"checkpoint_path=run:{previous_run_path}")
            print(f"Will load checkpoint from previous run: {previous_run_path}")
        
        stage_cfg = compose(config_name="train", overrides=stage_overrides)

        # 4. --- Run the training stage in a separate process ---
        # A queue is used for the child process to return the run_path.
        return_queue = multiprocessing.Queue(1)
        
        OmegaConf.resolve(stage_cfg)
        
        # run_training_stage(stage_cfg, return_queue)
        process = multiprocessing.Process(
            target=run_training_stage, 
            kwargs={'cfg': stage_cfg, 'return_queue': return_queue}
        )
        
        print(f"Starting child process for stage '{stage_name}'...")
        process.start()
        
        # Wait for the child process to finish (or exit).
        process.join()
        print(f"Child process for stage '{stage_name}' has finished.")

        # Check if the process exited successfully
        if process.exitcode != 0:
            print(f"ERROR: Child process for stage '{stage_name}' exited with code {process.exitcode}.")
            # Decide if you want to stop the whole sequence on error
            # raise RuntimeError(f"Stage '{stage_name}' failed.") 
            break # Or just break the loop

        # 5. --- Retrieve the result from the queue and update state ---
        try:
            current_run_path = return_queue.get(timeout=10)
            previous_run_path = current_run_path
            print(f"✅ COMPLETED STAGE {i+1}/{len(cfg.stages)}: {stage_name}")
            print(f"   Run Path: {current_run_path}")
        except Exception as e:
            print(f"ERROR: Could not retrieve run_path from child process for stage '{stage_name}'. Error: {e}")
            break # Stop the sequence if we can't get the path

        print("="*80)

    print("\n🎉🎉🎉 All training stages completed successfully! 🎉🎉🎉")


if __name__ == "__main__":
    main()