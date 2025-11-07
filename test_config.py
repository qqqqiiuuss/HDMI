#!/usr/bin/env python
"""Quick test to verify configuration loads correctly"""

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
import os

# Get absolute path to config directory
config_dir = os.path.join(os.getcwd(), "cfg")

print(f"Loading config from: {config_dir}")

# Initialize Hydra with the config directory
with initialize_config_dir(version_base=None, config_dir=config_dir):
    # Compose the configuration
    cfg = compose(
        config_name="train",
        overrides=[
            "algo=ppo_roa_train_twist_ref",
            "task=G1/hdmi/move_suitcase_twist_ref",
            'task.reference.twist_policy.checkpoint_path="/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/outputs/2025-11-03/21-56-06-G1TwistTeacherAligned-ppotest_1014_twist/wandb/run-20251103_215613-mmk3woo1/files/checkpoint_9000.pt"',
            "wandb.mode=disabled"
        ]
    )

    print("\n=== Configuration Loaded Successfully ===\n")

    print("Task name:", cfg.task.name)
    print("Algorithm:", cfg.algo.name if hasattr(cfg.algo, 'name') else "ppo_roa")
    print("Checkpoint path:", cfg.task.reference.twist_policy.checkpoint_path)

    print("\n=== Termination Configuration ===")
    if hasattr(cfg.task, 'termination'):
        print(OmegaConf.to_yaml(cfg.task.termination))
    else:
        print("No termination config found (will use defaults)")

    print("\n=== Reference Configuration ===")
    if hasattr(cfg.task, 'reference'):
        print(OmegaConf.to_yaml(cfg.task.reference))

    print("\n=== SUCCESS ===")
