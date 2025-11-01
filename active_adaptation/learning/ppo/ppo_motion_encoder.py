"""
PPO Policy with TWIST-style Motion Encoder

This module extends the standard PPO implementation with temporal motion encoding,
following the TWIST paper architecture.

Key features:
- 1D CNN motion encoder for temporal sequence compression
- Splits observations into motion and proprioceptive components
- Compatible with HDMI's observation format (past + current + future frames)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import warnings
import functools
import torch.utils._pytree as pytree

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, VecNorm
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule,
    TensorDictSequential,
    CudaGraphModule
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, Tuple
from collections import OrderedDict

from ..utils.valuenorm import ValueNorm1, ValueNormFake
from ..modules.distributions import IndependentNormal
from ..modules.motion_encoder import MotionEncoder1D
from .common import *

torch.set_float32_matmul_precision('high')

import active_adaptation
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class PPOMotionEncoderConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_motion_encoder.PPOMotionEncoderPolicy"
    name: str = "ppo_motion_encoder"
    train_every: int = 24  # Aligned with TWIST (was 32 in base PPO)
    ppo_epochs: int = 5  # Aligned with TWIST (was 3 in base PPO)
    num_minibatches: int = 4  # Aligned with TWIST (was 8 in base PPO)
    lr: float = 2e-4  # Aligned with TWIST (was 1e-4 in base PPO)
    clip_param: float = 0.2
    entropy_coef: float = 0.01  # Aligned with TWIST (was 0.001 in base PPO)
    init_noise_scale: float = 1.5
    load_noise_scale: float | None = None
    desired_kl: Union[float, None] = 0.008  # Adaptive LR (TWIST default)
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False
    compile: bool = True
    clamp_negative_rewards: bool = False  # CRITICAL: Don't clamp negative rewards!

    # Motion encoder specific
    use_motion_encoder: bool = True
    motion_latent_dim: int = 128
    motion_obs_key: str = "ref_motion_windowed"  # Key for motion observations
    motion_tsteps: int = 21  # HDMI default (past 10 + current 1 + future 10)
    motion_input_size: int = 32  # Single frame dim (root_pos 3 + root_ori 6 + joint_pos 23)

    # Network architecture
    actor_hidden_dims: Tuple[int, ...] = (512, 512, 256, 128)  # Aligned with TWIST
    critic_hidden_dims: Tuple[int, ...] = (512, 512, 256, 128)  # Aligned with TWIST

    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (OBS_KEY, OBS_PRIV_KEY)


cs = ConfigStore.instance()
cs.store("ppo_motion_encoder", node=PPOMotionEncoderConfig, group="algo")


class PPOMotionEncoderPolicy(TensorDictModuleBase):
    """
    PPO Policy with Motion Encoder

    Architecture:
    1. Motion Encoder: Compresses temporal reference motion sequences
       Input: [batch, tsteps * motion_input_size]
       Output: [batch, motion_latent_dim]

    2. Actor: Policy network
       Input: [motion_latent + current_frame_motion + proprio]
       Hidden: actor_hidden_dims
       Output: [action_mean, action_std]

    3. Critic: Value network
       Input: [motion_latent + current_frame_motion + proprio + priv]
       Hidden: critic_hidden_dims
       Output: [state_value]
    """

    def __init__(
        self,
        cfg: PPOMotionEncoderConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = PPOMotionEncoderConfig(**cfg)
        self.device = device

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.desired_kl = self.cfg.desired_kl
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)

        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()

        # Extract motion observation dimension from spec
        # TWIST observations are structured as CompositeSpec with named keys
        try:
            if hasattr(observation_spec[OBS_KEY], 'keys') and self.cfg.motion_obs_key in observation_spec[OBS_KEY].keys():
                # Structured observation (TensorDict)
                motion_obs_dim = observation_spec[OBS_KEY][self.cfg.motion_obs_key].shape[-1]
                self.num_motion_obs = motion_obs_dim
                self.motion_input_size = motion_obs_dim // self.cfg.motion_tsteps
                self.use_structured_obs = True
                print(f"[MotionEncoder] Detected STRUCTURED observation with '{self.cfg.motion_obs_key}' key")
                print(f"[MotionEncoder] motion_obs_dim={motion_obs_dim}, "
                      f"tsteps={self.cfg.motion_tsteps}, single_frame_dim={self.motion_input_size}")
            else:
                # Flat observation (concatenated tensor) - TWIST default
                # Assume motion obs comes AFTER proprio obs
                total_obs_dim = observation_spec[OBS_KEY].shape[-1]
                self.num_motion_obs = self.cfg.motion_tsteps * self.cfg.motion_input_size
                self.motion_input_size = self.cfg.motion_input_size
                self.use_structured_obs = False

                # Calculate proprio dimension
                self.num_proprio_obs = total_obs_dim - self.num_motion_obs

                print(f"[MotionEncoder] Detected FLAT observation (TWIST format)")
                print(f"[MotionEncoder] total_obs_dim={total_obs_dim}, "
                      f"proprio_dim={self.num_proprio_obs}, motion_dim={self.num_motion_obs}")
                print(f"[MotionEncoder] tsteps={self.cfg.motion_tsteps}, single_frame_dim={self.motion_input_size}")
                print(f"[MotionEncoder] Observation order: [proprio_history_combined, ref_motion_windowed]")
        except Exception as e:
            # Fallback to config defaults
            # Assume TWIST flat format by default
            total_obs_dim = observation_spec[OBS_KEY].shape[-1]
            self.num_motion_obs = self.cfg.motion_tsteps * self.cfg.motion_input_size
            self.motion_input_size = self.cfg.motion_input_size
            self.num_proprio_obs = total_obs_dim - self.num_motion_obs  # ← FIX: Set num_proprio_obs
            self.use_structured_obs = False
            print(f"[MotionEncoder] Warning: Could not detect observation structure ({e})")
            print(f"[MotionEncoder] Using TWIST flat format as fallback")
            print(f"[MotionEncoder] total_obs_dim={total_obs_dim}, "
                  f"proprio_dim={self.num_proprio_obs}, motion_dim={self.num_motion_obs}")
            print(f"[MotionEncoder] tsteps={self.cfg.motion_tsteps}, single_frame_dim={self.motion_input_size}")

        # Build Motion Encoder
        if self.cfg.use_motion_encoder:
            self.motion_encoder = MotionEncoder1D(
                activation_fn=nn.ELU(),
                input_size=self.motion_input_size,
                tsteps=self.cfg.motion_tsteps,
                output_size=self.cfg.motion_latent_dim,
                tanh_encoder_output=False
            ).to(self.device)
            print(f"[MotionEncoder] Initialized with {sum(p.numel() for p in self.motion_encoder.parameters()):,} parameters")
        else:
            self.motion_encoder = None
            print("[MotionEncoder] Disabled - using direct MLP")

        # Calculate actor input dimension
        # Input: motion_latent + current_frame_motion + proprio_obs
        # Note: We extract current frame (middle frame) from motion obs
        if self.cfg.use_motion_encoder:
            if hasattr(self, 'num_proprio_obs'):
                # TWIST flat format
                proprio_dim = self.num_proprio_obs
            else:
                # Legacy format or structured obs
                proprio_dim = observation_spec[OBS_KEY].shape[-1] - self.num_motion_obs

            actor_input_dim = (
                self.cfg.motion_latent_dim +  # Encoded temporal sequence
                self.motion_input_size +       # Current frame motion
                proprio_dim                    # Proprioceptive observations
            )
            print(f"[MotionEncoder] Actor input dim = {self.cfg.motion_latent_dim} (latent) + "
                  f"{self.motion_input_size} (current_frame) + {proprio_dim} (proprio) = {actor_input_dim}")
        else:
            actor_input_dim = observation_spec[OBS_KEY].shape[-1]
            print(f"[MotionEncoder] Actor input dim = {actor_input_dim} (no encoder, direct MLP)")

        # Build Actor
        actor_layers = []
        prev_dim = actor_input_dim
        for i, hidden_dim in enumerate(self.cfg.actor_hidden_dims):
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.cfg.layer_norm == "before" and i == len(self.cfg.actor_hidden_dims) - 2:
                actor_layers.append(nn.LayerNorm(hidden_dim))
            actor_layers.append(nn.ELU())
            prev_dim = hidden_dim

        self.actor_backbone = nn.Sequential(*actor_layers).to(self.device)

        actor_module = TensorDictSequential(
            TensorDictModule(
                self._process_actor_input,
                [OBS_KEY],
                ["_actor_input"]
            ),
            TensorDictModule(
                self.actor_backbone,
                ["_actor_input"],
                ["_actor_feature"]
            ),
            TensorDictModule(
                Actor(self.action_dim, init_noise_scale=self.cfg.init_noise_scale,
                     load_noise_scale=self.cfg.load_noise_scale),
                ["_actor_feature"],
                ["loc", "scale"]
            )
        )

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        # Build Critic (uses privileged info)
        critic_input_dim = actor_input_dim + observation_spec[OBS_PRIV_KEY].shape[-1]

        critic_layers = []
        prev_dim = critic_input_dim
        for i, hidden_dim in enumerate(self.cfg.critic_hidden_dims):
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.cfg.layer_norm == "before" and i == len(self.cfg.critic_hidden_dims) - 2:
                critic_layers.append(nn.LayerNorm(hidden_dim))
            critic_layers.append(nn.ELU())
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))

        self.critic_backbone = nn.Sequential(*critic_layers).to(self.device)

        self.critic = TensorDictSequential(
            TensorDictModule(
                self._process_critic_input,
                [OBS_KEY, OBS_PRIV_KEY],
                ["_critic_input"]
            ),
            TensorDictModule(
                self.critic_backbone,
                ["_critic_input"],
                ["state_value"]
            )
        ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        # Optimizer
        params = [
            {"params": self.actor.parameters()},
            {"params": self.critic.parameters()},
        ]
        if self.cfg.use_motion_encoder:
            params.append({"params": self.motion_encoder.parameters()})

        self.opt = torch.optim.Adam(params, lr=cfg.lr)

        # Initialize weights
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)

        self.actor_backbone.apply(init_)
        self.critic_backbone.apply(init_)
        if self.motion_encoder is not None:
            self.motion_encoder.apply(init_)

        # Distributed setup
        if active_adaptation.is_distributed():
            distr.init_process_group(
                backend="nccl",
                world_size=active_adaptation.get_world_size(),
                rank=active_adaptation.get_local_rank()
            )
            for param in self.parameters():
                distr.broadcast(param, src=0)
            self.world_size = active_adaptation.get_world_size()

        self.update = self._update
        if self.cfg.compile:
            self.update = torch.compile(self.update)

    def _extract_motion_and_proprio(self, obs: torch.Tensor):
        """
        Split observation into motion and proprioceptive components

        Args:
            obs: Full observation [batch, obs_dim]

        Returns:
            motion_obs: Motion sequence [batch, tsteps * motion_input_size]
            proprio_obs: Proprioceptive observation [batch, proprio_dim]
            current_frame_motion: Current frame motion [batch, motion_input_size]
        """
        # TWIST uses FLAT observation: [proprio_history_combined, ref_motion_windowed]
        # Motion obs comes AFTER proprio obs
        if hasattr(self, 'num_proprio_obs'):
            # TWIST flat format: [proprio, motion]
            proprio_obs = obs[:, :self.num_proprio_obs]
            motion_obs = obs[:, self.num_proprio_obs:]
        else:
            # Legacy format: motion first
            motion_obs = obs[:, :self.num_motion_obs]
            proprio_obs = obs[:, self.num_motion_obs:]

        # Extract current frame (middle frame of the sequence)
        # HDMI/TWIST: 21 frames (past 10 + current 1 + future 10)
        # Current frame is at index 10 (0-indexed)
        current_frame_idx = self.cfg.motion_tsteps // 2
        start_idx = current_frame_idx * self.motion_input_size
        end_idx = start_idx + self.motion_input_size
        current_frame_motion = motion_obs[:, start_idx:end_idx]

        return motion_obs, proprio_obs, current_frame_motion

    def _process_actor_input(self, obs: torch.Tensor):
        """Process observation for actor input"""
        if not self.cfg.use_motion_encoder:
            return obs

        motion_obs, proprio_obs, current_frame_motion = self._extract_motion_and_proprio(obs)

        # Encode motion sequence
        motion_latent = self.motion_encoder(motion_obs)

        # Concatenate: motion_latent + current_frame + proprio
        actor_input = torch.cat([
            motion_latent,
            current_frame_motion,
            proprio_obs
        ], dim=1)

        return actor_input

    def _process_critic_input(self, obs: torch.Tensor, obs_priv: torch.Tensor):
        """Process observation for critic input"""
        actor_input = self._process_actor_input(obs)
        critic_input = torch.cat([actor_input, obs_priv], dim=1)
        return critic_input

    def get_rollout_policy(self, mode: str = "train"):
        policy = TensorDictSequential(self.actor)
        if self.cfg.compile:
            policy = torch.compile(policy)
        return policy

    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.exclude("stats")
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self.update(minibatch))

                if self.desired_kl is not None:  # adaptive learning rate
                    kl = infos[-1]["actor/kl"]
                    actor_lr = self.opt.param_groups[0]["lr"]
                    if kl > self.desired_kl * 2.0:
                        actor_lr = max(1e-5, actor_lr / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        actor_lr = min(1e-2, actor_lr * 1.5)
                    for param_group in self.opt.param_groups:
                        param_group["lr"] = actor_lr

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_std"] = tensordict["ret"].std().item()
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _compute_advantage(
        self,
        tensordict: TensorDict,
        critic: TensorDictModule,
        adv_key: str = "adv",
        ret_key: str = "ret",
        update_value_norm: bool = True,
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(tensordict_flat)
                critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        # CRITICAL FIX: Do NOT clamp negative rewards (TWIST doesn't clamp)
        # Original base PPO had .clamp_min(0.) which discards negative rewards!
        if self.cfg.clamp_negative_rewards:
            rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True).clamp_min(0.)
        else:
            rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True)

        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)
        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        dist: IndependentNormal = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        valid = (~tensordict["is_init"])
        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1. - self.clip_param, 1. + self.clip_param)
        policy_loss = - (torch.min(surr1, surr2) * valid).mean()
        entropy_loss = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * valid).mean()

        loss = policy_loss + entropy_loss + value_loss
        self.opt.zero_grad()
        loss.backward()

        if active_adaptation.is_distributed():
            for param in self.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size

        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        if self.motion_encoder is not None:
            motion_encoder_grad_norm = nn.utils.clip_grad_norm_(self.motion_encoder.parameters(), self.max_grad_norm)
        else:
            motion_encoder_grad_norm = 0.0

        self.opt.step()

        with torch.no_grad():
            explained_var_numerator = F.mse_loss(values, b_returns)
            explained_var_denominator = b_returns.var()
            explained_var = 1 - explained_var_numerator / explained_var_denominator

            explained_var_valid_numerator = F.mse_loss(values[valid], b_returns[valid])
            explained_var_valid_denominator = b_returns[valid].var()
            explained_var_valid = 1 - explained_var_valid_numerator / explained_var_valid_denominator

            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            loc, scale = dist.loc, dist.scale
            loc_old, scale_old = tensordict["loc"], tensordict["scale"]
            kl = torch.sum(
                torch.log(scale) - torch.log(scale_old)
                + (torch.square(scale_old) + torch.square(loc_old - loc)) / (2.0 * torch.square(scale))
                - 0.5,
                axis=-1,
            ).mean()

        return {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/mean_std": tensordict["scale"].detach().mean(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/kl": kl,
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
            "critic/explained_var_valid": explained_var_valid,
            "motion_encoder/grad_norm": motion_encoder_grad_norm,
        }

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool = False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x / x.std().clamp(1e-7)
