"""
PPO Implementation Aligned with TWIST-MASTER Teacher Configuration

This PPO implementation exactly matches TWIST-MASTER's teacher training hyperparameters
but works with HDMI's observation format. No Motion Encoder - direct MLP processing.

Key alignments with TWIST:
- train_every: 24 (TWIST: num_steps_per_env=24)
- ppo_epochs: 5 (TWIST: num_learning_epochs=5)
- num_minibatches: 4 (TWIST: num_mini_batches=4)
- lr: 2e-4 (TWIST: learning_rate=2e-4)
- entropy_coef: 0.01 (TWIST: entropy_coef=0.01)
- desired_kl: 0.008 (TWIST: desired_kl=0.008, adaptive schedule)
- actor_hidden_dims: [512, 512, 256, 128] (TWIST: actor_hidden_dims)
- critic_hidden_dims: [512, 512, 256, 128] (TWIST: critic_hidden_dims)
- layer_norm: True (TWIST: layer_norm=True)
- init_noise_std: 1.0 (TWIST: init_noise_std=1.0)
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
from .common import *

torch.set_float32_matmul_precision('high')

import active_adaptation
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class PPOTwistConfig:
    """PPO Configuration Aligned with TWIST-MASTER Teacher"""
    _target_: str = "active_adaptation.learning.ppo.ppo_twist.PPOTwistPolicy"
    name: str = "ppo_twist"

    # TWIST-aligned training hyperparameters
    train_every: int = 24  # TWIST: num_steps_per_env = 24
    ppo_epochs: int = 5    # TWIST: num_learning_epochs = 5
    num_minibatches: int = 4  # TWIST: num_mini_batches = 4
    lr: float = 2e-4       # TWIST: learning_rate = 2e-4
    clip_param: float = 0.2  # TWIST: clip_param = 0.2
    entropy_coef: float = 0.01  # TWIST: entropy_coef = 0.01
    init_noise_scale: float = 1.0  # TWIST: init_noise_std = 1.0
    load_noise_scale: float | None = None
    desired_kl: Union[float, None] = 0.008  # TWIST: desired_kl = 0.008 (adaptive LR)

    # TWIST-aligned network architecture
    actor_hidden_dims: Tuple[int, ...] = (512, 512, 256, 128)  # TWIST: [512, 512, 256, 128]
    critic_hidden_dims: Tuple[int, ...] = (512, 512, 256, 128)  # TWIST: [512, 512, 256, 128]
    layer_norm: Union[str, None] = "before"  # TWIST: layer_norm = True
    activation: str = "silu"  # TWIST: activation = 'silu'

    # Value normalization and other settings
    value_norm: bool = False
    compile: bool = True
    max_grad_norm: float = 1.0  # TWIST: max_grad_norm = 1.0

    # GAE parameters (TWIST defaults)
    gamma: float = 0.99  # TWIST: gamma = 0.99
    lmbda: float = 0.95  # TWIST: lam = 0.95

    # Don't clamp negative rewards (TWIST doesn't clamp)
    clamp_negative_rewards: bool = False

    # Action std schedule (TWIST-aligned)
    # Format: [init_std, final_std, warmup_iters, decay_iters]
    std_schedule: Tuple[float, float, int, int] = (1.0, 0.4, 4000, 1500)
    use_std_schedule: bool = True  # Enable std scheduling

    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (OBS_KEY, OBS_PRIV_KEY)


cs = ConfigStore.instance()
cs.store("ppo_twist", node=PPOTwistConfig, group="algo")


class PPOTwistPolicy(TensorDictModuleBase):
    """
    PPO Policy Aligned with TWIST-MASTER Teacher

    Architecture:
    - Actor: [obs] → MLP[512, 512, 256, 128] → [action_mean, action_std]
    - Critic: [obs + priv] → MLP[512, 512, 256, 128] → [state_value]
    - No Motion Encoder - direct MLP processing of full observation

    Key differences from base PPO:
    - TWIST-aligned hyperparameters (see PPOTwistConfig)
    - Deeper network: 4 layers instead of 3
    - Adaptive learning rate based on KL divergence
    - SiLU activation (TWIST default)
    - Layer normalization on second-to-last layer
    """

    def __init__(
        self,
        cfg: PPOTwistConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = PPOTwistConfig(**cfg)
        self.device = device

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = self.cfg.max_grad_norm
        self.desired_kl = self.cfg.desired_kl
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(self.cfg.gamma, self.cfg.lmbda)

        # Std schedule tracking
        self.iteration_counter = 0
        self.std_schedule = self.cfg.std_schedule

        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()

        # Get activation function (TWIST uses SiLU)
        activation_fn = get_activation(self.cfg.activation)

        # Build Actor Network (TWIST architecture: [512, 512, 256, 128])
        actor_layers = []
        obs_dim = observation_spec[OBS_KEY].shape[-1]
        prev_dim = obs_dim

        for i, hidden_dim in enumerate(self.cfg.actor_hidden_dims):
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            # TWIST applies layer norm on second-to-last layer
            if self.cfg.layer_norm == "before" and i == len(self.cfg.actor_hidden_dims) - 2:
                actor_layers.append(nn.LayerNorm(hidden_dim))
            actor_layers.append(activation_fn)
            prev_dim = hidden_dim

        actor_backbone = nn.Sequential(*actor_layers)

        actor_module = TensorDictSequential(
            TensorDictModule(actor_backbone, [OBS_KEY], ["_actor_feature"]),
            TensorDictModule(
                Actor(self.action_dim,
                      init_noise_scale=self.cfg.init_noise_scale,
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

        # Build Critic Network (TWIST architecture: [512, 512, 256, 128])
        critic_layers = []
        critic_input_dim = observation_spec[OBS_KEY].shape[-1] + observation_spec[OBS_PRIV_KEY].shape[-1]
        prev_dim = critic_input_dim

        for i, hidden_dim in enumerate(self.cfg.critic_hidden_dims):
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            # TWIST applies layer norm on second-to-last layer
            if self.cfg.layer_norm == "before" and i == len(self.cfg.critic_hidden_dims) - 2:
                critic_layers.append(nn.LayerNorm(hidden_dim))
            critic_layers.append(activation_fn)
            prev_dim = hidden_dim

        # Final output layer
        critic_layers.append(nn.Linear(prev_dim, 1))

        self.critic = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "_critic_input"),
            TensorDictModule(nn.Sequential(*critic_layers), ["_critic_input"], ["state_value"])
        ).to(self.device)

        # Forward pass to initialize lazy layers
        self.actor(fake_input)
        self.critic(fake_input)

        # Optimizer
        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr
        )

        # Weight initialization (TWIST uses orthogonal init with gain=0.01)
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)

        self.actor.apply(init_)
        self.critic.apply(init_)

        # Distributed training setup
        if active_adaptation.is_distributed():
            distr.init_process_group(
                backend="nccl",
                world_size=active_adaptation.get_world_size(),
                rank=active_adaptation.get_local_rank()
            )
            for param in self.actor.parameters():
                distr.broadcast(param, src=0)
            for param in self.critic.parameters():
                distr.broadcast(param, src=0)
            self.world_size = active_adaptation.get_world_size()

        self.update = self._update
        if self.cfg.compile:
            self.update = torch.compile(self.update)

    def get_rollout_policy(self, mode: str="train"):
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

                # TWIST uses adaptive learning rate based on KL divergence
                if self.desired_kl is not None:
                    kl = infos[-1]["actor/kl"]
                    actor_lr = self.opt.param_groups[0]["lr"]
                    if kl > self.desired_kl * 2.0:
                        actor_lr = max(1e-5, actor_lr / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        actor_lr = min(1e-2, actor_lr * 1.5)
                    for param_group in self.opt.param_groups:
                        param_group["lr"] = actor_lr

        # Update action std schedule (TWIST-aligned)
        if self.cfg.use_std_schedule:
            self._update_action_std()

        # Increment iteration counter
        self.iteration_counter += 1

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["actor/action_std"] = self.actor.module[1].module.std.mean().item()  # Log current std
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_std"] = tensordict["ret"].std().item()
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _compute_advantage(
        self,
        tensordict: TensorDict,
        critic: TensorDictModule,
        adv_key: str="adv",
        ret_key: str="ret",
        update_value_norm: bool=True,
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(tensordict_flat)
                critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        # CRITICAL: Do NOT clamp negative rewards (TWIST doesn't clamp)
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
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
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
            for param in self.actor.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size
            for param in self.critic.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size

        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
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

    def _update_action_std(self):
        """
        Update action standard deviation based on schedule (TWIST-aligned)

        Schedule format: [init_std, final_std, warmup_iters, decay_iters]
        Example: [1.0, 0.4, 4000, 1500]

        Timeline:
        - iter 0-4000: std = 1.0 (warmup, high exploration)
        - iter 4000-5500: std = 1.0 → 0.4 (linear decay)
        - iter 5500+: std = 0.4 (exploitation)
        """
        init_std, final_std, warmup_iters, decay_iters = self.std_schedule

        # Calculate std coefficient based on current iteration
        if self.iteration_counter < warmup_iters:
            # Warmup period: keep initial std
            std_coef = 1.0
        elif self.iteration_counter < warmup_iters + decay_iters:
            # Decay period: linear interpolation
            progress = (self.iteration_counter - warmup_iters) / decay_iters
            std_coef = 1.0 - progress * (1.0 - final_std / init_std)
        else:
            # Post-decay: use final std
            std_coef = final_std / init_std

        # Calculate target std
        target_std = init_std * std_coef

        # Update actor std parameter
        # Actor structure: ProbabilisticActor(module=TensorDictSequential(...))
        # Access: self.actor.module[1].module is Actor class
        # Actor has self.actor_std parameter (see common.py:153)
        actor_module = self.actor.module[1].module  # TensorDictModule wrapping Actor
        with torch.no_grad():
            actor_module.actor_std.fill_(target_std)


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x / x.std().clamp(1e-7)


def get_activation(act_name):
    """Get activation function by name (TWIST compatibility)"""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "silu":
        return nn.SiLU()
    else:
        print(f"Invalid activation function: {act_name}. Using ELU.")
        return nn.ELU()

