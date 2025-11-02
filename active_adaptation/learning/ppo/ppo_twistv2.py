# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
# Modified to align with TWIST-master PPO hyperparameters
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


"""
PPO implementation aligned with TWIST-master hyperparameters.

Key differences from base PPO:
1. Learning rate: 2e-4 (vs 1e-4)
2. PPO epochs: 5 (vs 3)
3. Mini-batches: 4 (vs 8)
4. Entropy coef: 0.005 (vs 0.001)
5. Train every: 24 (vs 32)
6. Gamma: 0.998 (vs 0.99)
7. Desired KL: 0.008 (adaptive LR schedule)
8. Network: Deeper [512, 512, 256, 128] (vs [256, 256])
9. Activation: SiLU (vs ReLU)
10. Action std schedule: Decay from 1.0 to 0.4
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
class PPOTwistV2Config:
    """PPO配置，完全对齐TWIST-master的超参数"""
    _target_: str = "active_adaptation.learning.ppo.ppo_twistv2.PPOTwistV2Policy"
    name: str = "ppo_twistv2"

    # ==================== TWIST-Aligned Hyperparameters ====================
    # 对齐 TWIST-master/legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py
    # class algorithm (line 442-444)

    train_every: int = 24              # TWIST: num_steps_per_env=24 (vs HDMI: 32)
    ppo_epochs: int = 5                # TWIST: num_learning_epochs=5 (vs HDMI: 3)
    num_minibatches: int = 4           # TWIST: num_mini_batches=4 (vs HDMI: 8)
    lr: float = 2e-4                   # TWIST: learning_rate=2e-4 (vs HDMI: 1e-4)
    clip_param: float = 0.2            # TWIST: clip_param=0.2 (相同)
    entropy_coef: float = 0.005        # TWIST: entropy_coef=0.005 (vs HDMI: 0.001)

    # GAE parameters
    gamma: float = 0.998               # TWIST: gamma=0.998 (vs HDMI: 0.99)
    lam: float = 0.95                  # TWIST: lam=0.95 (相同)

    # Adaptive learning rate (KL-based)
    desired_kl: float = 0.008          # TWIST: desired_kl=0.008 (vs HDMI: None)
    schedule: str = "adaptive"         # TWIST: schedule='adaptive' (vs HDMI: 'fixed')

    # Gradient clipping
    max_grad_norm: float = 1.0         # TWIST: max_grad_norm=1.0 (相同)

    # Value function
    value_loss_coef: float = 1.0       # TWIST: value_loss_coef=1.0 (相同)
    use_clipped_value_loss: bool = True # TWIST: use_clipped_value_loss=True

    # ==================== TWIST Action Std Schedule ====================
    # 对齐 TWIST std_schedule = [1.0, 0.4, 4000, 1500]
    # 含义：从 init_std=1.0 在 warmup_steps=4000 步内线性衰减到 final_std=0.4
    # 然后保持 final_std=0.4 for hold_steps=1500 步
    init_noise_scale: float = 1.0      # TWIST: std_schedule[0] = 1.0
    final_noise_scale: float = 0.4     # TWIST: std_schedule[1] = 0.4
    std_warmup_steps: int = 4000       # TWIST: std_schedule[2] = 4000
    std_hold_steps: int = 1500         # TWIST: std_schedule[3] = 1500

    load_noise_scale: float | None = None  # 从checkpoint加载时的std

    # ==================== Network Architecture ====================
    # 对齐 TWIST actor_hidden_dims = [512, 512, 256, 128]
    actor_hidden_dims: Tuple[int, ...] = (512, 512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 512, 256, 128)
    activation: str = "silu"            # TWIST: activation='silu' (vs HDMI: 'relu')
    layer_norm: bool = True             # TWIST: layer_norm=True

    # ==================== Other Settings ====================
    value_norm: bool = False
    compile: bool = True
    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (OBS_KEY, OBS_PRIV_KEY)

    # Training iteration counter (for std schedule)
    current_iteration: int = 0


cs = ConfigStore.instance()
cs.store("ppo_twistv2", node=PPOTwistV2Config, group="algo")


def make_mlp_twist(
    dims: Tuple[int, ...],
    activation: str = "silu",
    layer_norm: bool = True,
    output_activation: bool = False
) -> nn.Sequential:
    """创建符合TWIST风格的MLP

    Args:
        dims: 层维度，例如 [512, 512, 256, 128]
        activation: 激活函数类型 ('silu', 'relu', 'elu')
        layer_norm: 是否使用LayerNorm
        output_activation: 最后一层是否使用激活函数
    """
    layers = []

    # 激活函数映射
    act_fn = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "elu": nn.ELU,
    }[activation.lower()]

    for i in range(len(dims) - 1):
        layers.append(nn.LazyLinear(dims[i]))

        # 添加 LayerNorm (在激活函数之前)
        if layer_norm:
            layers.append(nn.LayerNorm(dims[i]))

        # 添加激活函数（最后一层可选）
        if i < len(dims) - 2 or output_activation:
            layers.append(act_fn())

    # 最后一层
    layers.append(nn.LazyLinear(dims[-1]))
    if layer_norm and output_activation:
        layers.append(nn.LayerNorm(dims[-1]))
    if output_activation:
        layers.append(act_fn())

    return nn.Sequential(*layers)


class ActorTwist(nn.Module):
    """TWIST-style Actor with action std schedule"""

    def __init__(
        self,
        action_dim: int,
        init_noise_scale: float = 1.0,
        final_noise_scale: float = 0.4,
        load_noise_scale: float | None = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.init_noise_scale = init_noise_scale
        self.final_noise_scale = final_noise_scale
        self.current_noise_scale = init_noise_scale if load_noise_scale is None else load_noise_scale

        # Action mean
        self.loc = nn.LazyLinear(action_dim)

        # Action std (learnable, but scaled by schedule)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        loc = self.loc(x)  # [batch_size, action_dim]

        # TWIST-style: std = current_noise_scale * exp(log_std)
        # 确保scale有正确的batch维度: [batch_size, action_dim]
        exp_log_std = torch.exp(self.log_std)  # [action_dim]
        scale = self.current_noise_scale * exp_log_std.expand(loc.shape[0], -1)  # [batch_size, action_dim]

        return loc, scale

    def update_noise_scale(self, new_scale: float):
        """更新action std scale (根据schedule)"""
        self.current_noise_scale = new_scale


class PPOTwistV2Policy(TensorDictModuleBase):
    """PPO Policy完全对齐TWIST-master超参数"""

    def __init__(
        self,
        cfg: PPOTwistV2Config,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        if isinstance(cfg, PPOTwistV2Config):
            self.cfg = cfg
        else:
            self.cfg = PPOTwistV2Config(**cfg)
        self.device = device

        # PPO hyperparameters (TWIST-aligned)
        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = self.cfg.max_grad_norm
        self.desired_kl = self.cfg.desired_kl
        self.clip_param = self.cfg.clip_param
        self.value_loss_coef = self.cfg.value_loss_coef
        self.use_clipped_value_loss = self.cfg.use_clipped_value_loss

        self.critic_loss_fn = nn.MSELoss(reduction="none")
        # action_spec可能是CompositeSpec，需要从中提取action的shape
        if isinstance(action_spec, CompositeSpec) and "action" in action_spec:
            self.action_dim = action_spec["action"].shape[-1]
        else:
            self.action_dim = action_spec.shape[-1]

        # GAE with TWIST gamma
        self.gae = GAE(self.cfg.gamma, self.cfg.lam)

        # Value normalization
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        # Training iteration counter (for std schedule)
        self.current_iteration = self.cfg.current_iteration

        fake_input = observation_spec.zero()

        # ==================== Actor Network (TWIST-style) ====================
        # Architecture: [512, 512, 256, 128] with SiLU activation + LayerNorm
        actor_module = TensorDictSequential(
            TensorDictModule(
                make_mlp_twist(
                    self.cfg.actor_hidden_dims,
                    activation=self.cfg.activation,
                    layer_norm=self.cfg.layer_norm,
                    output_activation=True  # TWIST: 最后一层也有激活
                ),
                [OBS_KEY],
                ["_actor_feature"]
            ),
            TensorDictModule(
                ActorTwist(
                    self.action_dim,
                    init_noise_scale=self.cfg.init_noise_scale,
                    final_noise_scale=self.cfg.final_noise_scale,
                    load_noise_scale=self.cfg.load_noise_scale
                ),
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

        # ==================== Critic Network (TWIST-style) ====================
        # Architecture: [512, 512, 256, 128] with SiLU activation + LayerNorm
        self.critic = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "_critic_input"),
            TensorDictModule(
                make_mlp_twist(
                    self.cfg.critic_hidden_dims,
                    activation=self.cfg.activation,
                    layer_norm=self.cfg.layer_norm,
                    output_activation=True
                ),
                ["_critic_input"],
                ["_critic_feature"]
            ),
            TensorDictModule(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
        ).to(self.device)

        # Initialize networks
        self.actor(fake_input)
        self.critic(fake_input)

        # ==================== Optimizer (TWIST learning rate) ====================
        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr
        )

        # Orthogonal initialization (TWIST-style)
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)

        self.actor.apply(init_)
        self.critic.apply(init_)

        # Distributed training support
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
        """训练操作，对齐TWIST的训练流程"""
        tensordict = tensordict.exclude("stats")
        infos = []

        # Compute advantage and returns
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        # TWIST: 5 epochs (vs HDMI: 3)
        for epoch in range(self.cfg.ppo_epochs):
            # TWIST: 4 mini-batches (vs HDMI: 8)
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self.update(minibatch))

                # TWIST adaptive learning rate schedule based on KL divergence
                if self.cfg.schedule == "adaptive" and self.desired_kl is not None:
                    kl = infos[-1]["actor/kl"]
                    actor_lr = self.opt.param_groups[0]["lr"]

                    # TWIST logic: adjust LR based on KL
                    if kl > self.desired_kl * 2.0:
                        actor_lr = max(1e-5, actor_lr / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        actor_lr = min(1e-2, actor_lr * 1.5)

                    self.opt.param_groups[0]["lr"] = actor_lr
                    self.opt.param_groups[1]["lr"] = actor_lr  # Critic LR也跟着调整

        # Update action std schedule (TWIST-style)
        self._update_std_schedule()

        # Increment iteration counter
        self.current_iteration += 1

        # Aggregate statistics
        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["actor/noise_scale"] = self.get_current_noise_scale()
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_std"] = tensordict["ret"].std().item()
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        infos["training/iteration"] = self.current_iteration

        return dict(sorted(infos.items()))

    def _update_std_schedule(self):
        """更新action std schedule (TWIST-style)

        TWIST std_schedule = [init_std, final_std, warmup_steps, hold_steps]
        例如: [1.0, 0.4, 4000, 1500]

        逻辑：
        - iter < warmup_steps: 从 init_std 线性衰减到 final_std
        - warmup_steps <= iter < warmup_steps + hold_steps: 保持 final_std
        - iter >= warmup_steps + hold_steps: 保持 final_std
        """
        warmup = self.cfg.std_warmup_steps
        hold = self.cfg.std_hold_steps
        init_std = self.cfg.init_noise_scale
        final_std = self.cfg.final_noise_scale

        if self.current_iteration < warmup:
            # Linear decay
            progress = self.current_iteration / warmup
            new_std = init_std + (final_std - init_std) * progress
        else:
            # Hold at final_std
            new_std = final_std

        # Update actor's noise scale
        # ProbabilisticActor -> ModuleList[0] -> TensorDictSequential -> ModuleList[1] -> TensorDictModule -> ActorTwist
        actor_twist = self.actor.module[0].module[1].module
        actor_twist.update_noise_scale(new_std)

    def get_current_noise_scale(self) -> float:
        """获取当前的noise scale"""
        actor_twist = self.actor.module[0].module[1].module
        return actor_twist.current_noise_scale

    @torch.no_grad()
    def _compute_advantage(
        self,
        tensordict: TensorDict,
        critic: TensorDictModule,
        adv_key: str="adv",
        ret_key: str="ret",
        update_value_norm: bool=True,
    ):
        """计算优势函数和回报（使用TWIST的gamma=0.998）"""
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(tensordict_flat)
                critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True).clamp_min(0.)
        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        # GAE with gamma=0.998 (TWIST)
        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        """单次PPO更新（一个mini-batch）"""
        dist: IndependentNormal = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        valid = (~tensordict["is_init"])
        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)

        # PPO clipped objective
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - (torch.min(surr1, surr2) * valid).mean()
        entropy_loss = - self.entropy_coef * entropy

        # Value loss (TWIST: use_clipped_value_loss=True)
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]

        if self.use_clipped_value_loss:
            # Clipped value loss (TWIST-style)
            values_old = tensordict["state_value"].detach()
            values_clipped = values_old + (values - values_old).clamp(-self.clip_param, self.clip_param)
            value_loss_1 = self.critic_loss_fn(b_returns, values)
            value_loss_2 = self.critic_loss_fn(b_returns, values_clipped)
            value_loss = torch.max(value_loss_1, value_loss_2)
            value_loss = (value_loss * valid).mean()
        else:
            value_loss = self.critic_loss_fn(b_returns, values)
            value_loss = (value_loss * valid).mean()

        # Total loss
        loss = policy_loss + entropy_loss + self.value_loss_coef * value_loss

        # Backward and optimize
        self.opt.zero_grad()
        loss.backward()

        # Distributed gradient sync
        if active_adaptation.is_distributed():
            for param in self.actor.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size
            for param in self.critic.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size

        # Gradient clipping
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()

        # Logging metrics
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

            # KL divergence (for adaptive LR)
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
        """保存模型状态（包括iteration counter）"""
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.opt.state_dict(),
            "value_norm": self.value_norm.state_dict(),
            "current_iteration": self.current_iteration,
        }
        return state

    def load_state_dict(self, state_dict):
        """加载模型状态"""
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.opt.load_state_dict(state_dict["optimizer"])
        self.value_norm.load_state_dict(state_dict["value_norm"])
        self.current_iteration = state_dict.get("current_iteration", 0)
