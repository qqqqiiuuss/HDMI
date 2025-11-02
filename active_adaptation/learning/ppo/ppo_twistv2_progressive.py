"""
Progressive PPOTwistV2 - 渐进式网络架构

策略：
1. 初期使用类似HDMI PPO的简单网络
2. 逐步增加网络深度
3. 最终达到TWIST的网络架构

这样既能保持初期稳定性，又能获得后期的性能优势。
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
from .ppo_twistv2 import PPOTwistV2Policy, ActorTwist, make_mlp_twist

torch.set_float32_matmul_precision('high')

import active_adaptation
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class PPOTwistV2ProgressiveConfig:
    """渐进式PPOTwistV2配置"""
    _target_: str = "active_adaptation.learning.ppo.ppo_twistv2_progressive.PPOTwistV2ProgressivePolicy"
    name: str = "ppo_twistv2_progressive"

    # ==================== 基础参数 (与PPOTwistV2相同) ====================
    train_every: int = 24
    ppo_epochs: int = 5
    num_minibatches: int = 4
    lr: float = 2e-4
    clip_param: float = 0.2
    gamma: float = 0.998
    lam: float = 0.95
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    value_loss_coef: float = 1.0
    desired_kl: float = 0.008
    schedule: str = "adaptive"
    compile: bool = True

    # ==================== 渐进式网络配置 ====================
    # 阶段1: 类似HDMI PPO (稳定训练)
    stage1_epochs: int = 5000
    stage1_actor_dims: Tuple[int, ...] = (512, 256, 256)
    stage1_critic_dims: Tuple[int, ...] = (256, 256, 128)
    stage1_activation: str = "relu"  # 使用ReLU
    stage1_layer_norm: bool = False

    # 阶段2: 中等网络
    stage2_epochs: int = 10000
    stage2_actor_dims: Tuple[int, ...] = (512, 512, 256)
    stage2_critic_dims: Tuple[int, ...] = (512, 256, 128)
    stage2_activation: str = "silu"  # 开始使用SiLU
    stage2_layer_norm: bool = True

    # 阶段3: 完整TWIST网络
    stage3_start_epoch: int = 15000  # 开始使用完整网络
    stage3_actor_dims: Tuple[int, ...] = (512, 512, 256, 128)
    stage3_critic_dims: Tuple[int, ...] = (512, 512, 256, 128)
    stage3_activation: str = "silu"
    stage3_layer_norm: bool = True

    # ==================== 渐进式Action Std ====================
    stage1_init_noise: float = 1.2  # 比PPOTwistV2更保守
    stage1_final_noise: float = 0.6

    stage2_init_noise: float = 1.0
    stage2_final_noise: float = 0.5

    stage3_init_noise: float = 1.0  # PPOTwistV2原始值
    stage3_final_noise: float = 0.4

    # ==================== 渐进式熵系数 ====================
    stage1_entropy: float = 0.001    # HDMI级别
    stage2_entropy: float = 0.002    # 中等
    stage3_entropy: float = 0.005    # TWIST级别

    # ==================== 其他参数 ====================
    value_norm: bool = False
    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (OBS_KEY, OBS_PRIV_KEY)

cs = ConfigStore.instance()
cs.store("ppo_twistv2_progressive", node=PPOTwistV2ProgressiveConfig, group="algo")


class ProgressiveActor(nn.Module):
    """渐进式Actor，支持动态更换网络架构"""

    def __init__(
        self,
        action_dim: int,
        stages_config: dict,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.stages_config = stages_config
        self.current_stage = 1

        # 创建所有阶段的网络
        self.networks = {}
        for stage, config in stages_config.items():
            self.networks[stage] = TensorDictModule(
                make_mlp_twist(
                    config['actor_dims'],
                    activation=config['activation'],
                    layer_norm=config['layer_norm'],
                    output_activation=True
                ),
                ["_actor_feature"],
                ["_actor_feature"]
            )

        # ActorTwist组件（用于输出）
        self.actor_twist = ActorTwist(
            action_dim=action_dim,
            init_noise_scale=stages_config[1]['init_noise'],
            final_noise_scale=stages_config[1]['final_noise'],
        )

        # 当前使用的网络
        self.current_network = self.networks[1]

    def get_stage_config(self, iteration: int) -> int:
        """根据iteration确定当前阶段"""
        if iteration < self.stages_config[1]['epochs']:
            return 1
        elif iteration < self.stages_config[2]['epochs']:
            return 2
        else:
            return 3

    def set_stage(self, stage: int):
        """设置当前阶段的网络"""
        if stage in self.networks:
            self.current_network = self.networks[stage]
            self.current_stage = stage

            # 更新ActorTwist的噪声参数
            if 'init_noise' in self.stages_config[stage]:
                self.actor_twist.init_noise_scale = self.stages_config[stage]['init_noise']
            if 'final_noise' in self.stages_config[stage]:
                self.actor_twist.final_noise_scale = self.stages_config[stage]['final_noise']

    def forward(self, x):
        """前向传播"""
        # 通过当前阶段网络
        features = self.current_network(x)["_actor_feature"]

        # 通过ActorTwist输出
        return self.actor_twist(features)


class PPOTwistV2ProgressivePolicy(TensorDictModuleBase):
    """渐进式PPOTwistV2 Policy"""

    def __init__(
        self,
        cfg: PPOTwistV2ProgressiveConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        if isinstance(cfg, PPOTwistV2ProgressiveConfig):
            self.cfg = cfg
        else:
            self.cfg = PPOTwistV2ProgressiveConfig(**cfg)
        self.device = device

        # 渐进式网络配置
        self.stages_config = {
            1: {
                'epochs': self.cfg.stage1_epochs,
                'actor_dims': self.cfg.stage1_actor_dims,
                'critic_dims': self.cfg.stage1_critic_dims,
                'activation': self.cfg.stage1_activation,
                'layer_norm': self.cfg.stage1_layer_norm,
                'init_noise': self.cfg.stage1_init_noise,
                'final_noise': self.cfg.stage1_final_noise,
                'entropy': self.cfg.stage1_entropy,
            },
            2: {
                'epochs': self.cfg.stage2_epochs,
                'actor_dims': self.cfg.stage2_actor_dims,
                'critic_dims': self.cfg.stage2_critic_dims,
                'activation': self.cfg.stage2_activation,
                'layer_norm': self.cfg.stage2_layer_norm,
                'init_noise': self.cfg.stage2_init_noise,
                'final_noise': self.cfg.stage2_final_noise,
                'entropy': self.cfg.stage2_entropy,
            },
            3: {
                'epochs': self.cfg.stage3_start_epoch,  # 不使用，只是标识
                'actor_dims': self.cfg.stage3_actor_dims,
                'critic_dims': self.cfg.stage3_critic_dims,
                'activation': self.cfg.stage3_activation,
                'layer_norm': self.cfg.stage3_layer_norm,
                'init_noise': self.cfg.stage3_init_noise,
                'final_noise': self.cfg.stage3_final_noise,
                'entropy': self.cfg.stage3_entropy,
            }
        }

        # 基础参数
        self.entropy_coef = self.cfg.stage1_entropy
        self.max_grad_norm = self.cfg.max_grad_norm
        self.desired_kl = self.cfg.desired_kl
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")

        # action_dim
        if isinstance(action_spec, CompositeSpec) and "action" in action_spec:
            self.action_dim = action_spec["action"].shape[-1]
        else:
            self.action_dim = action_spec.shape[-1]

        # GAE
        self.gae = GAE(self.cfg.gamma, self.cfg.lam)

        # Value normalization
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        # Training iteration counter
        self.current_iteration = 0

        fake_input = observation_spec.zero()

        # 渐进式Actor
        self.actor = ProgressiveActor(
            action_dim=self.action_dim,
            stages_config=self.stages_config
        ).to(self.device)

        # ProbabilisticActor
        self.probabilistic_actor: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(self.actor),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        # 渐进式Critic
        self.critic_networks = {}
        for stage, config in self.stages_config.items():
            self.critic_networks[stage] = TensorDictSequential(
                CatTensors([OBS_KEY, OBS_PRIV_KEY], "_critic_input"),
                TensorDictModule(
                    make_mlp_twist(
                        config['critic_dims'],
                        activation=config['activation'],
                        layer_norm=config['layer_norm'],
                        output_activation=True
                    ),
                    ["_critic_input"],
                    ["_critic_feature"]
                ),
                TensorDictModule(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
            ).to(self.device)

        self.critic = self.critic_networks[1]

        # 初始化
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

        # 初始化
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)

        self.actor.apply(init_)
        self.critic.apply(init_)

        # Distributed training
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

    def _update_progressive_stage(self):
        """根据训练进度更新网络架构"""
        current_stage = self.actor.get_stage_config(self.current_iteration)

        if current_stage != self.actor.current_stage:
            print(f"🔄 Progressive Stage {self.actor.current_stage} → {current_stage}")
            print(f"   Iteration: {self.current_iteration}")
            print(f"   Actor dims: {self.stages_config[current_stage]['actor_dims']}")
            print(f"   Critic dims: {self.stages_config[current_stage]['critic_dims']}")
            print(f"   Activation: {self.stages_config[current_stage]['activation']}")
            print(f"   Layer norm: {self.stages_config[current_stage]['layer_norm']}")
            print(f"   Entropy coef: {self.stages_config[current_stage]['entropy']}")

            # 更新网络
            self.actor.set_stage(current_stage)
            self.critic = self.critic_networks[current_stage]

            # 更新熵系数
            self.entropy_coef = self.stages_config[current_stage]['entropy']

            # 重新初始化优化器（学习率调整）
            self.opt = torch.optim.Adam(
                [
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=self.cfg.lr
            )

    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(self.actor, self.probabilistic_actor)
        if self.cfg.compile:
            policy = torch.compile(policy)
        return policy

    def train_op(self, tensordict: TensorDict):
        """训练操作，支持渐进式网络更新"""
        # 检查是否需要更新网络架构
        self._update_progressive_stage()

        tensordict = tensordict.exclude("stats")
        infos = []

        # 计算advantage
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        # PPO更新
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self.update(minibatch))

                # 自适应学习率
                if self.cfg.schedule == "adaptive" and self.desired_kl is not None:
                    kl = infos[-1]["actor/kl"]
                    actor_lr = self.opt.param_groups[0]["lr"]
                    if kl > self.desired_kl * 2.0:
                        actor_lr = max(1e-5, actor_lr / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        actor_lr = min(1e-2, actor_lr * 1.5)
                    self.opt.param_groups[0]["lr"] = actor_lr
                    self.opt.param_groups[1]["lr"] = actor_lr

        # 更新iteration计数
        self.current_iteration += 1

        # 聚合统计信息
        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["actor/entropy_coef"] = self.entropy_coef
        infos["actor/current_stage"] = self.actor.current_stage
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_std"] = tensordict["ret"].std().item()
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        infos["training/iteration"] = self.current_iteration

        return dict(sorted(infos.items()))

    def _compute_advantage(self, tensordict: TensorDict, critic: TensorDictModule, adv_key: str="adv", ret_key: str="ret", update_value_norm: bool=True):
        """计算优势函数"""
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

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        """单次PPO更新"""
        dist: IndependentNormal = self.probabilistic_actor.get_dist(tensordict)
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

        # Value loss
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]

        if self.cfg.use_clipped_value_loss:
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
        loss = policy_loss + entropy_loss + self.cfg.value_loss_coef * value_loss

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

            # KL divergence
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