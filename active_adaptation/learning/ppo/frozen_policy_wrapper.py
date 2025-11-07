"""
Frozen Policy Wrapper for Reference Action Generation

This module provides a wrapper for using a frozen (non-trainable) policy as a reference
action provider in residual learning scenarios. Specifically designed for integrating
TWIST teacher policies into HDMI training.

Key Features:
- Load pre-trained policy checkpoints
- Freeze all parameters (no gradient computation)
- Build policy-specific observations on-the-fly
- Output reference actions for residual learning

Usage:
    frozen_policy = FrozenPolicyWrapper(
        checkpoint_path="run:hdmi/abc123",
        policy_task_cfg=twist_cfg,
        device="cuda:0"
    )
    ref_action = frozen_policy(env_state)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from tensordict import TensorDict
from omegaconf import DictConfig, OmegaConf
import os

from active_adaptation.utils.wandb import parse_checkpoint_path


class FrozenPolicyWrapper(nn.Module):
    """
    包装器：冻结的策略用于生成参考动作

    这个类加载一个预训练的策略（例如 TWIST teacher），冻结其参数，
    并在每个 step 根据环境状态生成参考动作。

    Args:
        checkpoint_path: 策略 checkpoint 路径 (支持 wandb 格式: "run:project/run_id")
        policy_obs_builder: 观察构建函数，将环境状态转换为策略所需的观察
        device: 计算设备

    Attributes:
        policy: 冻结的策略模块
        obs_builder: 观察构建函数
    """

    def __init__(
        self,
        checkpoint_path: str,
        policy_obs_builder: Optional[callable] = None,
        device: str = "cuda:0",
        observation_spec: Optional[Any] = None,
        action_spec: Optional[Any] = None
    ):
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.obs_builder = policy_obs_builder
        self.observation_spec = observation_spec
        self.action_spec = action_spec

        # 加载策略
        self.policy = self._load_policy(checkpoint_path)

        # 检查是否已经是完整的actor模块（不是PolicyStateHolder）
        if isinstance(self.policy, nn.Module) and hasattr(self.policy, '__call__'):
            # 检查是否是PolicyStateHolder
            if type(self.policy).__name__ == 'PolicyStateHolder':
                # 需要rebuild
                self._policy_rebuilt = False
                print(f"[FrozenPolicyWrapper] Loaded policy state_dict from {checkpoint_path}")
                print(f"[FrozenPolicyWrapper] Will rebuild policy later with observation/action specs")
            else:
                # 如果直接加载到了actor，冻结参数
                self._freeze_parameters()
                self._policy_rebuilt = True
                print(f"[FrozenPolicyWrapper] Loaded complete actor module from checkpoint")
                print(f"[FrozenPolicyWrapper] Total parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        else:
            # 否则需要在外部rebuild
            self._policy_rebuilt = False
            print(f"[FrozenPolicyWrapper] Loaded policy state_dict from {checkpoint_path}")

    def _load_policy(self, checkpoint_path: str) -> nn.Module:
        """
        从 checkpoint 加载策略

        Args:
            checkpoint_path: checkpoint 路径

        Returns:
            加载的策略模块
        """
        # 解析 checkpoint 路径（支持 wandb 格式）
        parsed_path = parse_checkpoint_path(checkpoint_path)

        if parsed_path is None:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")

        if not os.path.exists(parsed_path):
            raise FileNotFoundError(f"Checkpoint not found: {parsed_path}")

        # 加载 checkpoint
        print(f"[FrozenPolicyWrapper] Loading checkpoint from {parsed_path}")
        state_dict = torch.load(parsed_path, map_location=self.device, weights_only=False)

        if "policy" not in state_dict:
            raise KeyError("Checkpoint does not contain 'policy' key")

        # 提取策略
        policy_data = state_dict["policy"]

        # 检查policy_data是什么类型
        # 检查是否包含模块对象（如'actor', 'critic'）或state_dict
        if isinstance(policy_data, dict):
            # 打印类型信息用于调试
            if 'actor' in policy_data:
                actor_type = type(policy_data['actor'])
                print(f"[FrozenPolicyWrapper] actor type: {actor_type}")

                # 检查是否是nn.Module实例
                if isinstance(policy_data['actor'], nn.Module):
                    print(f"[FrozenPolicyWrapper] Found actor module in checkpoint")
                    # 直接使用actor模块
                    actor = policy_data['actor']
                    return actor.to(self.device)
                else:
                    # actor是OrderedDict或state_dict
                    print(f"[FrozenPolicyWrapper] actor is state_dict (OrderedDict)")
                    # 直接返回包含state_dict的holder
                    # 稍后在rebuild_ppo_policy中加载
                    class PolicyStateHolder(nn.Module):
                        """临时holder，存储state_dict"""
                        def __init__(self, state_dict):
                            super().__init__()
                            self._state_dict = state_dict

                        def forward(self, x):
                            raise NotImplementedError(
                                "PolicyStateHolder should not be called. "
                                "Use rebuild_ppo_policy() first."
                            )

                    holder = PolicyStateHolder(policy_data)
                    return holder.to(self.device)
            else:
                # 没有actor key，尝试重建
                policy = self._rebuild_policy_from_state_dict(policy_data, state_dict)
                return policy
        else:
            # policy_data本身就是一个模块
            return policy_data.to(self.device)

    def _rebuild_policy_from_state_dict(
        self,
        policy_state_dict: Dict[str, torch.Tensor],
        full_state_dict: Dict[str, Any]
    ) -> nn.Module:
        """
        从状态字典重建策略网络

        对于PPO checkpoint，我们需要重建完整的PPO policy然后只使用actor部分。

        Args:
            policy_state_dict: 策略的状态字典
            full_state_dict: 完整的 checkpoint 字典

        Returns:
            重建的完整policy对象
        """
        print("[FrozenPolicyWrapper] Rebuilding PPO policy from checkpoint...")

        # 先打印checkpoint中的keys来调试
        all_keys = list(policy_state_dict.keys())
        print(f"[FrozenPolicyWrapper] Total keys in checkpoint: {len(all_keys)}")
        print(f"[FrozenPolicyWrapper] First 10 keys: {all_keys[:10]}")

        # 检查是否是PPO还是PPO-ROA
        has_actor = any(k.startswith("actor.") for k in policy_state_dict.keys())
        has_actor_adapt = any(k.startswith("actor_adapt.") for k in policy_state_dict.keys())

        if not has_actor and not has_actor_adapt:
            print(f"[FrozenPolicyWrapper] ERROR: No actor found!")
            print(f"[FrozenPolicyWrapper] Available key prefixes:")
            prefixes = set([k.split('.')[0] for k in all_keys if '.' in k])
            for prefix in sorted(prefixes):
                count = sum(1 for k in all_keys if k.startswith(prefix + '.'))
                print(f"  {prefix}: {count} keys")
            raise ValueError("No actor found in checkpoint")

        # 从checkpoint中提取配置信息（如果有的话）
        # 否则使用默认配置重建
        print(f"[FrozenPolicyWrapper] Found actor in checkpoint")
        print(f"[FrozenPolicyWrapper] Checkpoint keys sample: {list(policy_state_dict.keys())[:5]}")

        # 返回完整的state_dict，让调用者来重建policy
        # 这样可以避免在这里硬编码policy结构
        class PolicyStateHolder(nn.Module):
            """临时holder，存储state_dict，等待外部重建policy"""
            def __init__(self, state_dict):
                super().__init__()
                self._state_dict = state_dict

            def forward(self, x):
                raise NotImplementedError(
                    "PolicyStateHolder should not be called directly. "
                    "Use rebuild_from_state() to create the actual policy first."
                )

        holder = PolicyStateHolder(policy_state_dict)
        return holder.to(self.device)

    def rebuild_ppo_policy(self, observation_spec, action_spec, reward_spec):
        """
        重建完整的PPO policy

        Args:
            observation_spec: 观察空间spec
            action_spec: 动作空间spec
            reward_spec: 奖励空间spec
        """
        print(f"\n{'='*60}")
        print(f"[FrozenPolicyWrapper] rebuild_ppo_policy called")
        print(f"[FrozenPolicyWrapper] _policy_rebuilt={self._policy_rebuilt}")
        print(f"[FrozenPolicyWrapper] self.policy type: {type(self.policy).__name__}")
        print(f"{'='*60}\n")

        if self._policy_rebuilt:
            print("[FrozenPolicyWrapper] Policy already rebuilt, skipping")
            return

        print("[FrozenPolicyWrapper] Starting PPO policy rebuild...")

        # 导入必要的模块
        from torchrl.modules import ProbabilisticActor
        from tensordict.nn import TensorDictModule, TensorDictSequential
        from active_adaptation.learning.modules.distributions import IndependentNormal
        from active_adaptation.learning.ppo.common import Actor, make_mlp, OBS_KEY, ACTION_KEY

        # 从 state_dict 中提取 actor 部分
        state_dict = self.policy._state_dict
        print(f"[FrozenPolicyWrapper] Loading state_dict with keys: {list(state_dict.keys())}")

        if "actor" not in state_dict:
            raise KeyError("State dict does not contain 'actor' key")

        actor_state_dict = state_dict["actor"]

        # 从 state_dict 推断网络结构
        # actor 的第一层是 make_mlp，输入维度在 module.0.module.0.module.0.weight 中
        first_layer_key = "module.0.module.0.module.0.weight"
        if first_layer_key not in actor_state_dict:
            raise KeyError(f"Cannot find {first_layer_key} in actor state_dict")

        # weight shape is [out_features, in_features]
        input_dim = actor_state_dict[first_layer_key].shape[1]
        print(f"[FrozenPolicyWrapper] Inferred actor input dimension: {input_dim}")

        # 重建 actor 网络结构（必须与训练时完全一致）
        # 根据 ppo.py 中的结构：TensorDictSequential(
        #     TensorDictModule(make_mlp([512, 256, 256]), [OBS_KEY], ["_actor_feature"]),
        #     TensorDictModule(Actor(...), ["_actor_feature"], ["loc", "scale"])
        # )
        actor_module = TensorDictSequential(
            TensorDictModule(
                make_mlp([512, 256, 256]),
                [OBS_KEY],
                ["_actor_feature"]
            ),
            TensorDictModule(
                Actor(
                    action_dim=action_spec.shape[-1],
                    init_noise_scale=1.0,
                    load_noise_scale=0.5
                ),
                ["_actor_feature"],
                ["loc", "scale"]
            )
        ).to(self.device)

        # 创建 ProbabilisticActor
        actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        print(f"[FrozenPolicyWrapper] Created actor structure, loading state_dict...")

        # 加载 state_dict
        try:
            actor.load_state_dict(actor_state_dict, strict=True)
            print(f"[FrozenPolicyWrapper] ✓ Actor state_dict loaded successfully")
        except RuntimeError as e:
            print(f"[FrozenPolicyWrapper] Warning: {str(e)}")
            print(f"[FrozenPolicyWrapper] Trying with strict=False...")
            actor.load_state_dict(actor_state_dict, strict=False)

        # 保存重建的 actor
        self.policy = actor

        # 冻结所有参数
        print("[FrozenPolicyWrapper] Freezing parameters...")
        self._freeze_parameters()

        self._policy_rebuilt = True
        print(f"\n{'='*60}")
        print(f"[FrozenPolicyWrapper] ✓ PPO actor extracted and frozen")
        print(f"[FrozenPolicyWrapper] ✓ Total parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"[FrozenPolicyWrapper] ✓ Final policy type: {type(self.policy).__name__}")
        print(f"{'='*60}\n")

    def _freeze_parameters(self):
        """冻结所有参数"""
        for param in self.policy.parameters():
            param.requires_grad = False

        # 设置为评估模式
        self.policy.eval()

    def set_obs_builder(self, obs_builder: callable):
        """
        设置观察构建函数

        Args:
            obs_builder: 观察构建函数
        """
        self.obs_builder = obs_builder

    @torch.no_grad()
    def forward(self, obs: TensorDict) -> torch.Tensor:
        """
        前向传播：生成参考动作

        Args:
            obs: 观察 TensorDict（可以是环境的原始观察）

        Returns:
            参考动作张量 [num_envs, action_dim]
        """
        if not self._policy_rebuilt:
            print(f"[FrozenPolicyWrapper.forward] ERROR: Policy not rebuilt!")
            print(f"[FrozenPolicyWrapper.forward] _policy_rebuilt={self._policy_rebuilt}")
            print(f"[FrozenPolicyWrapper.forward] self.policy type: {type(self.policy).__name__}")
            raise RuntimeError(
                "Policy not rebuilt yet. Call rebuild_ppo_policy() first."
            )

        # 如果提供了观察构建函数，先转换观察
        if self.obs_builder is not None:
            obs = self.obs_builder(obs)

        # 使用策略生成动作
        # policy现在直接就是actor（ProbabilisticActor）
        with torch.no_grad():
            output = self.policy(obs)

        # 提取动作
        if isinstance(output, TensorDict):
            action = output.get("action", None)
            if action is None:
                raise KeyError("Policy output does not contain 'action' key")
        else:
            action = output

        return action

    def __call__(self, obs: TensorDict) -> torch.Tensor:
        """允许直接调用"""
        return self.forward(obs)


class DualObservationBuilder:
    """
    双观察构建器

    这个类负责为两个不同的策略构建各自所需的观察：
    1. TWIST teacher policy: 需要 proprio_history_combined + ref_motion_windowed
    2. HDMI student policy: 需要 HDMI 的标准观察

    在同一个环境中维护两套观察系统。
    """

    def __init__(
        self,
        env,
        twist_command_manager,
        hdmi_command_manager
    ):
        """
        Args:
            env: 环境实例
            twist_command_manager: TWIST 命令管理器
            hdmi_command_manager: HDMI 命令管理器
        """
        self.env = env
        self.twist_cmd = twist_command_manager
        self.hdmi_cmd = hdmi_command_manager

        # 存储 TWIST 观察函数实例
        self.twist_obs_functions = {}

        # 初始化 TWIST 观察函数
        self._init_twist_observations()

    def _init_twist_observations(self):
        """初始化 TWIST 观察函数"""
        from active_adaptation.envs.mdp.commands.twist.observations import (
            proprio_history_combined,
            ref_motion_windowed
        )

        # 创建观察函数实例
        # 注意：这些函数需要访问 TWIST command manager
        self.twist_obs_functions["proprio_history"] = proprio_history_combined(
            env=self.env,
            command_manager=self.twist_cmd,
            history_length=11,
            root_ori_noise=0.0,  # 推理时不加噪声
            root_ang_vel_noise=0.0,
            joint_pos_noise=0.0,
            joint_vel_noise=0.0,
            action_noise=0.0
        )

        self.twist_obs_functions["ref_motion_window"] = ref_motion_windowed(
            env=self.env,
            command_manager=self.twist_cmd,
            past_frames=10,
            future_frames=10,
            coordinate_frame='world',  # TWIST 使用 world frame
            ref_root_pos_noise=0.0,
            ref_root_ori_noise=0.0,
            ref_joint_pos_noise=0.0
        )

    def build_twist_observation(self) -> TensorDict:
        """
        构建 TWIST 策略所需的观察

        Returns:
            TWIST 观察 TensorDict
        """
        # 更新观察
        for obs_fn in self.twist_obs_functions.values():
            obs_fn.update()

        # 构建观察字典
        twist_obs = TensorDict({
            "proprio_history_combined": self.twist_obs_functions["proprio_history"].compute(),
            "ref_motion_windowed": self.twist_obs_functions["ref_motion_window"].compute()
        }, batch_size=[self.env.num_envs])

        return twist_obs

    def reset_twist_observations(self, env_ids):
        """重置 TWIST 观察（用于环境重置时）"""
        for obs_fn in self.twist_obs_functions.values():
            if hasattr(obs_fn, 'reset'):
                obs_fn.reset(env_ids)
