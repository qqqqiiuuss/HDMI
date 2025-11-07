"""
Dual Command Manager for TWIST + HDMI Integration

This module provides a dual command manager system that maintains both:
1. TWIST command manager for frozen teacher policy observations
2. HDMI command manager for student policy training

Both managers share the same motion data but provide different interfaces
and observation formats.

Usage:
    dual_manager = DualCommandManager(
        env=env,
        hdmi_config=hdmi_cfg,
        twist_config=twist_cfg,
        shared_motion_path="data/motion/g1/omomo/sub1_suitcase_011"
    )
"""

import torch
from typing import Optional
from omegaconf import DictConfig
from dataclasses import dataclass

from active_adaptation.envs.mdp.commands.hdmi.command import RobotObjectTracking
from active_adaptation.envs.mdp.commands.twist.command import TwistMotionTracking


class DualCommandManager:
    """
    双命令管理器

    在同一个环境中维护两个独立的 command manager：
    - HDMI command manager: 用于 HDMI student policy 的训练
    - TWIST command manager: 用于 TWIST teacher policy 的推理

    两个 manager 共享相同的 motion 数据文件，但提供不同的接口。

    Args:
        env: 环境实例
        hdmi_config: HDMI command manager 配置
        twist_config: TWIST command manager 配置
        shared_motion_path: 共享的 motion 数据路径
    """

    def __init__(
        self,
        env,
        hdmi_config: DictConfig,
        twist_config: DictConfig,
        shared_motion_path: Optional[str] = None
    ):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        # 如果指定了共享路径，覆盖两个配置的路径
        if shared_motion_path is not None:
            hdmi_config.data_path = shared_motion_path
            twist_config.data_path = shared_motion_path

        # 创建两个独立的 command manager
        print("[DualCommandManager] Creating HDMI command manager...")
        self.hdmi_manager = self._create_hdmi_manager(env, hdmi_config)

        print("[DualCommandManager] Creating TWIST command manager...")
        self.twist_manager = self._create_twist_manager(env, twist_config)

        print(f"[DualCommandManager] Initialized with shared motion: {shared_motion_path}")
        print(f"[DualCommandManager] HDMI manager type: {type(self.hdmi_manager).__name__}")
        print(f"[DualCommandManager] TWIST manager type: {type(self.twist_manager).__name__}")

    def _create_hdmi_manager(self, env, config: DictConfig):
        """创建 HDMI command manager"""
        # 使用 hydra 实例化
        from hydra.utils import instantiate
        manager = instantiate(config, env=env)
        return manager

    def _create_twist_manager(self, env, config: DictConfig):
        """创建 TWIST command manager"""
        from hydra.utils import instantiate
        manager = instantiate(config, env=env)
        return manager

    def reset(self, env_ids: torch.Tensor):
        """
        重置指定环境的两个 command manager

        Args:
            env_ids: 要重置的环境 ID
        """
        self.hdmi_manager.reset(env_ids)
        self.twist_manager.reset(env_ids)

    def update(self):
        """更新两个 command manager"""
        self.hdmi_manager.update()
        self.twist_manager.update()

    def get_hdmi_observations(self):
        """获取 HDMI 观察"""
        # HDMI observations 通常通过环境的 observation manager 获取
        # 这里只是一个占位符
        pass

    def get_twist_observations(self):
        """获取 TWIST 观察"""
        # TWIST observations 需要特殊构建
        # 在 DualObservationBuilder 中实现
        pass

    @property
    def current_motion_time(self):
        """当前运动时间（使用 HDMI manager 的时间）"""
        return self.hdmi_manager.t if hasattr(self.hdmi_manager, 't') else None

    def __getattr__(self, name):
        """
        属性访问代理

        默认访问 HDMI manager 的属性（向后兼容）
        如果需要访问 TWIST manager，使用 twist_manager.xxx
        """
        if name in ['hdmi_manager', 'twist_manager', 'env', 'num_envs', 'device']:
            return object.__getattribute__(self, name)

        # 默认代理到 HDMI manager
        try:
            return getattr(self.hdmi_manager, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class TwistObservationAdapter:
    """
    TWIST 观察适配器

    负责从 TWIST command manager 构建 TWIST policy 所需的观察。
    这个类桥接了 TWIST command manager 和观察函数。

    Args:
        env: 环境实例
        twist_command_manager: TWIST 命令管理器
        cfg: TWIST 观察配置
    """

    def __init__(
        self,
        env,
        twist_command_manager,
        cfg: Optional[DictConfig] = None
    ):
        self.env = env
        self.command_manager = twist_command_manager
        self.cfg = cfg or {}

        # 初始化观察函数
        self.obs_functions = {}
        self._init_observations()

    def _init_observations(self):
        """初始化 TWIST 观察函数（延迟导入避免注册冲突）"""
        # 不要在模块级别导入 twist.observations，而是在这里动态导入
        # 这样可以避免与 hdmi.observations 的类名冲突

        # 动态导入 TWIST observations 模块
        import importlib
        twist_obs_module = importlib.import_module(
            'active_adaptation.envs.mdp.commands.twist.observations'
        )

        # 从配置中读取参数，如果没有则使用默认值
        proprio_cfg = self.cfg.get("proprio_history_combined", {})
        ref_motion_cfg = self.cfg.get("ref_motion_windowed", {})

        # 获取观察类（通过字符串查找，避免直接导入）
        proprio_history_cls = getattr(twist_obs_module, 'proprio_history_combined')
        ref_motion_windowed_cls = getattr(twist_obs_module, 'ref_motion_windowed')

        # CRITICAL: 临时替换 env.command_manager 为 TWIST manager
        # 观察函数在初始化时会从 env.command_manager 读取配置
        # 获取实际的环境对象（可能是 base_env 或者 env 本身）
        actual_env = getattr(self.env, 'base_env', self.env)
        original_command_manager = actual_env.command_manager
        actual_env.command_manager = self.command_manager

        try:
            # 创建 proprio history 观察 (TWIST特有)
            self.obs_functions["proprio_history_combined"] = proprio_history_cls(
                env=self.env,
                history_length=proprio_cfg.get("history_length", 11),
                root_ori_noise=0.0,  # 推理时不加噪声
                root_ang_vel_noise=0.0,
                joint_pos_noise=0.0,
                joint_vel_noise=0.0,
                action_noise=0.0,
                noise_increasing_steps=proprio_cfg.get("noise_increasing_steps", 3000)
            )

            # 创建 ref_motion_windowed 观察 (使用 TWIST command manager)
            self.obs_functions["ref_motion_windowed"] = ref_motion_windowed_cls(
                env=self.env,
                past_frames=ref_motion_cfg.get("past_frames", 10),
                future_frames=ref_motion_cfg.get("future_frames", 10),
                coordinate_frame=ref_motion_cfg.get("coordinate_frame", "world"),
                ref_root_pos_noise=0.0,
                ref_root_ori_noise=0.0,
                ref_joint_pos_noise=0.0
            )
        finally:
            # 恢复原始 command manager
            actual_env.command_manager = original_command_manager

        print(f"[TwistObservationAdapter] Initialized {len(self.obs_functions)} observation functions")
        print(f"[TwistObservationAdapter] Using TWIST command manager with {len(self.command_manager.tracking_joint_names)} joints")

    def reset(self, env_ids: torch.Tensor):
        """重置观察函数"""
        # 临时替换 command_manager
        actual_env = getattr(self.env, 'base_env', self.env)
        original_command_manager = actual_env.command_manager
        actual_env.command_manager = self.command_manager
        try:
            for obs_fn in self.obs_functions.values():
                if hasattr(obs_fn, 'reset'):
                    obs_fn.reset(env_ids)
        finally:
            actual_env.command_manager = original_command_manager

    def update(self):
        """更新所有观察函数"""
        # 临时替换 command_manager
        actual_env = getattr(self.env, 'base_env', self.env)
        original_command_manager = actual_env.command_manager
        actual_env.command_manager = self.command_manager
        try:
            for obs_fn in self.obs_functions.values():
                if hasattr(obs_fn, 'update'):
                    obs_fn.update()
        finally:
            actual_env.command_manager = original_command_manager

    def compute(self):
        """
        计算所有观察并返回字典

        Returns:
            观察字典 {obs_name: obs_tensor}
        """
        # 临时替换 command_manager
        actual_env = getattr(self.env, 'base_env', self.env)
        original_command_manager = actual_env.command_manager
        actual_env.command_manager = self.command_manager
        try:
            obs_dict = {}
            for name, obs_fn in self.obs_functions.items():
                obs_dict[name] = obs_fn.compute()
            return obs_dict
        finally:
            actual_env.command_manager = original_command_manager

    def get_observation_tensor(self):
        """
        获取拼接后的观察张量

        Returns:
            拼接的观察张量 [num_envs, total_obs_dim]
        """
        obs_dict = self.compute()
        obs_list = [obs_dict[name] for name in sorted(obs_dict.keys())]
        return torch.cat(obs_list, dim=-1)
