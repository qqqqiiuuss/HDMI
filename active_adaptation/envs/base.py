"""
强化学习环境基础框架

这个模块提供了构建机器人仿真环境的基础框架，支持 Isaac Lab 和 MuJoCo 两种物理引擎后端。
主要包含环境基类、观察组管理、奖励组管理等功能。

主要组件:
- ObsGroup: 观察组管理类，用于组合多个观察函数
- _Env: 环境基类，继承自 TorchRL 的 EnvBase
- RewardGroup: 奖励组管理类，用于组合多个奖励函数

作者: Active Adaptation Team
日期: 2024.10.10
版本: 支持多后端、模块化设计、性能监控
"""

import torch
import numpy as np
import hydra
# import inspect  # 暂时未使用
import re

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite, 
    Binary,
    UnboundedContinuous,
)
from collections import OrderedDict

from abc import abstractmethod
from typing import Dict
import time

import active_adaptation
import active_adaptation.envs.mdp as mdp
import active_adaptation.utils.symmetry as symmetry_utils

# 根据后端类型导入相应的模块
if active_adaptation.get_backend() == "isaac":
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import TiledCamera
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import UsdGeom


def parse_name_and_class(s: str):
    """
    解析配置字符串，提取名称和类名
    
    支持格式: "obs_name(ObsClass)" 或 "simple_name"
    
    Args:
        s: 配置字符串，格式为 "name(class)" 或 "name"
        
    Returns:
        tuple: (name, class_name) 或 (s, s) 如果解析失败
        
    Example:
        >>> parse_name_and_class("position_obs(PositionObservation)")
        ('position_obs', 'PositionObservation')
        >>> parse_name_and_class("simple_obs")
        ('simple_obs', 'simple_obs')
    """
    pattern = r'^(.+)\((.+)\)$'
    match = re.match(pattern, s)
    if match:
        name, cls = match.groups()
        return name, cls
    return s, s


class ObsGroup:
    """
    观察组管理类
    
    用于管理多个观察函数，将它们的结果拼接成一个统一的观察向量。
    支持延迟更新、规格自动生成、对称变换等功能。
    
    Attributes:
        name: 观察组名称
        funcs: 观察函数字典 {obs_name: obs_function}
        max_delay: 最大延迟步数
        timestamp: 当前时间戳
    """
    
    def __init__(
        self,
        name: str,
        funcs: Dict[str, mdp.Observation],
        max_delay: int = 0,
    ):
        """
        初始化观察组
        
        Args:
            name: 观察组名称
            funcs: 观察函数字典
            max_delay: 最大延迟步数（当前未使用）
        """
        self.name = name
        self.funcs = funcs
        self.max_delay = max_delay
        self.timestamp = -1

    @property
    def keys(self):
        """获取所有观察函数的键名"""
        return self.funcs.keys()

    @property
    def spec(self):
        """
        获取观察规格
        
        自动计算观察的形状和数据类型，并创建相应的规格对象。
        使用缓存机制避免重复计算。
        
        Returns:
            Composite: 观察规格对象
        """
        if not hasattr(self, "_spec"):
            # 通过实际计算获取形状和类型信息
            foo = self.compute({}, 0)
            spec = {}
            spec[self.name] = UnboundedContinuous(foo[self.name].shape, dtype=foo[self.name].dtype)
            self._spec = Composite(spec, shape=[foo[self.name].shape[0]]).to(foo[self.name].device)
        return self._spec

    def compute(self, tensordict: TensorDictBase, timestamp: int) -> torch.Tensor:
        """
        计算观察值并更新 tensordict
        
        Args:
            tensordict: 目标张量字典
            timestamp: 当前时间戳
            
        Returns:
            TensorDictBase: 更新后的张量字典
        """
        # torch.compiler.cudagraph_mark_step_begin()  # CUDA 图优化标记
        output = self._compute()
        tensordict[self.name] = output
        return tensordict
    
    # @torch.compile(mode="reduce-overhead")  # 可选：使用 torch.compile 优化
    def _compute(self) -> torch.Tensor:
        """
        内部计算观察值
        
        执行所有观察函数并将结果拼接成一个张量。
        
        Returns:
            torch.Tensor: 拼接后的观察张量
        """
        # 注释掉的代码用于导出观察元数据（调试用）
        # if self.name == "amp_obs_" and not hasattr(self, "_exported"):
        #     obs_metadata = []
        #     for obs_key, func in self.funcs.items():
        #         obs = func()
        #         metadata = {
        #             "obs_type": obs_key,
        #             "obs_dim": obs.shape[-1],
        #         }
        #         if hasattr(func, "joint_names"):
        #             metadata["joint_names"] = func.joint_names
        #         if hasattr(func, "body_names"):
        #             metadata["body_names"] = func.body_names
        #         if hasattr(func, 'history_steps'):
        #             metadata["history_steps"] = list(func.history_steps)
        #         obs_metadata.append(metadata)

        #     import os
        #     metadata_folder = "amp_obs/policy"
        #     metadata_path = f"{metadata_folder}/metadata.json"
        #     os.makedirs(metadata_folder, exist_ok=True)
        #     with open(metadata_path, 'w') as f:
        #         import json
        #         json.dump(obs_metadata, f, indent=2)
        #     breakpoint()
        #     self._exported = True
        
        # 计算所有观察函数的结果
        tensors = []
        # print(f"Computing observation group: {self.name}")  # 调试输出
        for obs_key, func in self.funcs.items():
            tensor = func()
            tensors.append(tensor)
            # print(f"\t{obs_key}: {tensor.shape}")  # 调试输出
        return torch.cat(tensors, dim=-1)  # 在最后一个维度拼接
    
    def symmetry_transforms(self):
        """
        获取观察的对称变换
        
        收集所有观察函数的对称变换并组合成一个统一的变换。
        
        Returns:
            SymmetryTransform: 组合后的对称变换对象
        """
        transforms = []
        for obs_key, func in self.funcs.items():
            transform = func.symmetry_transforms()
            transforms.append(transform)
        transform = symmetry_utils.SymmetryTransform.cat(transforms)
        return transform


class _Env(EnvBase):
    """
    强化学习环境基类
    
    这是所有强化学习环境的基础类，继承自 TorchRL 的 EnvBase。
    提供了完整的环境管理框架，包括观察、奖励、动作、终止条件等。
    
    主要特性:
    - 支持多后端物理引擎 (Isaac Lab, MuJoCo)
    - 模块化设计，支持插件系统
    - 完整的回调机制
    - 性能监控和统计
    - 配置驱动的组件管理
    
    更新日志:
    2024.10.10:
    - 禁用延迟机制
    - 重构翻转逻辑
    - 重置时不再重新计算观察
    """
    def __init__(self, cfg):
        """
        初始化环境
        
        Args:
            cfg: 环境配置对象，包含所有必要的参数
        """
        self.cfg = cfg
        self.backend = active_adaptation.get_backend()  # 获取物理引擎后端

        # 设置场景（需要在子类中实现）
        self.scene: InteractiveScene
        self.setup_scene()
        self._ground_mesh = None  # 地面网格缓存
        
        # 时间步长设置
        self.max_episode_length = self.cfg.max_episode_length  # 最大回合长度
        self.step_dt = self.cfg.sim.step_dt  # 环境步长时间
        self.physics_dt = self.sim.get_physics_dt()  # 物理步长时间
        self.decimation = int(self.step_dt / self.physics_dt)  # 子步数
        
        print(f"Step dt: {self.step_dt}, physics dt: {self.physics_dt}, decimation: {self.decimation}")

        # 初始化父类
        super().__init__(
            device=self.sim.device,
            batch_size=[self.num_envs],
            run_type_checks=False,  # 关闭类型检查以提高性能
        )
        
        # 回合管理
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.episode_count = 0  # 总回合数
        self.current_iter = 0  # 当前迭代次数

        # 定义环境规格
        # 终止条件规格：包含 done, terminated, truncated 三个布尔标志
        self.done_spec = Composite(
            done=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),  # 环境是否结束
            terminated=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),  # 是否因失败终止
            truncated=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),  # 是否因超时截断
            shape=[self.num_envs],
            device=self.device
        )

        # 奖励规格：包含统计信息
        self.reward_spec = Composite(
            {
                "stats": {
                    "episode_len": UnboundedContinuous([self.num_envs, 1]),  # 回合长度
                    "success": UnboundedContinuous([self.num_envs, 1]),  # 成功标志
                },
            },
            shape=[self.num_envs]
        ).to(self.device)

        # 获取类成员（用于动态发现组件）
        # members = dict(inspect.getmembers(self.__class__, inspect.isclass))  # 暂时未使用
        
        # 初始化命令管理器（负责生成任务命令）
        self.command_manager: mdp.Command = hydra.utils.instantiate(self.cfg.command, env=self)

        # 注释掉的代码用于动态发现随机化和终止函数（已弃用）
        # RAND_FUNCS = mdp.RAND_FUNCS
        # RAND_FUNCS.update(mdp.get_obj_by_class(members, mdp.Randomization))
        # TERM_FUNCS = mdp.TERM_FUNCS
        # for k, v in inspect.getmembers(self.command_manager):
        #     if getattr(v, "is_termination", False):
        #         TERM_FUNCS[k] = mdp.termination_wrapper(v)
        
        # 获取可用插件
        ADDONS = mdp.ADDONS

        # 初始化组件字典
        self.addons = OrderedDict()  # 环境插件
        self.randomizations = OrderedDict()  # 随机化组件
        self.observation_funcs: Dict[str, ObsGroup] = OrderedDict()  # 观察函数组
        self.reward_funcs = OrderedDict()  # 奖励函数（已弃用）
        
        # 回调函数列表
        # 回调系统说明：
        # - startup_callbacks: 环境初始化时调用一次，用于设置各组件的初始状态
        # - reset_callbacks: 每个回合开始时调用，用于重置各组件的状态
        # - pre_step_callbacks: 每个物理子步骤前调用，用于准备步骤执行
        # - post_step_callbacks: 每个物理子步骤后调用，用于处理步骤后的更新
        # - update_callbacks: 所有物理子步骤完成后调用，用于更新组件状态
        # - debug_draw_callbacks: 每步渲染时调用，用于绘制调试信息
        self._startup_callbacks = []  # 启动时回调
        self._update_callbacks = []  # 更新时回调
        self._perf_ema_update = {}  # 性能监控
        self._reset_callbacks = []  # 重置时回调
        self._debug_draw_callbacks = []  # 调试绘制回调
        self._pre_step_callbacks = []  # 步骤前回调
        self._post_step_callbacks = []  # 步骤后回调

        # 注册命令管理器回调
        # step: 每个物理子步骤前调用，用于更新命令状态（如运动进度）
        self._pre_step_callbacks.append(self.command_manager.step)
        # 注意：命令管理器的update在特殊位置处理，不在常规update回调中
        # self._update_callbacks.append(self.command_manager.update)  # 命令更新在特殊位置处理
        # reset: 每个回合开始时调用，用于重置命令状态和采样新的任务
        self._reset_callbacks.append(self.command_manager.reset)
        # debug_draw: 每步渲染时调用，用于绘制命令相关的调试信息（如目标轨迹）
        self._debug_draw_callbacks.append(self.command_manager.debug_draw)
        
        # 初始化动作管理器（负责处理智能体动作）
        self.action_manager: mdp.ActionManager = hydra.utils.instantiate(self.cfg.action, env=self)
        # reset: 每个回合开始时调用，用于重置动作管理器状态
        self._reset_callbacks.append(self.action_manager.reset)
        # debug_draw: 每步渲染时调用，用于绘制动作相关的调试信息
        self._debug_draw_callbacks.append(self.action_manager.debug_draw)
        
        # 定义动作规格
        self.action_spec = Composite(
            {
                "action": UnboundedContinuous((self.num_envs, self.action_dim))  # 动作向量
            },
            shape=[self.num_envs]
        ).to(self.device)
        
        # 初始化插件系统
        addons = self.cfg.get("addons", {})
        print(f"Addons: {ADDONS.keys()}")
        for key, params in addons.items():
            addon = ADDONS[key](self, **params if params is not None else {})
            self.addons[key] = addon
            # 注册插件的回调函数
            # reset: 每个回合开始时调用，用于重置插件状态
            self._reset_callbacks.append(addon.reset)
            # update: 所有物理子步骤完成后调用，用于更新插件状态
            self._update_callbacks.append(addon.update)
            # debug_draw: 每步渲染时调用，用于绘制插件相关的调试信息
            self._debug_draw_callbacks.append(addon.debug_draw)
        
        # 初始化随机化组件
        for key, params in self.cfg.randomization.items():
            if key == "body_scale":  # 跳过特定的随机化（可能在其他地方处理）
                continue
            rand = mdp.Randomization.registry[key](env=self, **(params if params is not None else {}))
            self.randomizations[key] = rand
            # 注册随机化组件的回调函数
            # startup: 环境初始化时调用一次，用于设置随机化的初始状态
            self._startup_callbacks.append(rand.startup)
            # reset: 每个回合开始时调用，用于重置随机化状态
            self._reset_callbacks.append(rand.reset)
            # debug_draw: 每步渲染时调用，用于绘制调试信息
            self._debug_draw_callbacks.append(rand.debug_draw)
            # step: 每个物理子步骤前调用，用于应用随机化效果
            self._pre_step_callbacks.append(rand.step)
            # update: 所有物理子步骤完成后调用，用于更新随机化状态
            self._update_callbacks.append(rand.update)

        # 初始化观察函数组
        for group_key, params in self.cfg.observation.items():
            funcs = OrderedDict()            
            for obs_spec, kwargs in params.items():
                # 解析观察规格字符串
                obs_name, obs_cls_name = parse_name_and_class(obs_spec)
                obs_cls = mdp.Observation.registry[obs_cls_name]
                
                # 过滤掉 _target_ 参数，因为观察类构造函数不接受这个参数
                if kwargs is not None:
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k != '_target_'}
                else:
                    filtered_kwargs = {}
                
                obs: mdp.Observation = obs_cls(env=self, **filtered_kwargs)
                funcs[obs_name] = obs

                # 注册观察函数的回调
                # startup: 环境初始化时调用一次，用于设置观察函数的初始状态
                self._startup_callbacks.append(obs.startup)
                # update: 所有物理子步骤完成后调用，用于更新观察数据
                self._update_callbacks.append(obs.update)
                # reset: 每个回合开始时调用，用于重置观察函数状态
                self._reset_callbacks.append(obs.reset)
                # debug_draw: 每步渲染时调用，用于绘制观察相关的调试信息
                self._debug_draw_callbacks.append(obs.debug_draw)
                # post_step: 每个物理子步骤后调用，用于处理子步骤后的观察更新
                self._post_step_callbacks.append(obs.post_step)
            
            # 创建观察组
            self.observation_funcs[group_key] = ObsGroup(group_key, funcs)
        
        # 执行所有启动回调
        # startup: 环境初始化完成后调用一次，用于设置各组件的初始状态
        for callback in self._startup_callbacks:
            callback()       
       
        # 初始化奖励系统
        reward_spec = Composite({})

        # 解析奖励配置
        self.mult_dt = self.cfg.reward.pop("_mult_dt_", True)  # 是否乘以时间步长

        # 统计和性能监控
        self._stats_ema = {}  # 指数移动平均统计
        self._perf_ema_reward = {}  # 奖励计算性能统计
        self._stats_ema_decay = 0.99  # 指数移动平均衰减率

        # 初始化奖励组
        self.reward_groups: Dict[str, RewardGroup] = OrderedDict()
        for group_name, func_specs in self.cfg.reward.items():
            print(f"Reward group: {group_name}")
            funcs = OrderedDict()
            self._stats_ema[group_name] = {}
            self._perf_ema_reward[group_name] = {}

            multiplicative = False  # 是否使用乘性奖励组合
            for rew_spec, params in func_specs.items():
                if params is None:
                    continue
                if rew_spec == "_multiplicative":  # 特殊标记：乘性组合
                    multiplicative = params
                    continue
                # 解析奖励规格
                rew_name, cls_name = parse_name_and_class(rew_spec)
                rew_cls = mdp.Reward.registry[cls_name]
                reward: mdp.Reward = rew_cls(env=self, **params)
                funcs[rew_name] = reward
                
                # 添加奖励统计规格
                reward_spec["stats", group_name, rew_name] = UnboundedContinuous(1, device=self.device)
                
                # 注册奖励函数的回调
                # update: 所有物理子步骤完成后调用，用于更新奖励计算所需的状态
                self._update_callbacks.append(reward.update)
                # reset: 每个回合开始时调用，用于重置奖励函数状态
                self._reset_callbacks.append(reward.reset)
                # debug_draw: 每步渲染时调用，用于绘制奖励相关的调试信息
                self._debug_draw_callbacks.append(reward.debug_draw)
                # step: 每个物理子步骤前调用，用于准备奖励计算
                self._pre_step_callbacks.append(reward.step)
                # post_step: 每个物理子步骤后调用，用于处理子步骤后的奖励更新
                self._post_step_callbacks.append(reward.post_step)
                
                print(f"\t{rew_name}: \t{reward.weight:.2f}, \t{reward.enabled}")
                
                # 初始化统计变量
                self._stats_ema[group_name][rew_name] = (torch.tensor(0., device=self.device), torch.tensor(0., device=self.device))
                self._perf_ema_reward[group_name][rew_name] = (torch.tensor(0., device=self.device), torch.tensor(0., device=self.device))
            
            # 创建奖励组
            self.reward_groups[group_name] = RewardGroup(self, group_name, funcs, multiplicative=multiplicative)
            reward_spec["stats", group_name, "return"] = UnboundedContinuous(1, device=self.device)

        # 完成奖励规格定义
        reward_spec["reward"] = UnboundedContinuous(max(1, len(self.reward_groups)), device=self.device)
        reward_spec["discount"] = UnboundedContinuous(1, device=self.device)
        self.reward_spec.update(reward_spec.expand(self.num_envs).to(self.device))
        self.discount = torch.ones((self.num_envs, 1), device=self.device)  # 折扣因子

        # 构建观察规格
        observation_spec = {}
        for group_key, group in self.observation_funcs.items():
            try:
                observation_spec.update(group.spec)
            except Exception as e:
                print(f"Error in computing observation spec for {group_key}: {e}")
                raise e

        self.observation_spec = Composite(
            observation_spec, 
            shape=[self.num_envs],
            device=self.device
        )

        # 初始化终止条件函数
        self.termination_funcs = OrderedDict()
        for key, params in self.cfg.termination.items():
            term_cls = mdp.Termination.registry[key]
            term_func = term_cls(env=self, **params)
            self.termination_funcs[key] = term_func
            
            # 注册终止条件函数的回调
            # update: 所有物理子步骤完成后调用，用于更新终止条件状态
            self._update_callbacks.append(term_func.update)
            # reset: 每个回合开始时调用，用于重置终止条件状态
            self._reset_callbacks.append(term_func.reset)
            
            # 添加终止条件统计
            self.reward_spec["stats", "termination", key] = UnboundedContinuous((self.num_envs, 1), device=self.device)

        # 初始化运行时变量
        self.timestamp = 0  # 当前时间戳

        # 初始化统计信息
        self.stats = self.reward_spec["stats"].zero()
    
        # 运行时状态
        self.input_tensordict = None  # 输入张量字典
        self.extra = {}  # 额外状态信息
        
        # 性能计时器
        self.reset_time = 0.  # 重置时间
        self.simulation_time = 0.  # 仿真时间
        self.update_time = 0.  # 更新时间
        self.reward_time = 0.  # 奖励计算时间
        self.command_time = 0.  # 命令更新时间
        self.termination_time = 0.  # 终止条件计算时间
        self.observation_time = 0.  # 观察计算时间
        self.ema_cnt = 0.  # 指数移动平均计数器
        
    def set_progress(self, progress: int):
        """
        设置训练进度
        
        Args:
            progress: 当前迭代次数
        """
        self.current_iter = progress

    @property
    def action_dim(self) -> int:
        """获取动作维度"""
        return self.action_manager.action_dim

    @property
    def num_envs(self) -> int:
        """
        获取并行环境数量
        
        Returns:
            int: 并行运行的环境实例数量
        """
        return self.scene.num_envs

    @property
    def stats_ema(self):
        """
        获取指数移动平均统计信息
        
        包含奖励统计、性能统计等各类指标。
        
        Returns:
            dict: 统计信息字典
        """
        result = {}
        
        # 奖励统计
        for group_key, group in self._stats_ema.items():
            for rew_key, (sum, cnt) in group.items():
                result[f"reward.{group_key}/{rew_key}"] = (sum / cnt).item()
        
        # 奖励计算性能统计
        for group_key, group in self._perf_ema_reward.items():
            group_time = 0.
            for rew_key, (sum, cnt) in group.items():
                group_time += (sum / cnt).item()
                result[f"performance_reward/{group_key}.{rew_key}"] = (sum / cnt).item()
            result[f"performance_reward/{group_key}/total"] = group_time
        
        # 更新性能统计
        for key, (sum, cnt) in self._perf_ema_update.items():
            result[f"performance_update/{key}"] = (sum / cnt).item()
        
        # 各阶段性能统计
        result["performance/reset_time"] = self.reset_time / self.ema_cnt
        result["performance/observation_time"] = self.observation_time / self.ema_cnt
        result["performance/reward_time"] = self.reward_time / self.ema_cnt
        result["performance/command_time"] = self.command_time / self.ema_cnt
        result["performance/termination_time"] = self.termination_time / self.ema_cnt
        result["performance/update_time"] = self.update_time / self.ema_cnt
        result["performance/simulation_time"] = self.simulation_time / self.ema_cnt
        
        return result
    
    def setup_scene(self):
        """
        设置仿真场景
        
        子类必须实现此方法来创建具体的仿真场景。
        """
        raise NotImplementedError
    
    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDictBase:
        """
        重置环境
        
        Args:
            tensordict: 可选的输入张量字典，包含重置掩码
            **kwargs: 额外参数
            
        Returns:
            TensorDictBase: 重置后的初始观察
        """
        start = time.perf_counter()
        
        # 确定需要重置的环境ID
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
            env_ids = env_mask.nonzero().squeeze(-1)
            self.episode_count += env_ids.numel()
        else:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 执行重置
        if len(env_ids):
            self._reset_idx(env_ids)  # 子类实现的具体重置逻辑
            self.scene.reset(env_ids)  # 重置场景
        
        # 重置回合长度计数器
        self.episode_length_buf[env_ids] = 0
        
        # 执行所有重置回调
        # reset_callbacks: 每个回合开始时调用，用于重置各组件的状态
        for callback in self._reset_callbacks:
            callback(env_ids)
        
        # 创建初始观察
        tensordict = TensorDict({}, self.num_envs, device=self.device)
        tensordict.update(self.observation_spec.zero())
        
        # 更新性能统计
        end = time.perf_counter()
        self.reset_time = self.reset_time * self._stats_ema_decay + (end - start)
        
        return tensordict

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        """
        重置指定环境的具体实现
        
        子类必须实现此方法来定义具体的重置逻辑。
        
        Args:
            env_ids: 需要重置的环境ID张量
        """
        raise NotImplementedError
    
    def apply_action(self, tensordict: TensorDictBase, substep: int):
        """
        应用动作到环境
        
        Args:
            tensordict: 包含动作的张量字典
            substep: 当前子步数
        """
        self.input_tensordict = tensordict
        self.action_manager(tensordict, substep)

    def _compute_observation(self, tensordict: TensorDictBase):
        """
        计算观察值
        
        Args:
            tensordict: 目标张量字典
        """
        start = time.perf_counter()
        
        # 计算所有观察组
        for group_key, obs_group in self.observation_funcs.items():
            obs_group.compute(tensordict, self.timestamp)
        
        # 更新性能统计
        end = time.perf_counter()
        self.observation_time = self.observation_time * self._stats_ema_decay + (end - start)
            
    def _compute_reward(self) -> TensorDictBase:
        """
        计算奖励
        
        Returns:
            TensorDictBase: 包含奖励信息的张量字典
        """
        start = time.perf_counter()
        
        # 如果没有奖励组，返回默认奖励
        if not self.reward_groups:
            return {"reward": torch.ones((self.num_envs, 1), device=self.device)}
        
        # 计算所有奖励组的奖励
        rewards = []
        for group, reward_group in self.reward_groups.items():
            reward = reward_group.compute()
            if self.mult_dt:  # 如果启用时间步长乘法
                reward *= self.step_dt
            rewards.append(reward)
            self.stats[group, "return"].add_(reward)  # 更新统计

        rewards = torch.cat(rewards, 1)  # 拼接所有奖励

        # 更新统计信息
        self.stats["episode_len"][:] = self.episode_length_buf.unsqueeze(1)
        self.stats["success"][:] = (self.episode_length_buf >= self.max_episode_length * 0.9).unsqueeze(1).float()
        if hasattr(self.command_manager, "success"):
            self.stats["success"][:] = self.command_manager.success.float()
        
        # 更新性能统计
        end = time.perf_counter()
        self.reward_time = self.reward_time * self._stats_ema_decay + (end - start)
        
        return {"reward": rewards}
    
    def _compute_termination(self) -> TensorDictBase:
        """
        计算终止条件
        
        Returns:
            TensorDictBase: 终止标志张量
        """
        start = time.perf_counter()
        
        # 如果没有终止条件函数，返回全False
        if not self.termination_funcs:
            return torch.zeros((self.num_envs, 1), dtype=bool, device=self.device)
        
        # 计算所有终止条件
        flags = []
        for key, func in self.termination_funcs.items():
            flag = func()
            self.stats["termination", key][:] = flag.float()  # 更新统计
            flags.append(flag)
        
        flags = torch.cat(flags, dim=-1)
        
        # 更新性能统计
        end = time.perf_counter()
        self.termination_time = self.termination_time * self._stats_ema_decay + (end - start)
        
        # 返回是否有任何终止条件被触发
        return flags.any(dim=-1, keepdim=True)

    def _update(self):
        """
        更新环境状态
        
        执行所有更新回调，处理渲染，更新计数器等。
        """
        start = time.perf_counter()
        
        # 执行所有更新回调
        # update_callbacks: 所有物理子步骤完成后调用，包括观察函数、奖励函数、随机化、终止条件等
        for callback in self._update_callbacks:
            # 注释掉的代码用于详细的性能监控
            # time_start = time.perf_counter()
            callback()
            # time_end = time.perf_counter()
            
            # # Get the class name and category
            # name = callback.__self__.__class__.__name__
            # category = classify_callback(callback)
            
            # # Create the new key format: category.name
            # key = f"{category}.{name}"
            
            # if key not in self._perf_ema_update:
            #     self._perf_ema_update[key] = (torch.tensor(0., device=self.device), torch.tensor(0., device=self.device))
            # sum_, cnt = self._perf_ema_update[key]
            # sum_.add_(time_end - time_start)
            # cnt.add_(1.)
        
        # 如果有GUI，进行渲染
        if self.sim.has_gui():
            self.sim.render()
        
        # 更新计数器和时间戳
        self.episode_length_buf.add_(1)
        self.timestamp += 1
        
        # 更新性能统计
        end = time.perf_counter()
        self.update_time = self.update_time * self._stats_ema_decay + (end - start)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        执行环境步骤
        
        完整的执行流程：
        1. 物理仿真阶段：
           - 应用动作 (apply_action)
           - 执行步骤前回调 (pre_step_callbacks)
           - 物理步进 (sim.step)
           - 执行步骤后回调 (post_step_callbacks)
        2. 状态更新阶段：
           - 执行更新回调 (update_callbacks)
        3. 计算阶段：
           - 计算奖励 (_compute_reward)
           - 更新命令管理器 (command_manager.update)
           - 计算观察 (_compute_observation)
           - 计算终止条件 (_compute_termination)
        4. 渲染阶段：
           - 执行调试绘制回调 (debug_draw_callbacks)
        
        Args:
            tensordict: 包含动作的张量字典
            
        Returns:
            TensorDictBase: 包含观察、奖励、终止条件等的张量字典
        """
        start = time.perf_counter()
        
        # 执行多个物理子步骤
        for substep in range(self.decimation):
            # 清除外部力（如果有的话）
            for asset in self.scene.articulations.values():
                if asset.has_external_wrench:
                    asset._external_force_b.zero_()
                    asset._external_torque_b.zero_()
                    asset.has_external_wrench = False
            
            # 应用动作
            self.apply_action(tensordict, substep)
            
            # 执行步骤前回调
            # 这些全都不管
            for callback in self._pre_step_callbacks:
                callback(substep)
            
            # 将数据写入仿真器并步进
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
            
            # 执行步骤后回调
            # post_step_callbacks: 每个物理子步骤后调用，包括观察函数、奖励函数等
            for callback in self._post_step_callbacks:
                callback(substep)
        
        # 更新仿真时间统计
        end = time.perf_counter()
        self.simulation_time = self.simulation_time * self._stats_ema_decay + (end - start)
        
        # 重置折扣因子
        self.discount.fill_(1.0)
        
        # 更新环境状态
        # _update: 所有物理子步骤完成后调用，执行所有update_callbacks
        self._update()
        
        # 创建输出张量字典
        tensordict = TensorDict({}, self.num_envs, device=self.device)
        tensordict.update(self._compute_reward())

        # 注意：命令更新是特殊情况，应该在奖励计算之后进行
        start = time.perf_counter()
        self.command_manager.update()
        end = time.perf_counter()
        self.command_time = self.command_time * self._stats_ema_decay + (end - start)

        # 计算观察和终止条件
        self._compute_observation(tensordict)
        terminated = self._compute_termination()
        truncated = (self.episode_length_buf >= self.max_episode_length).unsqueeze(1)
        
        # 如果命令管理器有完成标志，也考虑截断
        if hasattr(self.command_manager, "finished"):
            truncated = truncated | self.command_manager.finished
        
        # 设置终止和截断标志
        tensordict.set("terminated", terminated)
        tensordict.set("truncated", truncated)
        tensordict.set("done", terminated | truncated)
        tensordict.set("discount", self.discount.clone())
        tensordict["stats"] = self.stats.clone()

        # 调试绘制（如果有GUI）
        # debug_draw_callbacks: 每步渲染时调用，用于绘制各组件的调试信息
        if self.sim.has_gui():
            if hasattr(self, "debug_draw"): # isaac only
                self.debug_draw.clear()
            for callback in self._debug_draw_callbacks:
                callback()
        
        # 更新指数移动平均计数器
        self.ema_cnt = self.ema_cnt * self._stats_ema_decay + 1.
        
        return tensordict
    
    @property
    def ground_mesh(self):
        """
        获取地面网格（用于射线投射）
        
        Returns:
            Warp网格对象
        """
        if self.backend == "isaac":
            if self._ground_mesh is None:
                self._ground_mesh = _initialize_warp_meshes("/World/ground", self.device.type)
            return self._ground_mesh
        else:
            raise NotImplementedError
        
    def get_ground_height_at(self, pos: torch.Tensor) -> torch.Tensor:
        """
        获取指定位置的地面高度
        
        Args:
            pos: 位置张量，形状为 (..., 3)
            
        Returns:
            torch.Tensor: 地面高度，形状为 (...)
        """
        if self.backend == "isaac":
            bshape = pos.shape[:-1]
            ray_starts = pos.clone().reshape(-1, 3)
            ray_starts[:, 2] = 10.  # 从高处开始射线
            ray_directions = torch.tensor([0., 0., -1.], device=self.device)  # 向下射线
            
            # 执行射线投射
            ray_hits = raycast_mesh(
                ray_starts=ray_starts.reshape(-1, 3),
                ray_directions=ray_directions.expand(bshape.numel(), 3),
                max_dist=100.,
                mesh=self.ground_mesh,
                return_distance=False,
            )[0]
            
            # 计算距离并处理NaN值
            ray_distance = 10. - (ray_hits - ray_starts).norm(dim=-1)
            ray_distance = ray_distance.nan_to_num(10.)
            assert not ray_distance.isnan().any()
            return ray_distance.reshape(*bshape)
        elif self.backend == "mujoco":
            return torch.zeros(pos.shape[:-1], device=self.device)
    
    def _set_seed(self, seed: int = -1):
        """
        设置随机种子
        
        Args:
            seed: 随机种子值
        """
        # import omni.replicator.core as rep
        # rep.set_global_seed(seed)  # Isaac Lab 的随机种子设置
        torch.manual_seed(seed)

    def render(self, mode: str = "human"):
        """
        渲染环境
        
        Args:
            mode: 渲染模式
                - "human": 人类观看模式，无返回值
                - "rgb_array": 返回RGB数组
                - "ego_rgb": 返回第一人称RGB图像
                - "ego_depth": 返回第一人称深度图像
                
        Returns:
            根据模式返回相应的图像数据
        """
        self.sim.render()
        
        if mode == "human":
            return None
        elif mode == "rgb_array":
            # 获取RGB数据
            rgb_data = self._rgb_annotator.get_data()
            # 转换为numpy数组
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # 返回RGB数据
            return rgb_data[:, :, :3]
        elif mode == "ego_rgb":
            assert "tiled_camera" in self.scene.sensors, "Tiled camera is not set up in the scene."
            tiled_camera: TiledCamera = self.scene.sensors["tiled_camera"]
            ego_rgb_data = tiled_camera.data.output["rgb"][0] # 获取第一个环境的RGB数据
            return ego_rgb_data.cpu().numpy()[:, :, :3]  # 转换为numpy并保留RGB通道
        elif mode == "ego_depth":
            import cv2
            assert "tiled_camera" in self.scene.sensors, "Tiled camera is not set up in the scene."
            tiled_camera: TiledCamera = self.scene.sensors["tiled_camera"]
            ego_depth_data = tiled_camera.data.output["depth"][0].squeeze(-1) # 获取第一个环境的深度数据
            min_depth, max_depth = 0.1, 4.0
            ego_depth_data = torch.nan_to_num(ego_depth_data, nan=max_depth, posinf=max_depth, neginf=min_depth).cpu().numpy()
            ego_depth_data = (ego_depth_data - min_depth) / (max_depth - min_depth)
            ego_depth_data = (np.clip(ego_depth_data, 0, 1) * 255).astype(np.uint8)
            rgb = cv2.applyColorMap(ego_depth_data, colormap=cv2.COLORMAP_JET)
            return rgb
        else:
            raise NotImplementedError

    def state_dict(self):
        """
        获取环境状态字典
        
        Returns:
            dict: 包含环境规格的状态字典
        """
        sd = super().state_dict()
        sd["observation_spec"] = self.observation_spec
        sd["action_spec"] = self.action_spec
        sd["reward_spec"] = self.reward_spec
        return sd

    def get_extra_state(self) -> dict:
        """
        获取额外状态信息
        
        Returns:
            dict: 额外状态字典
        """
        return dict(self.extra)

    def close(self):
        """
        关闭环境
        
        清理资源，关闭仿真器等。
        """
        if not self.is_closed:
            if self.backend == "isaac":
                # 析构顺序敏感
                del self.scene
                # 清除回调和实例
                self.sim.clear_all_callbacks()
                self.sim.clear_instance()
                # 更新关闭状态
            super().close()

    def dump(self):
        """
        转储环境状态
        
        用于调试或保存环境状态。
        """
        if self.backend == "mujoco":
            self.scene.close()


class RewardGroup:
    """
    奖励组管理类
    
    用于管理多个奖励函数，支持加性和乘性两种组合方式。
    提供性能监控和统计功能。
    
    Attributes:
        env: 环境实例
        name: 奖励组名称
        funcs: 奖励函数字典
        multiplicative: 是否使用乘性组合
        enabled_rewards: 启用的奖励函数数量
        rew_buf: 奖励缓冲区
    """
    
    def __init__(self, env: _Env, name: str, funcs: OrderedDict[str, mdp.Reward], multiplicative: bool):
        """
        初始化奖励组
        
        Args:
            env: 环境实例
            name: 奖励组名称
            funcs: 奖励函数字典
            multiplicative: 是否使用乘性组合
        """
        self.env = env
        self.name = name
        self.funcs = funcs
        self.multiplicative = multiplicative
        self.enabled_rewards = sum([func.enabled for func in funcs.values()])
        self.rew_buf = torch.zeros(env.num_envs, self.enabled_rewards, device=env.device)
    
    def compute(self) -> torch.Tensor:
        """
        计算奖励组的总奖励
        
        Returns:
            torch.Tensor: 组合后的奖励张量
        """
        rewards = []
        # try:  # 异常处理（已注释）
        for key, func in self.funcs.items():
            # 性能计时
            time_start = time.perf_counter()
            reward, count = func()
            time_end = time.perf_counter()

            # 更新统计信息
            self.env.stats[self.name, key].add_(reward)

            # 更新指数移动平均统计
            sum, cnt = self.env._stats_ema[self.name][key]
            sum.mul_(self.env._stats_ema_decay).add_(reward.sum())
            cnt.mul_(self.env._stats_ema_decay).add_(count)

            # 更新性能统计
            sum_perf, cnt_perf = self.env._perf_ema_reward[self.name][key]
            sum_perf.mul_(self.env._stats_ema_decay).add_(time_end - time_start)
            cnt_perf.mul_(self.env._stats_ema_decay).add_(1.0)
            
            # 只收集启用的奖励
            if func.enabled:
                rewards.append(reward)
        # except Exception as e:
        #     raise RuntimeError(f"Error in computing reward for {key}: {e}")
        
        # 更新奖励缓冲区
        if len(rewards):
            self.rew_buf[:] = torch.cat(rewards, 1)

        # 根据组合方式返回结果
        if self.multiplicative:
            return self.rew_buf.prod(dim=1, keepdim=True)  # 乘性组合
        else:
            return self.rew_buf.sum(dim=1, keepdim=True)  # 加性组合


def classify_callback(callback):
    """
    根据回调函数的类型进行分类
    
    用于性能监控和调试，确定回调函数属于哪个组件类别。
    
    Args:
        callback: 要分类的回调函数
        
    Returns:
        str: 类别名称，包括 'reward', 'observation', 'randomization', 
             'termination', 'addon', 'command', 'unknown'
    """
    if not hasattr(callback, '__self__'):
        return 'unknown'
    
    callback_obj = callback.__self__
    
    # 检查继承层次结构
    if isinstance(callback_obj, mdp.Reward):
        return 'reward'
    elif isinstance(callback_obj, mdp.Observation):
        return 'observation'
    elif isinstance(callback_obj, mdp.Randomization):
        return 'randomization'
    elif isinstance(callback_obj, mdp.Termination):
        return 'termination'
    elif isinstance(callback_obj, mdp.AddOn):
        return 'addon'
    elif isinstance(callback_obj, mdp.Command):
        return 'command'
    else:
        return 'unknown'


def _initialize_warp_meshes(mesh_prim_path, device):
    """
    初始化 Warp 网格用于射线投射
    
    支持平面和网格两种类型的地面几何体。
    
    Args:
        mesh_prim_path: 网格图元路径
        device: 设备类型
        
    Returns:
        Warp网格对象
        
    Raises:
        RuntimeError: 当网格路径无效时
    """
    # 检查是否为平面 - 将 PhysX 平面作为特殊情况处理
    # 如果存在平面，需要创建一个无限大的平面网格
    mesh_prim = sim_utils.get_first_matching_child_prim(
        mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
    )
    
    # 如果没有找到平面，则需要读取网格
    if mesh_prim is None:
        # 获取网格图元
        mesh_prim = sim_utils.get_first_matching_child_prim(
            mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
        # 检查是否有效
        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
        # 转换为 UsdGeomMesh
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        # 读取顶点和面
        points = np.asarray(mesh_prim.GetPointsAttr().Get())
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
        wp_mesh = convert_to_warp_mesh(points, indices, device=device)
    else:
        # 创建无限大平面
        mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
        wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)
    
    # 返回 Warp 网格
    return wp_mesh