from active_adaptation.envs.mdp.commands.hdmi.command import RobotObjectTracking
from active_adaptation.envs.mdp.base import Randomization as BaseRandomization

import torch
from typing import Dict, Tuple, List, TYPE_CHECKING
from omegaconf import DictConfig
from isaaclab.utils.math import quat_apply_inverse, sample_uniform


RobotObjectTrackRandomization = BaseRandomization[RobotObjectTracking]

class object_body_randomization(RobotObjectTrackRandomization ):
    def __init__(
        self,
        dynamic_friction_range: Tuple[float, float]=(0.6, 1.0),
        restitution_range: Tuple[float, float]=(0.0, 0.2),
        mass_range: Tuple[float, float]=(1.0, 10.0),
        static_friction_range: Tuple[float, float] | None = None,
        static_dynamic_friction_ratio_range: Tuple[float, float] | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Validate that only one of static_friction_range or static_dynamic_friction_ratio_range is specified
        if static_friction_range is not None and static_dynamic_friction_ratio_range is not None:
            raise ValueError("Cannot specify both static_friction_range and static_dynamic_friction_ratio_range")
        if static_friction_range is None and static_dynamic_friction_ratio_range is None:
            raise ValueError("Must specify either static_friction_range or static_dynamic_friction_ratio_range")
        
        self.object = self.command_manager.object

        self.mass_range = mass_range

        self.all_indices_cpu = torch.arange(self.object.num_instances)

        # randomize all shapes of the object
        max_shapes = self.object.root_physx_view.max_shapes
        self.shape_ids = torch.arange(0, max_shapes) 

        self.num_buckets = 64
        
        # Sample dynamic friction and restitution buckets
        self.dynamic_friction_buckets = sample_uniform(*tuple(dynamic_friction_range), (self.num_buckets,), "cpu")
        self.restitution_buckets = sample_uniform(*tuple(restitution_range), (self.num_buckets,), "cpu")
        
        # Handle static friction based on which parameter is specified
        if static_friction_range is not None:
            self.static_friction_buckets = sample_uniform(*tuple(static_friction_range), (self.num_buckets,), "cpu")
        else:
            self.static_dynamic_friction_ratio_buckets = sample_uniform(*tuple(static_dynamic_friction_ratio_range), (self.num_buckets,), "cpu")

    def startup(self):
        masses = self.object.data.default_mass.clone()
        inertias = self.object.data.default_inertia.clone()
        new_masses = sample_uniform(*self.mass_range, masses.shape, "cpu")

        scale = new_masses / masses
        masses[:] *= scale
        if inertias.ndim == 2:
            inertias[:] *= scale
        elif inertias.ndim == 3:
            inertias[:] *= scale.unsqueeze(-1)
        else:
            raise ValueError(f"Invalid shape for inertias: {inertias.shape}")
        self.object.root_physx_view.set_masses(masses, self.all_indices_cpu)
        self.object.root_physx_view.set_inertias(inertias, self.all_indices_cpu)
        assert torch.allclose(self.object.root_physx_view.get_masses(), masses, atol=1e-4)
        assert torch.allclose(self.object.root_physx_view.get_inertias(), inertias, atol=1e-4)

        materials = self.object.root_physx_view.get_material_properties().clone()
        shape = (self.object.num_instances, 1)
        dynamic_friction = self.dynamic_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        restitution = self.restitution_buckets[torch.randint(0, self.num_buckets, shape)]
        if hasattr(self, "static_friction_buckets"):
            static_friction = self.static_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        else:
            static_friction_ratio = self.static_dynamic_friction_ratio_buckets[torch.randint(0, self.num_buckets, shape)]
            static_friction = dynamic_friction * static_friction_ratio
        materials[:, self.shape_ids, 0] = static_friction
        materials[:, self.shape_ids, 1] = dynamic_friction
        materials[:, self.shape_ids, 2] = restitution
        self.object.root_physx_view.set_material_properties(materials.flatten(), self.all_indices_cpu)
        assert torch.allclose(self.object.root_physx_view.get_material_properties(), materials, atol=1e-4)

class object_joint_randomization(RobotObjectTrackRandomization):
    def __init__(
        self,
        friction_range: Tuple[float, float]=(0.0, 0.1),
        damping_range: Tuple[float, float]=(1.0, 10.0),
        armature_range: Tuple[float, float]=(0.0, 0.02),
        **kwargs
    ):
        super().__init__(**kwargs)
        if TYPE_CHECKING:
            from active_adaptation.assets.objects import CustomArticulation
        self.object: CustomArticulation = self.command_manager.object
        self.friction_range = friction_range
        self.damping_range = damping_range
        self.armature_range = armature_range

        self.joint_id_asset = 0
    
    def startup(self):
        door_armature = sample_uniform(*self.armature_range, (self.object.num_instances, 1), self.device)
        self.object.write_joint_armature_to_sim(door_armature, joint_ids=[self.joint_id_asset])

    def reset(self, env_ids: torch.Tensor):
        joint_friction = sample_uniform(*self.friction_range, (len(env_ids),), self.device)
        joint_damping = sample_uniform(*self.damping_range, (len(env_ids),), self.device)

        self.object._custom_friction[env_ids] = joint_friction
        self.object._custom_damping[env_ids] = joint_damping

# class keypoint_virtual_force(RobotTrackRandomization):
#     def __init__(
#         self,
#         body_names: str | List[str] = ".*",
#         stiffness_range: Tuple[float, float]=(20.0, 30.0),
#         annealing_steps: int=500,
#         pos_tolerance: float | Dict[str, float] = 0.0,
#         vel_tolerance: float | Dict[str, float] = 0.0,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.annealing_steps = annealing_steps

#         from isaaclab.utils.string import resolve_matching_names_values, resolve_matching_names
#         tracking_body_names = self.command_manager.tracking_keypoint_names
#         self.apply_force_body_names = resolve_matching_names(body_names, tracking_body_names)[1]
#         self.apply_force_body_indices_asset = []
#         self.apply_force_body_indices_motion = []
#         for name in self.apply_force_body_names:
#             body_idx_asset = self.command_manager.asset.body_names.index(name)
#             body_idx_motion = tracking_body_names.index(name)

#             self.apply_force_body_indices_asset.append(body_idx_asset)
#             self.apply_force_body_indices_motion.append(body_idx_motion)
        
#         self.pos_tolerance = torch.zeros(len(self.apply_force_body_names), device=self.device)
#         if isinstance(pos_tolerance, float):
#             self.pos_tolerance.fill_(pos_tolerance)
#         elif isinstance(pos_tolerance, DictConfig):
#             indices, names, values = resolve_matching_names_values(dict(pos_tolerance), self.apply_force_body_names)
#             self.pos_tolerance[indices] = torch.tensor(values, device=self.device)
#         else:
#             raise ValueError(f"Invalid type for pos_tolerance: {type(pos_tolerance)}")

#         self.vel_tolerance = torch.zeros(len(self.apply_force_body_names), device=self.device)
#         if isinstance(vel_tolerance, float):
#             self.vel_tolerance.fill_(vel_tolerance)
#         elif isinstance(vel_tolerance, DictConfig):
#             indices, names, values = resolve_matching_names_values(dict(vel_tolerance), self.apply_force_body_names)
#             self.vel_tolerance[indices] = torch.tensor(values, device=self.device)
#         else:
#             raise ValueError(f"Invalid type for vel_tolerance: {type(vel_tolerance)}")

#         self.stiffness_start = sample_uniform(*stiffness_range, (self.env.num_envs, 1, 1), self.device)
#         self.stiffness = self.stiffness_start.clone()
#         self.damping = self.stiffness.sqrt() * 2
        
#         self.ref_keypoint_pos_w = self.command_manager.ref_body_pos_w[:, self.apply_force_body_indices_motion]
#         self.ref_keypoint_lin_vel_w = self.command_manager.ref_body_lin_vel_w[:, self.apply_force_body_indices_motion]
    
#     def reset(self, env_ids: torch.Tensor):
#         # do not apply force in the first {decimation} steps
#         # because update is called after step after a reset
#         self.stiffness[env_ids] = 0.0
#         self.damping[env_ids] = 0.0
    
#     def update(self):
#         self.stiffness = self.stiffness_start * max(1.0 - self.env.current_iter / self.annealing_steps, 0.0)
#         self.damping = self.stiffness.sqrt() * 2

#         self.ref_keypoint_pos_w = self.command_manager.ref_body_pos_w[:, self.apply_force_body_indices_motion]
#         self.ref_keypoint_lin_vel_w = self.command_manager.ref_body_lin_vel_w[:, self.apply_force_body_indices_motion]

#     def step(self, substep):
#         if self.env.current_iter >= self.annealing_steps:
#             return
        
#         robot_keypoint_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.apply_force_body_indices_asset]
#         robot_keypoint_lin_vel_w = self.command_manager.asset.data.body_com_lin_vel_w[:, self.apply_force_body_indices_asset]

#         # compute force in world frame
#         diff_pos_w = self.ref_keypoint_pos_w - robot_keypoint_pos_w
#         diff_lin_vel_w = self.ref_keypoint_lin_vel_w - robot_keypoint_lin_vel_w
#         diff_pos_w = diff_pos_w * (diff_pos_w.abs() > self.pos_tolerance.unsqueeze(-1))
#         diff_lin_vel_w = diff_lin_vel_w * (diff_lin_vel_w.abs() > self.vel_tolerance.unsqueeze(-1))
#         self.forces_w = forces_w = self.stiffness * diff_pos_w + self.damping * diff_lin_vel_w
#         body_quat_w = self.command_manager.asset.data.body_quat_w[:, self.apply_force_body_indices_asset]
#         forces_b = quat_apply_inverse(body_quat_w, forces_w)

#         # apply force to asset
#         ext_forces_b = self.command_manager.asset._external_force_b
#         ext_forces_b[:, self.apply_force_body_indices_asset] += forces_b
#         self.command_manager.asset.has_external_wrench = True
    
#     def debug_draw(self):
#         if self.env.backend != "isaac":
#             return

#         if self.env.current_iter >= self.annealing_steps:
#             return

#         # draw force as vectors
#         body_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.apply_force_body_indices_asset]
#         self.env.debug_draw.vector(
#             body_pos_w.reshape(-1, 3),
#             (self.forces_w / self.stiffness).reshape(-1, 3) * 2.0,
#             # orange
#             color=(1.0, 0.5, 0.0, 1.0)
#         )


# ==================== TWIST-Aligned Domain Randomization (新增) ====================
# 以下功能对齐TWIST-MASTER的域随机化
# 参考: TWIST-master/legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py

from active_adaptation.envs.mdp.commands.twist.command import TwistMotionTracking
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, matrix_from_quat
import re

TwistRandomization = BaseRandomization[TwistMotionTracking]


class randomize_gravity(TwistRandomization):
    """
    重力方向随机化（TWIST对齐）

    通过在XY平面旋转重力向量，模拟机器人在倾斜地面上行走。

    TWIST原版: g1_mimic_distill_config.py line 247-249
    - randomize_gravity = True
    - gravity_rand_interval_s = 4
    - gravity_range = (-0.1, 0.1)  # ±5.7°

    Args:
        gravity_range: 重力倾斜角度范围 [min, max] (rad)
        interval_s: 重力改变的时间间隔（秒）
    """

    def __init__(self, gravity_range: list = [-0.1, 0.1], interval_s: float = 4.0, **kwargs):
        # 过滤掉Hydra配置参数
        kwargs.pop('_target_', None)
        super().__init__(**kwargs)
        self.gravity_range = gravity_range
        self.interval_s = interval_s
        self.interval_steps = int(interval_s / self.env.step_dt)
        self.step_counter = 0

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset时应用随机重力"""
        pass  # 在update中统一处理

    def update(self) -> None:
        """定期更新重力方向（每interval_s秒）"""
        self.step_counter += 1

        if self.step_counter % self.interval_steps == 0:
            # 为所有环境随机化重力方向
            roll_rand = torch.rand(self.num_envs, device=self.device) * \
                        (self.gravity_range[1] - self.gravity_range[0]) + self.gravity_range[0]
            pitch_rand = torch.rand(self.num_envs, device=self.device) * \
                         (self.gravity_range[1] - self.gravity_range[0]) + self.gravity_range[0]

            # 旋转重力向量 [0, 0, -9.81]
            # 注意：IsaacLab的重力设置可能需要通过scene或physics_sim_view
            # 这里提供基础实现框架
            pass  # 实际实现需要根据IsaacLab API调整


class push_robots(TwistRandomization):
    """
    外部推力随机化（TWIST对齐）

    定期对机器人施加随机推力，模拟碰撞或扰动。

    TWIST原版: g1_mimic_distill_config.py line 260-262
    - push_robots = True
    - push_interval_s = 4
    - max_push_vel_xy = 1.0
    """

    def __init__(
        self,
        max_push_vel_xy: float = 1.0,
        interval_s: float = 4.0,
        push_probability: float = 0.3,
        **kwargs
    ):
        # 过滤掉Hydra配置参数
        kwargs.pop('_target_', None)
        super().__init__(**kwargs)
        self.max_push_vel_xy = max_push_vel_xy
        self.interval_s = interval_s
        self.push_probability = push_probability
        self.interval_steps = int(interval_s / self.env.step_dt)
        self.step_counter = 0

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset时不推动"""
        pass

    def update(self) -> None:
        """定期推动机器人"""
        self.step_counter += 1

        if self.step_counter % self.interval_steps == 0:
            # 随机选择要推动的环境
            push_mask = torch.rand(self.num_envs, device=self.device) < self.push_probability
            push_env_ids = push_mask.nonzero(as_tuple=False).squeeze(-1)

            if len(push_env_ids) > 0:
                # 生成随机推力 (x, y方向)
                push_vel_xy = (torch.rand(len(push_env_ids), 2, device=self.device) * 2 - 1) * self.max_push_vel_xy

                # 获取机器人根节点
                robot = self.env.scene["robot"]

                # 获取当前速度 [len(push_env_ids), 3] for linear velocity
                current_lin_vel = robot.data.root_lin_vel_w[push_env_ids].clone()
                current_ang_vel = robot.data.root_ang_vel_w[push_env_ids].clone()

                # 添加推力到线速度的x, y分量
                current_lin_vel[:, 0:2] += push_vel_xy

                # 合并为6D速度 [len(push_env_ids), 6]
                velocity_6d = torch.cat([current_lin_vel, current_ang_vel], dim=-1)

                # 写入到仿真
                robot.write_root_velocity_to_sim(
                    velocity_6d,
                    env_ids=push_env_ids
                )


class push_end_effector(TwistRandomization):
    """
    末端执行器推力随机化（TWIST对齐）

    TWIST原版: g1_mimic_distill_config.py line 264-267
    """

    def __init__(
        self,
        body_names: list = [".*_wrist_yaw_link", ".*_ankle_roll_link"],
        max_push_vel: float = 0.5,
        interval_s: float = 4.0,
        push_probability: float = 0.3,
        **kwargs
    ):
        # 过滤掉Hydra配置参数
        kwargs.pop('_target_', None)
        super().__init__(**kwargs)
        self.body_names_patterns = body_names
        self.max_push_vel = max_push_vel
        self.interval_s = interval_s
        self.push_probability = push_probability
        self.interval_steps = int(interval_s / self.env.step_dt)
        self.step_counter = 0
        self.body_indices = None

    def _lazy_init(self):
        """延迟初始化body索引"""
        if self.body_indices is not None:
            return

        robot = self.env.scene["robot"]
        body_names = robot.data.body_names

        # 查找匹配的body
        self.body_indices = []
        for pattern in self.body_names_patterns:
            regex = re.compile(pattern)
            for i, name in enumerate(body_names):
                if regex.match(name):
                    self.body_indices.append(i)

        self.body_indices = list(set(self.body_indices))  # 去重

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset时不推动"""
        pass

    def update(self) -> None:
        """定期推动末端执行器

        注意：IsaacLab不支持直接修改单个body的速度，需要使用外力实现。
        当前实现暂时禁用，避免运行时错误。
        TODO: 使用 apply_external_force_and_torque API 重新实现。
        """
        # self._lazy_init()

        # if len(self.body_indices) == 0:
        #     return

        # self.step_counter += 1

        # if self.step_counter % self.interval_steps == 0:
        #     # 随机选择要推动的环境
        #     push_mask = torch.rand(self.num_envs, device=self.device) < self.push_probability
        #     push_env_ids = push_mask.nonzero(as_tuple=False).squeeze(-1)

        #     if len(push_env_ids) > 0:
        #         robot = self.env.scene["robot"]

        #         for env_id in push_env_ids:
        #             # 随机选择一个body
        #             body_idx = self.body_indices[torch.randint(0, len(self.body_indices), (1,)).item()]

        #             # 生成随机推力
        #             push_vel = (torch.rand(3, device=self.device) * 2 - 1) * self.max_push_vel

        #             # TODO: 使用外力API而不是直接修改速度
        #             # robot.apply_external_force(force, body_idx, env_ids=[env_id])

        pass  # 暂时禁用此功能


class randomize_motor_strength(TwistRandomization):
    """
    电机强度随机化（TWIST对齐）

    随机化每个电机的输出强度，模拟电机老化、电池电量不足等。

    TWIST原版: g1_mimic_distill_config.py line 269-270
    - randomize_motor = True
    - motor_strength_range = [0.8, 1.2]
    """

    def __init__(self, strength_range: list = [0.8, 1.2], **kwargs):
        # 过滤掉Hydra配置参数
        kwargs.pop('_target_', None)
        super().__init__(**kwargs)
        self.strength_range = strength_range

        # 存储每个环境的电机强度
        self.motor_strength = None

    def reset(self, env_ids: torch.Tensor) -> None:
        """每次reset时随机化电机强度"""
        robot = self.env.scene["robot"]
        num_dof = robot.num_joints

        if self.motor_strength is None:
            # 首次初始化
            self.motor_strength = torch.ones(
                self.num_envs, num_dof,
                device=self.device
            )

        # 为reset的环境随机化电机强度
        self.motor_strength[env_ids] = torch.rand(
            len(env_ids), num_dof,
            device=self.device
        ) * (self.strength_range[1] - self.strength_range[0]) + self.strength_range[0]

    def update(self) -> None:
        """每步应用电机强度缩放"""
        # 电机强度在action manager中应用
        # 存储供action_manager使用: torques *= motor_strength
        pass

    def get_motor_strength(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """获取电机强度（供action manager调用）"""
        if env_ids is None:
            return self.motor_strength
        else:
            return self.motor_strength[env_ids]
