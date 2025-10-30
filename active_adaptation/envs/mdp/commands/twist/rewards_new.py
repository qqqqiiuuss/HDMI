"""
TWIST-Aligned Reward Functions

此文件实现了完全对齐 TWIST-MAIN 原始版本的奖励函数
参考文件: /home/ubuntu/DATA2/workspace/xmh/TWIST-master/legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py

奖励函数对应关系：
  TWIST-MAIN                              HDMI Implementation
  --------------------------------------------------------------
  tracking_keybody_pos (2.0)      →      keypoint_pos_tracking_local_product
  tracking_joint_dof (0.6)        →      joint_pos_tracking_product
  tracking_joint_vel (0.2)        →      joint_vel_tracking_product
  tracking_root_pose (0.6)        →      keypoint_ori_tracking_local_product
  tracking_root_vel (1.0)         →      keypoint_lin_vel_tracking_product
  feet_slip (-0.1)                →      feet_slip_twist
  feet_contact_forces (-5e-4)     →      feet_contact_forces_twist
  feet_stumble (-1.25)            →      feet_stumble_twist
  dof_pos_limits (-5.0)           →      joint_pos_limits_twist
  dof_torque_limits (-1.0)        →      joint_torque_limits_twist
  dof_vel (-1e-4)                 →      dof_vel_twist
  dof_acc (-5e-8)                 →      dof_acc_twist
  action_rate (-0.01)             →      action_rate_l2_twist
  feet_air_time (5.0)             →      feet_air_time_twist
  ang_vel_xy (-0.01)              →      ang_vel_xy_twist
  ankle_dof_acc (-1e-7)           →      ankle_dof_acc_twist
  ankle_dof_vel (-2e-4)           →      ankle_dof_vel_twist

注意：所有regularization reward类都添加了 _twist 后缀以避免与通用reward类重名
"""

from active_adaptation.envs.mdp.commands.twist.command import TwistMotionTracking
from active_adaptation.envs.mdp.base import Reward as BaseReward

from typing import List, Dict, TYPE_CHECKING
from omegaconf import DictConfig
from isaaclab.utils.string import resolve_matching_names, resolve_matching_names_values
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_conjugate, axis_angle_from_quat, yaw_quat

import torch

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor


TrackReward = BaseReward[TwistMotionTracking]


# ========================= Base Classes =========================

class _tracking_keypoint_aligned(TrackReward):
    """Base class for keypoint tracking rewards (TWIST-aligned version)"""
    def __init__(self, body_names: List[str] | str | None = None, sigma: float = 0.2, tolerance: float | Dict[str, float] = 0.0, **kwargs):
        super().__init__(**kwargs)
        if body_names is None:
            body_names = self.command_manager.tracking_keypoint_names

        self.sigma = sigma
        body_indices_motion, matched_names_motion = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)
        body_indices_asset, matched_names_asset = resolve_matching_names(body_names, self.command_manager.asset.body_names)

        matched_names = set(matched_names_motion) & set(matched_names_asset)
        assert set(matched_names) == set(matched_names_motion) == set(matched_names_asset), "body names in motion dataset and robot not matched"

        self.body_indices_motion = []
        self.body_indices_asset = []
        self.body_names = list(sorted(matched_names))
        self.num_bodies = len(self.body_names)
        for body_name in self.body_names:
            body_idx_motion = self.command_manager.tracking_keypoint_names.index(body_name)
            body_idx_asset = self.command_manager.asset.body_names.index(body_name)
            self.body_indices_motion.append(body_idx_motion)
            self.body_indices_asset.append(body_idx_asset)

        self.tolerance = torch.zeros(len(self.body_names), device=self.device)
        if isinstance(tolerance, float):
            self.tolerance[:] = tolerance
        elif isinstance(tolerance, DictConfig):
            tolerance = dict(tolerance)
            tolerance_indices, tolerance_names, tolerance_values = resolve_matching_names_values(tolerance, self.body_names)
            self.tolerance[tolerance_indices] = torch.tensor(tolerance_values, device=self.device)


class _tracking_joint_aligned(TrackReward):
    """Base class for joint tracking rewards (TWIST-aligned version)"""
    def __init__(self, joint_names: List[str] | str | None = None, sigma: float = 0.2, tolerance: float | Dict[str, float] = 0.0, **kwargs):
        super().__init__(**kwargs)
        if joint_names is None:
            joint_names = self.command_manager.tracking_joint_names

        self.sigma = sigma
        joint_indices_motion, matched_names_motion = resolve_matching_names(joint_names, self.command_manager.tracking_joint_names)
        joint_indices_asset, matched_names_asset = resolve_matching_names(joint_names, self.command_manager.asset.joint_names)

        matched_names = set(matched_names_motion) & set(matched_names_asset)
        assert set(matched_names) == set(matched_names_motion) == set(matched_names_asset), "joint names in motion dataset and robot not matched"

        self.joint_indices_motion = []
        self.joint_indices_asset = []
        self.joint_names = list(sorted(matched_names))
        self.num_joints = len(self.joint_names)
        for joint_name in self.joint_names:
            joint_idx_motion = self.command_manager.tracking_joint_names.index(joint_name)
            joint_idx_asset = self.command_manager.asset.joint_names.index(joint_name)
            self.joint_indices_motion.append(joint_idx_motion)
            self.joint_indices_asset.append(joint_idx_asset)

        self.tolerance = torch.zeros(len(self.joint_names), device=self.device)
        if isinstance(tolerance, float):
            self.tolerance[:] = tolerance
        elif isinstance(tolerance, DictConfig):
            tolerance = dict(tolerance)
            tolerance_indices, tolerance_names, tolerance_values = resolve_matching_names_values(tolerance, self.joint_names)
            self.tolerance[tolerance_indices] = torch.tensor(tolerance_values, device=self.device)


# ========================= Tracking Rewards =========================

class keypoint_pos_tracking_local_product(_tracking_keypoint_aligned):
    """
    TWIST: tracking_keybody_pos = 2.0
    Reward for tracking keypoint positions in robot-relative coordinates
    """
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]

        root_pos_asset = self.command_manager.robot_root_pos_w.clone()
        root_pos_motion = self.command_manager.ref_root_pos_w.clone()
        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w

        # Zero out Z coordinate and use only yaw rotation
        root_pos_asset[..., 2] = 0.0
        root_pos_motion[..., 2] = 0.0
        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)

        # Expand to match body dimensions
        root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        # Transform to local coordinates
        body_pos_asset_relative = quat_apply_inverse(root_quat_asset, body_pos_asset - root_pos_asset)
        body_pos_motion_relative = quat_apply_inverse(root_quat_motion, body_pos_motion - root_pos_motion)

        diff = body_pos_motion_relative - body_pos_asset_relative
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)


class joint_pos_tracking_product(_tracking_joint_aligned):
    """
    TWIST: tracking_joint_dof = 0.6
    Reward for tracking joint positions
    """
    def compute(self):
        joint_pos_asset = self.command_manager.asset.data.joint_pos[:, self.joint_indices_asset]
        joint_pos_motion = self.command_manager.ref_joint_pos[:, self.joint_indices_motion]
        diff = joint_pos_motion - joint_pos_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)


class joint_vel_tracking_product(_tracking_joint_aligned):
    """
    TWIST: tracking_joint_vel = 0.2
    Reward for tracking joint velocities
    """
    def compute(self):
        joint_vel_asset = self.command_manager.asset.data.joint_vel[:, self.joint_indices_asset]
        joint_vel_motion = self.command_manager.ref_joint_vel[:, self.joint_indices_motion]
        diff = joint_vel_motion - joint_vel_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_ori_tracking_local_product(_tracking_keypoint_aligned):
    """
    TWIST: tracking_root_pose = 0.6
    Reward for tracking keypoint orientations in robot-relative coordinates
    """
    def compute(self):
        body_quat_asset = self.command_manager.asset.data.body_link_quat_w[:, self.body_indices_asset]
        body_quat_motion = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]

        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w

        # Use only yaw rotation
        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)

        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        # Transform to local coordinates
        body_quat_asset_relative = quat_mul(quat_conjugate(root_quat_asset), body_quat_asset)
        body_quat_motion_relative = quat_mul(quat_conjugate(root_quat_motion), body_quat_motion)

        # Compute orientation error using axis-angle representation
        quat_diff = quat_mul(body_quat_motion_relative, quat_conjugate(body_quat_asset_relative))
        axis_angle_diff = axis_angle_from_quat(quat_diff)
        error = axis_angle_diff.norm(dim=-1)
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_lin_vel_tracking_product(_tracking_keypoint_aligned):
    """
    TWIST: tracking_root_vel = 1.0
    Reward for tracking keypoint linear velocities
    """
    def compute(self):
        body_lin_vel_asset = self.command_manager.asset.data.body_com_lin_vel_w[:, self.body_indices_asset]
        body_lin_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion]
        diff = body_lin_vel_motion - body_lin_vel_asset
        error = diff.norm(dim=-1)
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)


# ========================= Regularization Rewards =========================

class feet_slip_twist(TrackReward):
    """
    TWIST: feet_slip = -0.1
    Penalize foot sliding
    """
    def __init__(self, body_names: List[str] | str = ".*ankle_roll_link", **kwargs):
        super().__init__(**kwargs)
        # Get both asset body indices and contact sensor indices
        self.asset_body_indices, _ = resolve_matching_names(body_names, self.command_manager.asset.body_names)
        self.asset_body_indices = torch.tensor(self.asset_body_indices, device=self.device, dtype=torch.long)

        # Get contact sensor
        from isaaclab.sensors import ContactSensor
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.contact_body_indices, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.contact_body_indices = torch.tensor(self.contact_body_indices, device=self.device, dtype=torch.long)

    def compute(self):
        # Get foot velocities from asset
        foot_velocities = self.command_manager.asset.data.body_lin_vel_w[:, self.asset_body_indices]

        # Get contact states from contact sensor
        in_contact = self.contact_sensor.data.current_contact_time[:, self.contact_body_indices] > 0.02

        # Penalize XY velocity when in contact
        xy_velocity = foot_velocities[..., :2].norm(dim=-1)
        slip = (in_contact.float() * xy_velocity).sum(dim=-1)

        return slip.unsqueeze(1)


class feet_contact_forces_twist(TrackReward):
    """
    TWIST: feet_contact_forces = -5e-4
    Penalize high contact forces on feet
    """
    def __init__(self, max_contact_force: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.max_contact_force = max_contact_force

        # Get contact sensor
        from isaaclab.sensors import ContactSensor
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_indices, self.body_names = self.contact_sensor.find_bodies(".*ankle_roll_link")
        self.body_indices = torch.tensor(self.body_indices, device=self.device, dtype=torch.long)

    def compute(self):
        # Get contact forces from contact sensor
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.body_indices]
        contact_force_magnitude = contact_forces.norm(dim=-1)
        excessive_force = (contact_force_magnitude - self.max_contact_force).clamp_min(0.0)
        return excessive_force.sum(dim=-1).unsqueeze(1)


class feet_stumble_twist(TrackReward):
    """
    TWIST: feet_stumble = -1.25
    Penalize foot stumbling (foot moving downward while in contact)
    """
    def __init__(self, threshold: float = 1.0, body_names: List[str] | str = ".*ankle_roll_link", **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

        # Get asset body indices for velocities
        self.asset_body_indices, _ = resolve_matching_names(body_names, self.command_manager.asset.body_names)
        self.asset_body_indices = torch.tensor(self.asset_body_indices, device=self.device, dtype=torch.long)

        # Get contact sensor
        from isaaclab.sensors import ContactSensor
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.contact_body_indices, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.contact_body_indices = torch.tensor(self.contact_body_indices, device=self.device, dtype=torch.long)

    def compute(self):
        # Get foot velocities from asset
        foot_velocities = self.command_manager.asset.data.body_lin_vel_w[:, self.asset_body_indices]

        # Get contact forces from contact sensor
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.contact_body_indices]

        # Contact if z-force > threshold
        contact_mask = (contact_forces[..., 2] > self.threshold).float()

        # Penalize downward velocity while in contact
        downward_velocity = (-foot_velocities[..., 2]).clamp_min(0.0)
        stumble = (downward_velocity * contact_mask).sum(dim=-1)

        return stumble.unsqueeze(1)


class joint_pos_limits_twist(TrackReward):
    """
    TWIST: dof_pos_limits = -5.0
    Penalize joints approaching position limits
    """
    def __init__(self, soft_factor: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.soft_factor = soft_factor

    def compute(self):
        joint_pos = self.command_manager.asset.data.joint_pos
        joint_limits = self.command_manager.asset.data.soft_joint_pos_limits

        # Soft limits
        soft_limits = joint_limits * self.soft_factor
        lower_violation = (soft_limits[:, :, 0] - joint_pos).clamp_min(0.0)
        upper_violation = (joint_pos - soft_limits[:, :, 1]).clamp_min(0.0)

        violation = (lower_violation + upper_violation).sum(dim=-1)
        return violation.unsqueeze(1)


class joint_torque_limits_twist(TrackReward):
    """
    TWIST: dof_torque_limits = -1.0
    Penalize joints approaching torque limits
    """
    def __init__(self, soft_factor: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.soft_factor = soft_factor

    def compute(self):
        joint_torques = self.command_manager.asset.data.applied_torque
        torque_limits = self.command_manager.asset.data.soft_joint_effort_limits

        soft_limits = torque_limits * self.soft_factor
        violation = (joint_torques.abs() - soft_limits).clamp_min(0.0)

        return violation.sum(dim=-1).unsqueeze(1)


class dof_vel_twist(TrackReward):
    """
    TWIST: dof_vel = -1e-4
    Penalize high joint velocities
    """
    def compute(self):
        joint_vel = self.command_manager.asset.data.joint_vel
        return (joint_vel ** 2).sum(dim=-1).unsqueeze(1)


class dof_acc_twist(TrackReward):
    """
    TWIST: dof_acc = -5e-8
    Penalize high joint accelerations
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_joint_vel = None

    def compute(self):
        joint_vel = self.command_manager.asset.data.joint_vel

        if self.last_joint_vel is None:
            self.last_joint_vel = joint_vel.clone()
            return torch.zeros(self.num_envs, 1, device=self.device)

        joint_acc = (joint_vel - self.last_joint_vel) / self.env.step_dt
        self.last_joint_vel = joint_vel.clone()

        return (joint_acc ** 2).sum(dim=-1).unsqueeze(1)


class action_rate_l2_twist(TrackReward):
    """
    TWIST: action_rate = -0.01
    Penalize action changes
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_actions = None

    def compute(self):
        actions = self.env.action_manager.action

        if self.last_actions is None:
            self.last_actions = actions.clone()
            return torch.zeros(self.num_envs, 1, device=self.device)

        action_diff = actions - self.last_actions
        self.last_actions = actions.clone()

        return (action_diff ** 2).sum(dim=-1).unsqueeze(1)


class feet_air_time_twist(TrackReward):
    """
    TWIST: feet_air_time = 5.0
    Reward feet spending appropriate time in the air
    """
    def __init__(self, target_air_time: float = 0.5, body_names: List[str] | str = ".*ankle_roll_link", **kwargs):
        super().__init__(**kwargs)
        self.target_air_time = target_air_time

        # Get contact sensor
        from isaaclab.sensors import ContactSensor
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_indices, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_indices = torch.tensor(self.body_indices, device=self.device, dtype=torch.long)

        # Track air time for each foot
        self.air_time = torch.zeros(self.num_envs, len(self.body_indices), device=self.device)
        self.last_contact = torch.ones(self.num_envs, len(self.body_indices), device=self.device, dtype=torch.bool)

    def compute(self):
        # Get contact from contact sensor
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_indices] > 0.02

        # Update air time
        self.air_time += self.env.step_dt
        self.air_time[in_contact] = 0.0

        # Reward when landing after target air time
        landing = in_contact & (~self.last_contact)
        reward = torch.zeros(self.num_envs, device=self.device)

        for i in range(len(self.body_indices)):
            landing_mask = landing[:, i]
            air_time_error = torch.abs(self.air_time[landing_mask, i] - self.target_air_time)
            reward[landing_mask] += torch.exp(-air_time_error / 0.25)

        self.last_contact = in_contact.clone()
        return reward.unsqueeze(1)


class ang_vel_xy_twist(TrackReward):
    """
    TWIST: ang_vel_xy = -0.01
    Penalize angular velocity in X and Y axes
    """
    def compute(self):
        ang_vel = self.command_manager.asset.data.root_ang_vel_b
        return (ang_vel[..., :2] ** 2).sum(dim=-1).unsqueeze(1)


class ankle_dof_acc_twist(TrackReward):
    """
    TWIST: ankle_dof_acc = -1e-7 (double weight for ankle joints)
    Penalize high ankle joint accelerations
    """
    def __init__(self, joint_names: List[str] | str = ".*ankle.*", **kwargs):
        super().__init__(**kwargs)
        joint_indices, self.joint_names = resolve_matching_names(joint_names, self.command_manager.asset.joint_names)
        self.joint_indices = torch.tensor(joint_indices, device=self.device, dtype=torch.long)
        self.last_joint_vel = None

    def compute(self):
        joint_vel = self.command_manager.asset.data.joint_vel[:, self.joint_indices]

        if self.last_joint_vel is None:
            self.last_joint_vel = joint_vel.clone()
            return torch.zeros(self.num_envs, 1, device=self.device)

        joint_acc = (joint_vel - self.last_joint_vel) / self.env.step_dt
        self.last_joint_vel = joint_vel.clone()

        return (joint_acc ** 2).sum(dim=-1).unsqueeze(1)


class ankle_dof_vel_twist(TrackReward):
    """
    TWIST: ankle_dof_vel = -2e-4 (double weight for ankle joints)
    Penalize high ankle joint velocities
    """
    def __init__(self, joint_names: List[str] | str = ".*ankle.*", **kwargs):
        super().__init__(**kwargs)
        joint_indices, self.joint_names = resolve_matching_names(joint_names, self.command_manager.asset.joint_names)
        self.joint_indices = torch.tensor(joint_indices, device=self.device, dtype=torch.long)

    def compute(self):
        joint_vel = self.command_manager.asset.data.joint_vel[:, self.joint_indices]
        return (joint_vel ** 2).sum(dim=-1).unsqueeze(1)
