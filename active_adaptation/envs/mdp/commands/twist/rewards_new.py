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

# class keypoint_pos_tracking_local_product(_tracking_keypoint_aligned):
#     """
#     TWIST: tracking_keybody_pos = 2.0
#     Reward for tracking keypoint positions in robot-relative coordinates
#     """
#     def compute(self):
#         body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
#         body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]

#         root_pos_asset = self.command_manager.robot_root_pos_w.clone()
#         root_pos_motion = self.command_manager.ref_root_pos_w.clone()
#         root_quat_asset = self.command_manager.robot_root_quat_w
#         root_quat_motion = self.command_manager.ref_root_quat_w

#         # Zero out Z coordinate and use only yaw rotation
#         root_pos_asset[..., 2] = 0.0
#         root_pos_motion[..., 2] = 0.0
#         root_quat_asset = yaw_quat(root_quat_asset)
#         root_quat_motion = yaw_quat(root_quat_motion)

#         # Expand to match body dimensions
#         root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
#         root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
#         root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
#         root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

#         # Transform to local coordinates
#         body_pos_asset_relative = quat_apply_inverse(root_quat_asset, body_pos_asset - root_pos_asset)
#         body_pos_motion_relative = quat_apply_inverse(root_quat_motion, body_pos_motion - root_pos_motion)

#         diff = body_pos_motion_relative - body_pos_asset_relative
#         error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
#         return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)


# class joint_pos_tracking_product(_tracking_joint_aligned):
#     """
#     TWIST: tracking_joint_dof = 0.6
#     Reward for tracking joint positions
#     """
#     def compute(self):
#         joint_pos_asset = self.command_manager.asset.data.joint_pos[:, self.joint_indices_asset]
#         joint_pos_motion = self.command_manager.ref_joint_pos[:, self.joint_indices_motion]
#         diff = joint_pos_motion - joint_pos_asset
#         error = (diff.abs() - self.tolerance).clamp_min(0.0)
#         return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)


# class joint_vel_tracking_product(_tracking_joint_aligned):
#     """
#     TWIST: tracking_joint_vel = 0.2
#     Reward for tracking joint velocities
#     """
#     def compute(self):
#         joint_vel_asset = self.command_manager.asset.data.joint_vel[:, self.joint_indices_asset]
#         joint_vel_motion = self.command_manager.ref_joint_vel[:, self.joint_indices_motion]
#         diff = joint_vel_motion - joint_vel_asset
#         error = (diff.abs() - self.tolerance).clamp_min(0.0)
#         return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)


# class keypoint_ori_tracking_local_product(_tracking_keypoint_aligned):
#     """
#     TWIST: tracking_root_pose = 0.6
#     Reward for tracking keypoint orientations in robot-relative coordinates
#     """
#     def compute(self):
#         body_quat_asset = self.command_manager.asset.data.body_link_quat_w[:, self.body_indices_asset]
#         body_quat_motion = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]

#         root_quat_asset = self.command_manager.robot_root_quat_w
#         root_quat_motion = self.command_manager.ref_root_quat_w

#         # Use only yaw rotation
#         root_quat_asset = yaw_quat(root_quat_asset)
#         root_quat_motion = yaw_quat(root_quat_motion)

#         root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
#         root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

#         # Transform to local coordinates
#         body_quat_asset_relative = quat_mul(quat_conjugate(root_quat_asset), body_quat_asset)
#         body_quat_motion_relative = quat_mul(quat_conjugate(root_quat_motion), body_quat_motion)

#         # Compute orientation error using axis-angle representation
#         quat_diff = quat_mul(body_quat_motion_relative, quat_conjugate(body_quat_asset_relative))
#         axis_angle_diff = axis_angle_from_quat(quat_diff)
#         error = axis_angle_diff.norm(dim=-1)
#         return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)


# class keypoint_lin_vel_tracking_product(_tracking_keypoint_aligned):
#     """
#     TWIST: tracking_root_vel = 1.0
#     Reward for tracking keypoint linear velocities
#     """
#     def compute(self):
#         body_lin_vel_asset = self.command_manager.asset.data.body_com_lin_vel_w[:, self.body_indices_asset]
#         body_lin_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion]
#         diff = body_lin_vel_motion - body_lin_vel_asset
#         error = diff.norm(dim=-1)
#         return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)


# ========================= Regularization Rewards =========================

class feet_slip_twist(TrackReward):
    """
    TWIST: feet_slip = -0.1
    Penalize foot sliding (aligned with TWIST-master implementation)
    TWIST uses sqrt(velocity) instead of raw velocity
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

        # Get contact states from contact sensor (TWIST uses force > 5N)
        in_contact = self.contact_sensor.data.current_contact_time[:, self.contact_body_indices] > 0.02

        # TWIST implementation: sqrt(norm(v_xy))
        # Penalize XY velocity when in contact
        xy_velocity_norm = foot_velocities[..., :2].norm(dim=-1)
        slip = (in_contact.float() * torch.sqrt(xy_velocity_norm)).sum(dim=-1)

        return slip.unsqueeze(1)


class feet_contact_forces_twist(TrackReward):
    """
    TWIST: feet_contact_forces = -5e-4
    Penalize high contact forces on feet (aligned with TWIST-master)
    TWIST only checks Z-direction force, not 3D force magnitude
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

        # TWIST implementation: only check Z-direction force
        z_force = contact_forces[..., 2].abs()

        # rew[rew < max_force] = 0
        # rew[rew > max_force] -= max_force
        excessive_force = (z_force - self.max_contact_force).clamp_min(0.0)
        return excessive_force.sum(dim=-1).unsqueeze(1)


class feet_stumble_twist(TrackReward):
    """
    TWIST: feet_stumble = -1.25
    Penalize foot stumbling (aligned with TWIST-master implementation)
    TWIST checks if XY forces > 4 * |Z force|, indicating lateral collision
    """
    def __init__(self, threshold: float = 1.0, body_names: List[str] | str = ".*ankle_roll_link", **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold  # Not used in TWIST implementation, kept for compatibility

        # Get contact sensor
        from isaaclab.sensors import ContactSensor
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.contact_body_indices = self.contact_sensor.find_bodies(body_names)[0]

    def compute(self):
        # Get contact forces from contact sensor
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.contact_body_indices]

        # TWIST implementation: check if XY force magnitude > 4 * |Z force|
        # This indicates stumbling (lateral collision while foot should be planted)
        xy_force_norm = contact_forces[..., :2].norm(dim=-1)
        z_force_abs = contact_forces[..., 2].abs()

        # stumble = any(||F_xy|| > 4 * |F_z|)
        stumble_mask = xy_force_norm > 4.0 * z_force_abs
        stumble = stumble_mask.any(dim=-1).float()

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
        torque_limits = self.command_manager.asset.data.joint_effort_limits  # 修复：正确的属性名

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
    Penalize action changes (aligned with TWIST-master)
    TWIST uses L2 norm, not squared sum
    """
    def compute(self):
        # 使用action_buf而不是action属性
        action_buf = self.env.action_manager.action_buf
        action_diff = action_buf[:, :, 0] - action_buf[:, :, 1]
        # TWIST implementation: torch.norm(..., dim=1)
        return torch.norm(action_diff, dim=-1).unsqueeze(1)


class feet_air_time_twist(TrackReward):
    """
    TWIST: feet_air_time = 5.0
    Reward feet spending appropriate time in the air (aligned with TWIST-master)
    TWIST uses linear reward: clamp(t - target, max=0) and only rewards when moving
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
        self.last_contact = torch.zeros(self.num_envs, len(self.body_indices), device=self.device, dtype=torch.bool)
        self.contact_filt = torch.zeros(self.num_envs, len(self.body_indices), device=self.device, dtype=torch.bool)

    def compute(self):
        # Get contact from contact sensor
        # TWIST uses force > 5N for contact detection
        contact_forces_z = self.contact_sensor.data.net_forces_w[:, self.body_indices, 2]
        in_contact = contact_forces_z > 5.0

        # TWIST implementation
        self.contact_filt = torch.logical_or(in_contact, self.last_contact)
        self.last_contact = in_contact.clone()

        # first_contact = (air_time > 0) * contact_filt
        first_contact = (self.air_time > 0.0) * self.contact_filt

        # Increment air time
        self.air_time += self.env.step_dt

        # Linear reward: (t - target) * first_contact, clamped to max=0
        # This only rewards when air_time < target (negative value becomes positive reward)
        air_time_reward = (self.air_time - self.target_air_time) * first_contact.float()
        air_time_reward = air_time_reward.clamp(max=0.0)

        # Reset air time on contact
        self.air_time *= ~self.contact_filt

        # Sum over feet
        reward = air_time_reward.sum(dim=1)

        # TWIST-MAIN: only reward when moving (ref velocity > 0.05 m/s)
        # TWIST uses _ref_root_vel with threshold 0.05, not commands with 0.1
        ref_root_vel = self.command_manager.ref_root_lin_vel_w  # [num_envs, 3]
        is_moving = torch.norm(ref_root_vel[..., :2], dim=-1) > 0.05  # TWIST threshold
        reward *= is_moving.float()

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


# ========================= TWIST-Aligned Tracking Rewards =========================
# These tracking rewards are FULLY aligned with TWIST-master implementation
# They replace the HDMI keypoint-based tracking with TWIST's root-only tracking

class tracking_root_pose_twist_aligned(TrackReward):
    """
    TWIST: tracking_root_pose = 0.6
    Track ROOT POSE ONLY (not all keypoints like HDMI)
    Formula: exp(-5.0 * (pos_err + 0.1 * rot_err))
    """
    def __init__(self, sigma: float = 0.2, global_obs: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.global_obs = global_obs
        self.root_pose_scale = 5.0

    def compute(self):
        # Current root state
        root_pos = self.command_manager.robot_root_pos_w
        root_quat = self.command_manager.robot_root_quat_w

        # Reference root state
        ref_root_pos = self.command_manager.ref_root_pos_w
        ref_root_quat = self.command_manager.ref_root_quat_w

        # Position error
        if self.global_obs:
            root_pos_diff = ref_root_pos - root_pos
        else:
            # TWIST: only compare Z coordinate in local frame
            root_pos_diff = ref_root_pos[:, 2:3] - root_pos[:, 2:3]

        root_pos_err = (root_pos_diff ** 2).sum(dim=-1)

        # Rotation error (quaternion angle difference)
        from isaaclab.utils.math import quat_error_magnitude
        root_rot_err = quat_error_magnitude(root_quat, ref_root_quat)
        root_rot_err = root_rot_err ** 2

        # TWIST formula: exp(-5.0 * (pos_err + 0.1 * rot_err))
        reward = torch.exp(-self.root_pose_scale * (root_pos_err + 0.1 * root_rot_err))

        return reward.unsqueeze(1)


class tracking_root_vel_twist_aligned(TrackReward):
    """
    TWIST: tracking_root_vel = 1.0
    Track ROOT VELOCITY ONLY (not all keypoints like HDMI)
    Includes both linear and angular velocity
    Formula: exp(-1.0 * (lin_vel_err + 0.5 * ang_vel_err))
    """
    def __init__(self, sigma: float = 0.5, global_obs: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.global_obs = global_obs
        self.root_vel_scale = 1.0

    def compute(self):
        # Current root velocity
        root_lin_vel = self.command_manager.asset.data.root_lin_vel_w
        root_ang_vel = self.command_manager.asset.data.root_ang_vel_w

        # Reference root velocity
        ref_root_lin_vel = self.command_manager.ref_root_lin_vel_w
        ref_root_ang_vel = self.command_manager.ref_root_ang_vel_w

        root_quat = self.command_manager.robot_root_quat_w
        ref_root_quat = self.command_manager.ref_root_quat_w

        if self.global_obs:
            root_vel_diff = ref_root_lin_vel - root_lin_vel
            root_ang_vel_diff = ref_root_ang_vel - root_ang_vel
        else:
            # TWIST: transform to local frame
            local_ref_lin_vel = quat_apply_inverse(ref_root_quat, ref_root_lin_vel)
            root_vel_diff = local_ref_lin_vel - quat_apply_inverse(root_quat, root_lin_vel)

            local_ref_ang_vel = quat_apply_inverse(ref_root_quat, ref_root_ang_vel)
            root_ang_vel_diff = local_ref_ang_vel - quat_apply_inverse(root_quat, root_ang_vel)

        # Linear velocity error
        root_vel_err = (root_vel_diff ** 2).sum(dim=-1)

        # Angular velocity error
        root_ang_vel_err = (root_ang_vel_diff ** 2).sum(dim=-1)

        # TWIST formula: exp(-1.0 * (lin_vel_err + 0.5 * ang_vel_err))
        reward = torch.exp(-self.root_vel_scale * (root_vel_err + 0.5 * root_ang_vel_err))

        return reward.unsqueeze(1)


class tracking_joint_dof_twist_aligned(TrackReward):
    """
    TWIST: tracking_joint_dof = 0.6
    Track joint positions with WEIGHTED L2 squared error (aligned with TWIST-master)
    Formula: exp(-0.15 * sum(w * diff^2))

    TWIST uses per-joint weights (dof_err_w) for 23 DOF
    Extended to 29 DOF by adding wrist joint weights:
    - Legs: [1.0, 0.8, 0.8, 1.0, 0.5, 0.5] x 2
    - Waist: [0.6, 0.6, 0.6]
    - Arms: [0.8, 0.8, 0.8, 1.0] x 2
    - Wrists: [0.6, 0.5, 0.5] x 2 (wrist_roll, wrist_pitch, wrist_yaw)
    """
    def __init__(self, sigma: float = 0.2, use_29dof: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.pos_scale = 0.15
        self.use_29dof = use_29dof

        if use_29dof:
            # Extended weights for 29 DOF (G1 with wrist joints)
            dof_err_w = [
                1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # Left Leg (6)
                1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # Right Leg (6)
                0.6, 0.6, 0.6,                      # Waist (3)
                0.8, 0.8, 0.8, 1.0,                 # Left Arm (4): shoulder x3, elbow
                0.6, 0.5, 0.5,                      # Left Wrist (3): roll, pitch, yaw
                0.8, 0.8, 0.8, 1.0,                 # Right Arm (4): shoulder x3, elbow
                0.6, 0.5, 0.5,                      # Right Wrist (3): roll, pitch, yaw
            ]
        else:
            # Original TWIST weights for 23 DOF (no wrist joints)
            dof_err_w = [
                1.0, 0.8, 0.8, 1.0, 0.5, 0.5,  # Left Leg
                1.0, 0.8, 0.8, 1.0, 0.5, 0.5,  # Right Leg
                0.6, 0.6, 0.6,                  # waist yaw, roll, pitch
                0.8, 0.8, 0.8, 1.0,             # Left Arm
                0.8, 0.8, 0.8, 1.0,             # Right Arm
            ]

        self.dof_err_w = torch.tensor(dof_err_w, device=self.device, dtype=torch.float32)

    def compute(self):
        # Current and reference joint positions
        joint_pos = self.command_manager.asset.data.joint_pos
        ref_joint_pos = self.command_manager.ref_joint_pos

        # TWIST implementation: weighted L2 squared
        dof_diff = ref_joint_pos - joint_pos
        dof_err = (self.dof_err_w * dof_diff ** 2).sum(dim=-1)

        # TWIST formula: exp(-0.15 * weighted_error)
        reward = torch.exp(-self.pos_scale * dof_err)

        return reward.unsqueeze(1)


class tracking_joint_vel_twist_aligned(TrackReward):
    """
    TWIST: tracking_joint_vel = 0.2
    Track joint velocities with WEIGHTED L2 squared error (aligned with TWIST-master)
    Formula: exp(-0.01 * sum(w * diff^2))
    Uses same weights as tracking_joint_dof

    Extended to 29 DOF by adding wrist joint weights:
    - Wrists: [0.6, 0.5, 0.5] x 2 (wrist_roll, wrist_pitch, wrist_yaw)
    """
    def __init__(self, sigma: float = 0.5, use_29dof: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.vel_scale = 0.01
        self.use_29dof = use_29dof

        if use_29dof:
            # Extended weights for 29 DOF (G1 with wrist joints)
            dof_err_w = [
                1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # Left Leg (6)
                1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # Right Leg (6)
                0.6, 0.6, 0.6,                      # Waist (3)
                0.8, 0.8, 0.8, 1.0,                 # Left Arm (4): shoulder x3, elbow
                0.6, 0.5, 0.5,                      # Left Wrist (3): roll, pitch, yaw
                0.8, 0.8, 0.8, 1.0,                 # Right Arm (4): shoulder x3, elbow
                0.6, 0.5, 0.5,                      # Right Wrist (3): roll, pitch, yaw
            ]
        else:
            # Original TWIST weights for 23 DOF (no wrist joints)
            dof_err_w = [
                1.0, 0.8, 0.8, 1.0, 0.5, 0.5,  # Left Leg
                1.0, 0.8, 0.8, 1.0, 0.5, 0.5,  # Right Leg
                0.6, 0.6, 0.6,                  # waist yaw, roll, pitch
                0.8, 0.8, 0.8, 1.0,             # Left Arm
                0.8, 0.8, 0.8, 1.0,             # Right Arm
            ]

        self.dof_err_w = torch.tensor(dof_err_w, device=self.device, dtype=torch.float32)

    def compute(self):
        # Current and reference joint velocities
        joint_vel = self.command_manager.asset.data.joint_vel
        ref_joint_vel = self.command_manager.ref_joint_vel

        # TWIST implementation: weighted L2 squared
        vel_diff = ref_joint_vel - joint_vel
        vel_err = (self.dof_err_w * vel_diff ** 2).sum(dim=-1)

        # TWIST formula: exp(-0.01 * weighted_error)
        reward = torch.exp(-self.vel_scale * vel_err)

        return reward.unsqueeze(1)


class tracking_keybody_pos_twist_aligned(TrackReward):
    """
    TWIST: tracking_keybody_pos = 2.0
    Track key body positions (aligned with TWIST-master)
    Formula: exp(-10.0 * sum(sum(diff^2)))
    Key difference from HDMI: uses SUM instead of MEAN for aggregation
    """
    def __init__(self,
                 body_names: List[str] | str = None,
                 sigma: float = 0.2,
                 global_obs: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.global_obs = global_obs
        self.key_body_pos_scale = 10.0

        # TWIST key_bodies (from g1_mimic_distill_config.py line 289)
        # ["left_rubber_hand", "right_rubber_hand", "left_ankle_roll_link", "right_ankle_roll_link",
        #  "left_knee_link", "right_knee_link", "left_elbow_link", "right_elbow_link", "head_mocap"]
        if body_names is None:
            body_names = [
                ".*_hip_(pitch|yaw)_link",
                ".*_knee_link",
                ".*_ankle_roll_link",
                "pelvis",
                "torso_link",
                ".*_shoulder_pitch_link",
                ".*_elbow_link",
                ".*_wrist_yaw_link"
            ]

        self.body_indices_asset, _ = resolve_matching_names(body_names, self.command_manager.asset.body_names)
        self.body_indices_asset = torch.tensor(self.body_indices_asset, device=self.device, dtype=torch.long)

        # Use tracking_keypoint_names which contains the motion body names
        motion_body_names = self.command_manager.tracking_keypoint_names
        self.body_indices_motion, _ = resolve_matching_names(body_names, motion_body_names)
        self.body_indices_motion = torch.tensor(self.body_indices_motion, device=self.device, dtype=torch.long)

    def compute(self):
        # Get body positions
        key_body_pos = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        ref_key_body_pos = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]

        # Get root states
        root_pos = self.command_manager.robot_root_pos_w
        ref_root_pos = self.command_manager.ref_root_pos_w

        # Transform to relative coordinates
        key_body_pos = key_body_pos - root_pos.unsqueeze(1)
        ref_key_body_pos = ref_key_body_pos - ref_root_pos.unsqueeze(1)

        if not self.global_obs:
            # TWIST: only use yaw rotation for local frame
            root_quat = self.command_manager.robot_root_quat_w
            ref_root_quat = self.command_manager.ref_root_quat_w

            # Extract yaw quaternion only
            base_yaw_quat = yaw_quat(root_quat)
            ref_yaw_quat = yaw_quat(ref_root_quat)

            # Transform to local coordinates
            base_yaw_quat = base_yaw_quat.unsqueeze(1).expand(-1, key_body_pos.shape[1], -1)
            ref_yaw_quat = ref_yaw_quat.unsqueeze(1).expand(-1, ref_key_body_pos.shape[1], -1)

            key_body_pos = quat_apply_inverse(base_yaw_quat, key_body_pos)
            ref_key_body_pos = quat_apply_inverse(ref_yaw_quat, ref_key_body_pos)

        # TWIST implementation: sum(sum(diff^2))
        key_body_pos_diff = key_body_pos - ref_key_body_pos
        key_body_pos_err = (key_body_pos_diff ** 2).sum(dim=-1)  # sum over xyz
        key_body_pos_err = key_body_pos_err.sum(dim=-1)  # sum over all key bodies

        # TWIST formula: exp(-10.0 * total_error)
        reward = torch.exp(-self.key_body_pos_scale * key_body_pos_err)

        return reward.unsqueeze(1)
