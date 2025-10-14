"""
机器人跟踪和物体交互跟踪任务的观察函数模块

本模块定义了用于机器人运动跟踪和机器人-物体交互跟踪任务的各种观察函数。
这些观察函数为强化学习智能体提供环境状态信息，包括：
- 参考运动数据（关节位置、速度、根节点位置和方向等）
- 机器人当前状态
- 物体状态信息
- 各种坐标系转换后的观察数据

主要包含两类观察函数：
1. RobotTrackObservation: 基础机器人跟踪观察函数
2. RobotObjectTrackObservation: 带物体交互的机器人跟踪观察函数
"""

from active_adaptation.envs.mdp.commands.hdmi.command import RobotTracking, RobotObjectTracking
from active_adaptation.envs.mdp.base import Observation as BaseObservation

import torch
from isaaclab.utils.math import (
    quat_apply_inverse,
    quat_mul,
    quat_conjugate,
    matrix_from_quat,
    yaw_quat,
    wrap_to_pi
)
from active_adaptation.utils.math import batchify

# 对四元数逆变换函数进行批处理优化
quat_apply_inverse = batchify(quat_apply_inverse)

# 定义机器人跟踪观察函数的基类
RobotTrackObservation = BaseObservation[RobotTracking]

class ref_joint_pos_future(RobotTrackObservation):
    """
    未来参考关节位置观察函数
    
    返回未来时间步的参考关节位置数据，用于指导机器人跟踪目标运动。
    数据形状: [num_envs, num_future_steps * num_joints]
    """
    def compute(self):
        return self.command_manager.ref_joint_pos_future_.view(self.num_envs, -1)

class ref_joint_vel_future(RobotTrackObservation):
    """
    未来参考关节速度观察函数
    
    返回未来时间步的参考关节速度数据，用于指导机器人跟踪目标运动。
    数据形状: [num_envs, num_future_steps * num_joints]
    """
    def compute(self):
        return self.command_manager.ref_joint_vel_future_.view(self.num_envs, -1)

class ref_joint_pos_action(RobotTrackObservation):
    """
    参考关节位置动作观察函数
    
    返回当前参考运动中的关节位置，仅包含动作空间中的关节。
    用于为策略网络提供参考关节位置信息。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 获取动作管理器中的关节名称
        action_manager = self.env.action_manager
        action_joint_names = action_manager.joint_names
        # 找到动作关节在运动数据中的索引
        self.action_indices_motion = [self.command_manager.dataset.joint_names.index(joint_name) for joint_name in action_joint_names]

    def compute(self):
        # 返回当前参考运动中动作关节的位置
        ref_joint_pos = self.command_manager.current_ref_motion.joint_pos[:, self.action_indices_motion]
        return ref_joint_pos

class ref_joint_pos_action_policy(RobotTrackObservation):
    """
    参考关节位置策略观察函数
    
    返回经过动作缩放处理的参考关节位置，用于策略网络训练。
    将参考关节位置转换为策略动作空间。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 获取动作管理器中的关节名称
        action_manager = self.env.action_manager
        action_joint_names = action_manager.joint_names
        # 找到动作关节在运动数据中的索引
        self.action_indices_motion = [self.command_manager.dataset.joint_names.index(joint_name) for joint_name in action_joint_names]

        # 获取动作缩放参数和默认关节位置
        self.action_scaling = action_manager.action_scaling
        self.default_joint_pos = action_manager.default_joint_pos[:, action_manager.joint_ids]

    def compute(self):
        # 获取参考关节位置并转换为策略动作
        ref_joint_pos = self.command_manager.current_ref_motion.joint_pos[:, self.action_indices_motion]
        ref_joint_action = (ref_joint_pos - self.default_joint_pos) / self.action_scaling
        return ref_joint_action

class ref_root_pos_future_b(RobotTrackObservation):
    """
    机器人根节点坐标系下的未来参考根节点位置观察函数
    
    将世界坐标系下的未来参考根节点位置转换到机器人根节点坐标系中。
    这种转换使得机器人能够相对于自身位置来理解目标位置。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_future_steps = self.command_manager.num_future_steps
        # 初始化存储转换后位置的张量
        self.ref_root_pos_future_b = torch.zeros(self.num_envs, num_future_steps, 3, device=self.device)

    def update(self):
        # 获取世界坐标系下的未来参考根节点位置
        ref_root_pos_future_w = self.command_manager.ref_root_pos_future_w # shape: [num_envs, num_future_steps, 3]
        # 获取机器人根节点在世界坐标系中的位置和方向
        robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, :] # shape: [num_envs, 1, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]
        
        # 将参考位置转换到机器人根节点坐标系
        ref_root_pos_future_b = quat_apply_inverse(robot_root_quat_w, ref_root_pos_future_w - robot_root_pos_w)
        self.ref_root_pos_future_b = ref_root_pos_future_b

    def compute(self):
        return self.ref_root_pos_future_b.view(self.num_envs, -1)
    
class ref_root_ori_future_b(RobotTrackObservation):
    """
    机器人根节点坐标系下的未来参考根节点方向观察函数
    
    将世界坐标系下的未来参考根节点方向转换到机器人根节点坐标系中。
    只保留前两行（x和y轴方向），忽略z轴方向以减少观察维度。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_future_steps = self.command_manager.num_future_steps
        # 初始化存储转换后方向的张量 (只存储前两行)
        self.ref_root_ori_future_b = torch.zeros(self.num_envs, num_future_steps, 2, 3, device=self.device)

    def update(self):
        # 获取世界坐标系下的未来参考根节点四元数
        ref_root_quat_future_w = self.command_manager.ref_root_quat_future_w # shape: [num_envs, num_future_steps, 4]
        # 获取机器人根节点在世界坐标系中的四元数
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]
        
        # 将参考四元数转换到机器人根节点坐标系
        ref_root_quat_future_b = quat_mul(
            quat_conjugate(robot_root_quat_w).expand_as(ref_root_quat_future_w),
            ref_root_quat_future_w
        )
        # 将四元数转换为旋转矩阵，只保留前两行
        ref_root_ori_future_b = matrix_from_quat(ref_root_quat_future_b)
        self.ref_root_ori_future_b = ref_root_ori_future_b[:, :, :2, :]

    def compute(self):
        return self.ref_root_ori_future_b.reshape(self.num_envs, -1)

class ref_body_pos_future_local(RobotTrackObservation):
    """
    运动根节点坐标系下的未来参考身体位置观察函数
    
    将世界坐标系下的未来参考身体位置转换到运动根节点坐标系中。
    使用yaw-only四元数进行转换，忽略pitch和roll，只考虑yaw旋转。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储转换后身体位置的张量
        self.ref_body_pos_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)
    
    def update(self):
        # 获取世界坐标系下的未来参考身体位置
        ref_body_pos_future_w = self.command_manager.ref_body_pos_future_w    # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        # 获取参考根节点在世界坐标系中的位置和方向
        ref_root_pos_w = self.command_manager.ref_root_pos_w[:, None, None, :].clone() # shape: [num_envs, 1, 1, 3]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        # 将z坐标设为0，只考虑水平面
        ref_root_pos_w[..., 2] = 0.0
        # 只保留yaw旋转，忽略pitch和roll
        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        # 将身体位置转换到运动根节点坐标系
        ref_body_pos_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_pos_future_w - ref_root_pos_w)
        self.ref_body_pos_future_local = ref_body_pos_future_local
    
    def compute(self):
        return self.ref_body_pos_future_local.view(self.num_envs, -1)

class ref_body_ori_future_local(RobotTrackObservation):
    """
    运动根节点坐标系下的未来参考身体方向观察函数
    
    将世界坐标系下的未来参考身体方向转换到运动根节点坐标系中。
    使用yaw-only四元数进行转换，只保留前两行（x和y轴方向）。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储转换后身体方向的张量
        self.ref_body_ori_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, 3, device=self.device)
    
    def update(self):
        # 获取世界坐标系下的未来参考身体四元数
        ref_body_quat_future_w = self.command_manager.ref_body_quat_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        # 获取参考根节点在世界坐标系中的四元数
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        # 只保留yaw旋转
        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        # 将参考身体四元数转换到运动根节点坐标系
        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_root_quat_w).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_w
        )
        # 将四元数转换为旋转矩阵
        self.ref_body_ori_future_local = matrix_from_quat(ref_body_quat_future_local)
    
    def compute(self):
        # 只返回前两行（x和y轴方向），减少观察维度
        return self.ref_body_ori_future_local[:, :, :, :2, :].reshape(self.num_envs, -1)

class diff_body_pos_future_local(RobotTrackObservation):
    """
    未来身体位置差异观察函数（局部坐标系）
    
    计算参考身体位置与机器人身体位置的差异，分别在各自的根节点坐标系中。
    这种差异表示机器人需要如何调整身体位置来匹配参考运动。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储身体位置差异的张量
        self.diff_body_pos_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)

    def update(self):
        # 获取世界坐标系下的未来参考身体位置
        ref_body_pos_future_w = self.command_manager.ref_body_pos_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        # 获取参考根节点在世界坐标系中的位置和方向
        ref_root_pos_w = self.command_manager.ref_root_pos_w[:, None, None, :].clone() # shape: [num_envs, 1, 1, 3]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        # 获取机器人身体在世界坐标系中的位置
        robot_body_pos_w = self.command_manager.robot_body_pos_w # shape: [num_envs, num_tracking_bodies, 3]
        # 获取机器人根节点在世界坐标系中的位置和方向
        robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, :].clone() # shape: [num_envs, 1, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        # 将z坐标设为0，只考虑水平面
        ref_root_pos_w[..., 2] = 0.0
        robot_root_pos_w[..., 2] = 0.0
        # 只保留yaw旋转
        ref_root_quat_w = yaw_quat(ref_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        # 将参考身体位置转换到参考根节点坐标系
        ref_body_pos_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_pos_future_w - ref_root_pos_w)
        # 将机器人身体位置转换到机器人根节点坐标系
        robot_body_pos_local = quat_apply_inverse(robot_root_quat_w, robot_body_pos_w - robot_root_pos_w)

        # 计算差异（参考位置 - 机器人位置）
        self.diff_body_pos_future_local = ref_body_pos_future_local - robot_body_pos_local.unsqueeze(1)

    def compute(self):
        return self.diff_body_pos_future_local.view(self.num_envs, -1)
    
class diff_body_lin_vel_future_local(RobotTrackObservation):
    """
    未来身体线性速度差异观察函数（局部坐标系）
    
    计算参考身体线性速度与机器人身体线性速度的差异，分别在各自的根节点坐标系中。
    这种差异表示机器人需要如何调整身体速度来匹配参考运动。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储身体线性速度差异的张量
        self.diff_body_lin_vel_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)
    
    def update(self):
        # 获取世界坐标系下的未来参考身体线性速度
        ref_body_lin_vel_future_w = self.command_manager.ref_body_lin_vel_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        # 获取参考根节点在世界坐标系中的方向
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]
        # 获取机器人身体在世界坐标系中的线性速度
        robot_body_lin_vel_w = self.command_manager.robot_body_lin_vel_w # shape: [num_envs, num_tracking_bodies, 3]
        # 获取机器人根节点在世界坐标系中的方向
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        # 只保留yaw旋转
        ref_root_quat_w = yaw_quat(ref_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        # 将参考身体线性速度转换到参考根节点坐标系
        ref_body_lin_vel_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_lin_vel_future_w)
        # 将机器人身体线性速度转换到机器人根节点坐标系
        robot_body_lin_vel_local = quat_apply_inverse(robot_root_quat_w, robot_body_lin_vel_w)

        # 计算差异（参考速度 - 机器人速度）
        self.diff_body_lin_vel_future_local = ref_body_lin_vel_future_local - robot_body_lin_vel_local.unsqueeze(1)

    def compute(self):
        return self.diff_body_lin_vel_future_local.view(self.num_envs, -1)

    
class diff_body_ori_future_local(RobotTrackObservation):
    """
    未来身体方向差异观察函数（局部坐标系）
    
    计算参考身体方向与机器人身体方向的差异，分别在各自的根节点坐标系中。
    这种差异表示机器人需要如何调整身体方向来匹配参考运动。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储身体方向差异的张量
        self.diff_body_ori_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, 3, device=self.device)

    def update(self):
        # 获取世界坐标系下的未来参考身体四元数
        ref_body_quat_future_w = self.command_manager.ref_body_quat_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        # 获取参考根节点在世界坐标系中的四元数
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]
        # 获取机器人身体在世界坐标系中的四元数
        robot_body_quat_w = self.command_manager.robot_body_quat_w # shape: [num_envs, num_tracking_bodies, 4]
        # 获取机器人根节点在世界坐标系中的四元数
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        # 只保留yaw旋转
        ref_root_quat_w = yaw_quat(ref_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        # 将参考身体四元数转换到参考根节点坐标系
        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_root_quat_w).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_w
        )
        # 将机器人身体四元数转换到机器人根节点坐标系
        robot_body_quat_local = quat_mul(
            quat_conjugate(robot_root_quat_w).expand_as(robot_body_quat_w),
            robot_body_quat_w
        ).unsqueeze(1)
        # 计算四元数差异（相对旋转）
        diff_body_quat_future = quat_mul(
            quat_conjugate(robot_body_quat_local).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_local
        )
        # 将四元数转换为旋转矩阵
        self.diff_body_ori_future_local = matrix_from_quat(diff_body_quat_future)

    def compute(self):
        # 只返回前两行（x和y轴方向），减少观察维度
        return self.diff_body_ori_future_local[:, :, :, :2, :].reshape(self.num_envs, -1)

class diff_body_ang_vel_future_local(RobotTrackObservation):
    """
    未来身体角速度差异观察函数（局部坐标系）
    
    计算参考身体角速度与机器人身体角速度的差异，分别在各自的根节点坐标系中。
    这种差异表示机器人需要如何调整身体角速度来匹配参考运动。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储身体角速度差异的张量
        self.diff_body_ang_vel_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)
    
    def update(self):
        # 获取世界坐标系下的未来参考身体角速度
        ref_body_ang_vel_future_w = self.command_manager.ref_body_ang_vel_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        # 获取参考根节点在世界坐标系中的方向
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]
        # 获取机器人身体在世界坐标系中的角速度
        robot_body_ang_vel_w = self.command_manager.robot_body_ang_vel_w # shape: [num_envs, num_tracking_bodies, 3]
        # 获取机器人根节点在世界坐标系中的方向
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        # 只保留yaw旋转
        ref_root_quat_w = yaw_quat(ref_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        # 将参考身体角速度转换到参考根节点坐标系
        ref_body_ang_vel_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_ang_vel_future_w)
        # 将机器人身体角速度转换到机器人根节点坐标系
        robot_body_ang_vel_local = quat_apply_inverse(robot_root_quat_w, robot_body_ang_vel_w)

        # 计算差异（参考角速度 - 机器人角速度）
        self.diff_body_ang_vel_future_local = ref_body_ang_vel_future_local - robot_body_ang_vel_local.unsqueeze(1)

    def compute(self):
        return self.diff_body_ang_vel_future_local.view(self.num_envs, -1)

class ref_motion_phase(RobotTrackObservation):
    """
    参考运动相位观察函数
    
    返回当前运动在总运动长度中的相位（0到1之间的值）。
    用于让智能体了解当前运动执行的进度。
    """
    def compute(self):
        # 计算运动相位：当前时间 / 总运动长度
        return (self.command_manager.t / self.command_manager.motion_len).unsqueeze(1)


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """
    从四元数中提取yaw角度
    
    Args:
        quat: 四元数张量，形状为 [..., 4]
    
    Returns:
        yaw角度张量，形状为 [...]
    """
    qw, qx, qy, qz = torch.unbind(quat, dim=-1)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return yaw

# 定义机器人-物体交互跟踪观察函数的基类
RobotObjectTrackObservation = BaseObservation[RobotObjectTracking]

class ref_contact_pos_b(RobotObjectTrackObservation):
    """
    机器人根节点坐标系下的参考接触位置观察函数
    
    将世界坐标系下的参考末端执行器目标位置转换到机器人根节点坐标系中。
    支持添加噪声以提高训练的鲁棒性。
    
    Args:
        noise_std: 每步噪声标准差
        episodic_noise_std: 每个回合的噪声标准差
        yaw_only: 是否只考虑yaw旋转
    """
    def __init__(self, noise_std: float=0.0, episodic_noise_std: float=0.0, yaw_only: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = noise_std
        self.episodic_noise_std = episodic_noise_std
        self.yaw_only = yaw_only
        # 初始化存储转换后接触位置的张量
        self.ref_contact_pos_b = torch.zeros_like(self.command_manager.contact_target_pos_w)

        # 初始化噪声张量
        self.step_noise = torch.zeros_like(self.command_manager.contact_target_pos_w)
        self.episodic_noise = torch.zeros_like(self.command_manager.contact_target_pos_w)
    
    def reset(self, env_ids):
        """重置指定环境的回合噪声"""
        if self.episodic_noise_std > 0.0:
            self.episodic_noise[env_ids] = torch.empty(len(env_ids), *self.command_manager.contact_target_pos_w.shape[1:], device=self.device).uniform_(-1, 1) * self.episodic_noise_std
    
    def update(self):
        """更新接触位置观察"""
        # 生成每步噪声
        if self.noise_std > 0.0:
            self.step_noise = torch.randn_like(self.command_manager.contact_target_pos_w).clamp(-3, 3) * self.noise_std

        # 获取世界坐标系下的参考接触目标位置
        ref_contact_target_pos_w = self.command_manager.contact_target_pos_w # shape: [num_envs, n, 3]
        # 获取机器人根节点在世界坐标系中的位置和方向
        robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, :] # shape: [num_envs, 1, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        # 如果只考虑yaw旋转，则使用yaw-only四元数
        if self.yaw_only:
            robot_root_quat_w = yaw_quat(robot_root_quat_w)

        # 将接触位置转换到机器人根节点坐标系
        ref_contact_pos_b = quat_apply_inverse(robot_root_quat_w, ref_contact_target_pos_w - robot_root_pos_w)
        
        # 添加噪声（如果启用）
        if self.noise_std > 0.0:
            noise = torch.randn_like(ref_contact_pos_b).clamp(-1, 1) * self.noise_std
            ref_contact_pos_b += noise
        
        # 应用所有噪声
        self.ref_contact_pos_b = ref_contact_pos_b + self.episodic_noise + self.step_noise

    def compute(self):
        return self.ref_contact_pos_b.view(self.num_envs, -1)

class diff_contact_pos_b(RobotObjectTrackObservation):
    """
    机器人根节点坐标系下的接触位置差异观察函数
    
    计算参考接触目标位置与机器人末端执行器位置的差异，转换到机器人根节点坐标系中。
    这种差异表示机器人末端执行器需要移动多少距离才能到达目标接触位置。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储接触位置差异的张量
        self.diff_contact_pos_b = torch.zeros_like(self.command_manager.contact_target_pos_w)

    def update(self):
        # 获取世界坐标系下的参考接触目标位置
        ref_contact_target_pos_w = self.command_manager.contact_target_pos_w # shape: [num_envs, n, 3]
        # 获取机器人末端执行器在世界坐标系中的位置
        contact_eef_pos_w = self.command_manager.contact_eef_pos_w # shape: [num_envs, n, 3]
        # 获取机器人根节点在世界坐标系中的方向
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]
        
        # 计算世界坐标系下的位置差异
        diff_contact_pos_w = ref_contact_target_pos_w - contact_eef_pos_w
        # 将差异转换到机器人根节点坐标系
        self.diff_contact_pos_b = quat_apply_inverse(robot_root_quat_w, diff_contact_pos_w)

    def compute(self):
        return self.diff_contact_pos_b.view(self.num_envs, -1)
    
class object_xy_b(RobotObjectTrackObservation):
    """
    机器人根节点坐标系下的物体XY位置观察函数
    
    将世界坐标系下的物体位置转换到机器人根节点坐标系中，只保留XY坐标。
    支持添加噪声以提高训练的鲁棒性。
    
    Args:
        noise_std: 每步噪声标准差
        episodic_noise_std: 每个回合的噪声标准差
    """
    def __init__(self, noise_std: float=0.0, episodic_noise_std: float=0.0, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储物体XY位置的张量
        self.object_xy_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.noise_std = noise_std
        self.episodic_noise_std = episodic_noise_std

        # 初始化噪声张量
        self.step_noise = torch.zeros(self.num_envs, 2, device=self.device)
        self.episodic_noise = torch.zeros(self.num_envs, 2, device=self.device)

    def reset(self, env_ids):
        """重置指定环境的回合噪声"""
        if self.episodic_noise_std > 0.0:
            self.episodic_noise[env_ids] = torch.empty(len(env_ids), 2, device=self.device).uniform_(-1, 1) * self.episodic_noise_std

    def update(self):
        """更新物体XY位置观察"""
        # 生成每步噪声
        if self.noise_std > 0.0:
            self.step_noise = torch.randn_like(self.object_xy_b).clamp(-3, 3) * self.noise_std
        
        # 获取世界坐标系下的物体位置
        object_pos_w = self.command_manager.object.data.root_link_pos_w # shape: [num_envs, 3]
        # 获取机器人根节点在世界坐标系中的位置和方向
        robot_root_pos_w = self.command_manager.robot_root_pos_w # shape: [num_envs, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w # shape: [num_envs, 4]
        # 只考虑yaw旋转
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        # 将物体位置转换到机器人根节点坐标系，只保留XY坐标，并添加噪声
        self.object_xy_b = quat_apply_inverse(robot_root_quat_w, object_pos_w - robot_root_pos_w)[:, :2] + self.episodic_noise + self.step_noise

    def compute(self):
        return self.object_xy_b.view(self.num_envs, -1)

class object_heading_b(RobotObjectTrackObservation):
    """
    机器人根节点坐标系下的物体朝向观察函数
    
    计算物体相对于机器人根节点的yaw角度，并转换为cos/sin表示。
    支持添加噪声以提高训练的鲁棒性。
    
    Args:
        noise_std: 每步噪声标准差
        episodic_noise_std: 每个回合的噪声标准差
    """
    def __init__(self, noise_std: float=0.0, episodic_noise_std: float=0.0, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储物体yaw角度的张量
        self.object_yaw_b = torch.zeros(self.num_envs, 1, device=self.device)
        self.noise_std = noise_std
        self.episodic_noise_std = episodic_noise_std

        # 初始化噪声张量
        self.step_noise = torch.zeros_like(self.object_yaw_b)
        self.episodic_noise = torch.zeros_like(self.object_yaw_b)

    def reset(self, env_ids):
        """重置指定环境的回合噪声"""
        if self.episodic_noise_std > 0.0:
            self.episodic_noise[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(-1, 1) * self.episodic_noise_std

    def update(self):
        """更新物体朝向观察"""
        # 生成每步噪声
        if self.noise_std > 0.0:
            self.step_noise = torch.randn_like(self.object_yaw_b).clamp(-3, 3) * self.noise_std
        
        # 获取物体和机器人根节点在世界坐标系中的四元数
        object_quat_w = self.command_manager.object.data.root_link_quat_w # shape: [num_envs, 4]
        robot_root_quat_w = self.command_manager.robot_root_quat_w # shape: [num_envs, 4]

        # 提取yaw角度
        object_yaw_w = yaw_from_quat(object_quat_w)
        robot_root_yaw_w = yaw_from_quat(robot_root_quat_w)
        
        # 计算相对yaw角度并包装到[-π, π]范围内，添加噪声
        self.object_yaw_b = wrap_to_pi(object_yaw_w - robot_root_yaw_w)[:, None] + self.episodic_noise + self.step_noise

    def compute(self):
        # 将yaw角度转换为cos/sin表示，提供更稳定的观察
        object_heading_b = torch.cat([torch.cos(self.object_yaw_b), torch.sin(self.object_yaw_b)], dim=-1).view(self.num_envs, -1)
        return object_heading_b
    
    
class object_pos_b(RobotObjectTrackObservation):
    """
    机器人根节点坐标系下的物体位置观察函数
    
    将世界坐标系下的物体位置转换到机器人根节点坐标系中。
    用于让机器人了解物体相对于自身的位置。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储物体位置的张量
        self.object_pos_b = torch.zeros(self.num_envs, 3, device=self.device)

    def update(self):
        # 获取世界坐标系下的物体位置
        object_pos_w = self.command_manager.object.data.root_link_pos_w # shape: [num_envs, 3]
        # 获取机器人根节点在世界坐标系中的位置和方向
        robot_root_pos_w = self.command_manager.robot_root_pos_w # shape: [num_envs, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w # shape: [num_envs, 4]

        # 将物体位置转换到机器人根节点坐标系
        self.object_pos_b = quat_apply_inverse(robot_root_quat_w, object_pos_w - robot_root_pos_w)

    def compute(self):
        return self.object_pos_b.view(self.num_envs, -1)

class object_ori_b(RobotObjectTrackObservation):
    """
    机器人根节点坐标系下的物体方向观察函数
    
    将世界坐标系下的物体方向转换到机器人根节点坐标系中。
    用于让机器人了解物体相对于自身的方向。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储物体方向的张量
        self.object_ori_b = torch.zeros(self.num_envs, 3, 3, device=self.device)

    def update(self):
        # 获取世界坐标系下的物体和机器人根节点四元数
        object_quat_w = self.command_manager.object.data.root_link_quat_w # shape: [num_envs, 4]
        robot_root_quat_w = self.command_manager.robot_root_quat_w # shape: [num_envs, 4]

        # 将物体四元数转换到机器人根节点坐标系
        object_quat_b = quat_mul(
            quat_conjugate(robot_root_quat_w).expand_as(object_quat_w),
            object_quat_w
        )
        # 将四元数转换为旋转矩阵
        self.object_ori_b = matrix_from_quat(object_quat_b)

    def compute(self):
        return self.object_ori_b.view(self.num_envs, -1)
    
class object_joint_pos(RobotObjectTrackObservation):
    """
    物体关节位置观察函数
    
    返回物体的关节位置信息，用于了解物体的当前配置状态。
    """
    def compute(self):
        return self.command_manager.object_joint_pos.unsqueeze(1)

class object_joint_vel(RobotObjectTrackObservation):
    """
    物体关节速度观察函数
    
    返回物体的关节速度信息，用于了解物体的运动状态。
    """
    def compute(self):
        return self.command_manager.object_joint_vel.unsqueeze(1)

class object_joint_torque(RobotObjectTrackObservation):
    """
    物体关节扭矩观察函数
    
    返回物体关节的施加扭矩信息，用于了解物体的受力状态。
    """
    def compute(self):
        return self.command_manager.object.data.applied_torque

class object_joint_friction(RobotObjectTrackObservation):
    """
    物体关节摩擦观察函数
    
    返回物体关节的摩擦系数信息，用于了解物体的物理属性。
    """
    def compute(self):
        return self.command_manager.object._custom_friction.unsqueeze(1)

class object_joint_damping(RobotObjectTrackObservation):
    """
    物体关节阻尼观察函数
    
    返回物体关节的阻尼系数信息，用于了解物体的物理属性。
    """
    def compute(self):
        return self.command_manager.object._custom_damping.unsqueeze(1)

class diff_object_pos_future(RobotObjectTrackObservation):
    """
    未来物体位置差异观察函数（物体坐标系）
    
    计算未来参考物体位置与当前物体位置的差异，转换到物体坐标系中。
    这种差异表示物体需要移动多少距离才能到达目标位置。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储物体位置差异的张量
        self.diff_object_pos_future_b = torch.zeros(self.num_envs, self.command_manager.num_future_steps, 3, device=self.device)

    def update(self):
        # 获取世界坐标系下的未来参考物体位置
        ref_object_pos_future_w = self.command_manager.ref_object_pos_future_w # shape: [num_envs, num_future_steps, 3]
        # 获取当前物体在世界坐标系中的位置
        object_pos_w = self.command_manager.object.data.root_link_pos_w.unsqueeze(1)
        # 计算世界坐标系下的位置差异
        diff_object_pos_future_w = ref_object_pos_future_w - object_pos_w

        # 获取物体在世界坐标系中的方向
        object_quat_w = self.command_manager.object.data.root_quat_w.unsqueeze(1) # shape: [num_envs, 1, 4]
        # 将差异转换到物体坐标系
        self.diff_object_pos_future_b = quat_apply_inverse(object_quat_w, diff_object_pos_future_w)
    
    def compute(self):
        return self.diff_object_pos_future_b.view(self.num_envs, -1)

class diff_object_ori_future(RobotObjectTrackObservation):
    """
    未来物体方向差异观察函数（物体坐标系）
    
    计算未来参考物体方向与当前物体方向的差异，转换到物体坐标系中。
    这种差异表示物体需要旋转多少角度才能到达目标方向。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化存储物体方向差异的张量
        self.diff_object_ori_future_b = torch.zeros(self.num_envs, self.command_manager.num_future_steps, 3, 3, device=self.device)

    def update(self):
        # 获取世界坐标系下的未来参考物体四元数
        ref_object_quat_future_w = self.command_manager.ref_object_quat_future_w # shape: [num_envs, num_future_steps, 4]
        # 获取当前物体在世界坐标系中的四元数
        object_quat_w = self.command_manager.object.data.root_link_quat_w.unsqueeze(1) # shape: [num_envs, 1, 4]
        
        # 计算四元数差异（相对旋转）
        diff_object_quat_future = quat_mul(
            quat_conjugate(object_quat_w).expand_as(ref_object_quat_future_w),
            ref_object_quat_future_w
        )
        # 将四元数转换为旋转矩阵
        self.diff_object_ori_future_b = matrix_from_quat(diff_object_quat_future)

    def compute(self):
        return self.diff_object_ori_future_b.view(self.num_envs, -1)

class diff_object_joint_pos_future(RobotObjectTrackObservation):
    """
    未来物体关节位置差异观察函数
    
    计算参考物体关节位置与当前物体关节位置的差异。
    用于指导机器人如何调整物体关节以达到目标配置。
    """
    def compute(self):
        # 获取未来参考物体关节位置
        ref_object_joint_pos_future = self.command_manager.ref_object_joint_pos_future
        # 获取当前物体关节位置
        object_joint_pos = self.command_manager.object_joint_pos
        # 计算差异
        diff_object_joint_pos_future = ref_object_joint_pos_future - object_joint_pos.unsqueeze(1)
        return diff_object_joint_pos_future

class ref_object_contact_future(RobotObjectTrackObservation):
    """
    未来参考物体接触观察函数
    
    返回未来时间步的参考物体接触状态信息。
    用于指导机器人何时与物体进行接触交互。
    """
    def compute(self):
        return self.command_manager.ref_object_contact_future.view(self.num_envs, -1)
