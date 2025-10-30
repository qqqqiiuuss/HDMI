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


class priv_info(RobotTrackObservation):
    """
    TWIST 特权信息观察函数

    提供特权信息用于训练 Teacher policy。

    根据 TWIST 论文，特权信息包含约 85 维:
    - 3: base_lin_vel (根节点线速度，世界坐标系)
    - 1: root_height (根节点高度)
    - 27 = 3*9: key_body_pos (9个关键点的当前3D位置)
    - 2: contact mask (左右脚接触状态)
    - 4: privileged latent (特权潜变量，用于蒸馏)
    - 48 = 2*23: joint info (关节力矩、刚度等信息)

    返回形状: [num_envs, ~85]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 延迟初始化索引，在第一次 compute() 调用时进行
        self.left_foot_indices = None
        self.right_foot_indices = None
        self.key_body_indices = None
        self._initialized = False

    def _lazy_init(self):
        """延迟初始化，在环境完全创建后调用"""
        if self._initialized:
            return

        # 获取机器人 articulation 对象
        robot = self.env.scene["robot"]

        # 检查是否有接触传感器
        self.has_contact_sensor = "contact_forces" in self.env.scene.sensors

        if self.has_contact_sensor:
            from isaaclab.sensors import ContactSensor
            self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

            # 查找左右脚的接触传感器索引
            left_foot_ids, _ = self.contact_sensor.find_bodies(".*left.*ankle.*")
            right_foot_ids, _ = self.contact_sensor.find_bodies(".*right.*ankle.*")

            self.left_foot_sensor_ids = left_foot_ids if left_foot_ids else []
            self.right_foot_sensor_ids = right_foot_ids if right_foot_ids else []
        else:
            # 如果没有接触传感器，使用高度判断（简化版本）
            self.left_foot_sensor_ids = []
            self.right_foot_sensor_ids = []

            # 找到左右脚的body索引用于高度判断
            left_foot_body_ids, _ = robot.find_bodies(".*left.*ankle.*")
            right_foot_body_ids, _ = robot.find_bodies(".*right.*ankle.*")

            self.left_foot_body_ids = left_foot_body_ids if left_foot_body_ids else []
            self.right_foot_body_ids = right_foot_body_ids if right_foot_body_ids else []

        # 获取跟踪的关键点索引
        self.key_body_indices = self.command_manager.tracking_body_indices_asset
        self._initialized = True

    def compute(self):
        # 延迟初始化
        self._lazy_init()

        # 获取机器人 articulation 对象
        robot = self.env.scene["robot"]

        # 1. 根节点线速度 (世界坐标系) - 3 dims
        base_lin_vel = robot.data.root_lin_vel_w  # [num_envs, 3]

        # 2. 根节点高度 - 1 dim
        root_height = robot.data.root_pos_w[:, 2:3]  # [num_envs, 1]

        # 3. 关键点位置 (当前时刻) - 27 dims (9个关键点 × 3D)
        key_body_pos = robot.data.body_pos_w[:, self.key_body_indices, :]  # [num_envs, 9, 3]
        key_body_pos_flat = key_body_pos.reshape(self.num_envs, -1)  # [num_envs, 27]

        # 4. 接触掩码 (左右脚) - 2 dims
        left_foot_contact = torch.zeros(self.num_envs, 1, device=self.device)
        right_foot_contact = torch.zeros(self.num_envs, 1, device=self.device)

        if self.has_contact_sensor:
            # 使用接触传感器判断接触
            if len(self.left_foot_sensor_ids) > 0:
                # 获取接触力的法向分量 (z方向)
                contact_force_data = self.contact_sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
                left_forces_z = contact_force_data[:, self.left_foot_sensor_ids, 2]  # [num_envs, num_left_feet]
                left_foot_contact = (left_forces_z.abs().max(dim=1, keepdim=True)[0] > 1.0).float()

            if len(self.right_foot_sensor_ids) > 0:
                contact_force_data = self.contact_sensor.data.net_forces_w
                right_forces_z = contact_force_data[:, self.right_foot_sensor_ids, 2]
                right_foot_contact = (right_forces_z.abs().max(dim=1, keepdim=True)[0] > 1.0).float()
        else:
            # 使用高度判断接触（简化版本）
            if len(self.left_foot_body_ids) > 0:
                left_foot_heights = robot.data.body_pos_w[:, self.left_foot_body_ids, 2]  # [num_envs, num_left_feet]
                left_foot_contact = (left_foot_heights.min(dim=1, keepdim=True)[0] < 0.05).float()

            if len(self.right_foot_body_ids) > 0:
                right_foot_heights = robot.data.body_pos_w[:, self.right_foot_body_ids, 2]
                right_foot_contact = (right_foot_heights.min(dim=1, keepdim=True)[0] < 0.05).float()

        contact_mask = torch.cat([left_foot_contact, right_foot_contact], dim=1)  # [num_envs, 2]

        # 5. 特权潜变量 - 4 dims (占位符，用于后续蒸馏训练)

        # 6. 关节额外信息 - 48 dims (2*24, 这里用关节力矩和位置作为近似)
        joint_torques = robot.data.applied_torque  # [num_envs, num_joints]
        joint_pos = robot.data.joint_pos  # [num_envs, num_joints]

        # 拼接所有特权信息
        priv_info = torch.cat([
            base_lin_vel,         # 3 dims
            root_height,          # 1 dim
            key_body_pos_flat,    # 27 dims
            contact_mask,         # 2 dims
            joint_torques,        # num_joints dims
            joint_pos,            # num_joints dims
        ], dim=1)

        return priv_info


# ============================= PAPER-SPECIFIED OBSERVATIONS =============================
# The following observation functions implement the TWIST paper specification:
# - Proprioceptive: [θt, ωt, qt, q˙t, ahist_t]_{t-10:t} (11 frames)
# - Reference motion: [p̂t, θ̂t, q̂t]_{t-10:t+10} (21 frames)
# ========================================================================================

class proprio_history_combined(RobotTrackObservation):
    """
    论文规定的本体感受观察历史 (Proprioceptive History)

    根据 TWIST 论文，本体感受观察包含 [θt, ωt, qt, q˙t, ahist_t]_{t-10:t}，共 11 帧历史。
    - θt: 根节点方向 (旋转矩阵前两行: 2×3=6 dims)
    - ωt: 根节点角速度 (3 dims)
    - qt: 关节位置 (num_joints dims, 论文中为 29, G1 为 23)
    - q˙t: 关节速度 (num_joints dims)
    - ahist_t: 动作历史 (action_dim dims)

    总维度: 11 frames × (6 + 3 + num_joints + num_joints + action_dim) dims
    对于 G1 (23 joints): 11 × (6 + 3 + 23 + 23 + 23) = 11 × 78 = 858 dims

    Args:
        history_length: 历史长度（论文中为 11，包含 t-10 到 t）
        root_ori_noise: 根节点方向噪声标准差 (θt)
        root_ang_vel_noise: 根节点角速度噪声标准差 (ωt)
        joint_pos_noise: 关节位置噪声标准差 (qt)
        joint_vel_noise: 关节速度噪声标准差 (q˙t)
        action_noise: 动作历史噪声标准差 (ahist_t)
    """
    def __init__(
        self,
        history_length: int = 11,
        root_ori_noise: float = 0.0,
        root_ang_vel_noise: float = 0.0,
        joint_pos_noise: float = 0.0,
        joint_vel_noise: float = 0.0,
        action_noise: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.history_length = history_length

        # 细粒度噪声配置
        self.root_ori_noise = root_ori_noise
        self.root_ang_vel_noise = root_ang_vel_noise
        self.joint_pos_noise = joint_pos_noise
        self.joint_vel_noise = joint_vel_noise
        self.action_noise = action_noise

        # 延迟初始化，等待动作管理器就绪
        self.action_dim = None
        self.num_joints = None
        self.proprio_dim = None

        # 历史缓冲区
        self.history_buffer = None
        self._initialized = False

    def _lazy_init(self):
        """延迟初始化，在环境完全创建后调用"""
        if self._initialized:
            return

        # 获取机器人 articulation 对象
        robot = self.env.scene["robot"]

        # 获取动作维度和关节数量
        self.action_dim = self.env.action_manager.action_dim
        self.num_joints = len(robot.data.joint_pos[0])

        # 计算单帧本体感受观察维度
        # θt (2x3=6) + ωt (3) + qt (num_joints) + q˙t (num_joints) + ahist_t (action_dim)
        self.proprio_dim = 6 + 3 + self.num_joints + self.num_joints + self.action_dim

        # 初始化历史缓冲区: [num_envs, history_length, proprio_dim]
        self.history_buffer = torch.zeros(
            self.num_envs, self.history_length, self.proprio_dim,
            device=self.device
        )

        self._initialized = True

    def reset(self, env_ids):
        """重置指定环境的历史缓冲区"""
        if self._initialized:
            self.history_buffer[env_ids] = 0.0

    def update(self):
        """更新历史缓冲区"""
        self._lazy_init()

        # 获取机器人 articulation 对象
        robot = self.env.scene["robot"]

        # 1. 根节点方向 θt (旋转矩阵前两行: 2×3=6 dims)
        root_quat_b = robot.data.root_quat_w  # [num_envs, 4]
        root_rot_mat = matrix_from_quat(root_quat_b)  # [num_envs, 3, 3]
        root_ori = root_rot_mat[:, :2, :].reshape(self.num_envs, 6)  # [num_envs, 6]

        # 2. 根节点角速度 ωt (3 dims)
        root_ang_vel = robot.data.root_ang_vel_b  # [num_envs, 3]

        # 3. 关节位置 qt (num_joints dims)
        joint_pos = robot.data.joint_pos  # [num_envs, num_joints]

        # 4. 关节速度 q˙t (num_joints dims)
        joint_vel = robot.data.joint_vel  # [num_envs, num_joints]

        # 5. 动作历史 ahist_t (action_dim dims)
        # 从动作管理器获取最近的动作
        action_hist = self.env.action_manager.action_buf[:, :, 0]  # [num_envs, action_dim]

        # 拼接当前帧的本体感受观察
        current_proprio = torch.cat([
            root_ori,       # 6 dims
            root_ang_vel,   # 3 dims
            joint_pos,      # num_joints dims
            joint_vel,      # num_joints dims
            action_hist,    # action_dim dims
        ], dim=1)  # [num_envs, proprio_dim]

        # 将历史向前移动一帧（最旧的帧被丢弃）
        self.history_buffer[:, :-1, :] = self.history_buffer[:, 1:, :].clone()
        # 将当前帧添加到最新位置
        self.history_buffer[:, -1, :] = current_proprio

    def compute(self):
        """返回完整的本体感受历史观察，为不同组件应用细粒度噪声"""
        self._lazy_init()

        # 获取展平的历史缓冲区: [num_envs, history_length * proprio_dim]
        obs = self.history_buffer.view(self.num_envs, -1)

        # 应用细粒度噪声到不同组件
        # 组件在 proprio_dim 中的顺序: [root_ori(6), root_ang_vel(3), joint_pos(n), joint_vel(n), action(n)]
        if any([self.root_ori_noise > 0, self.root_ang_vel_noise > 0,
                self.joint_pos_noise > 0, self.joint_vel_noise > 0, self.action_noise > 0]):

            # 重塑为 [num_envs, history_length, proprio_dim] 以便分别处理每个组件
            obs_reshaped = obs.view(self.num_envs, self.history_length, self.proprio_dim)

            # 计算每个组件的起始索引
            root_ori_start = 0
            root_ori_end = 6
            root_ang_vel_start = 6
            root_ang_vel_end = 9
            joint_pos_start = 9
            joint_pos_end = 9 + self.num_joints
            joint_vel_start = 9 + self.num_joints
            joint_vel_end = 9 + 2 * self.num_joints
            action_start = 9 + 2 * self.num_joints
            action_end = 9 + 2 * self.num_joints + self.action_dim

            # 1. 根节点方向噪声 (θt): 6 dims
            if self.root_ori_noise > 0.0:
                noise = torch.randn_like(obs_reshaped[:, :, root_ori_start:root_ori_end]).clamp(-3, 3)
                obs_reshaped[:, :, root_ori_start:root_ori_end] += noise * self.root_ori_noise

            # 2. 根节点角速度噪声 (ωt): 3 dims
            if self.root_ang_vel_noise > 0.0:
                noise = torch.randn_like(obs_reshaped[:, :, root_ang_vel_start:root_ang_vel_end]).clamp(-3, 3)
                obs_reshaped[:, :, root_ang_vel_start:root_ang_vel_end] += noise * self.root_ang_vel_noise

            # 3. 关节位置噪声 (qt): num_joints dims
            if self.joint_pos_noise > 0.0:
                noise = torch.randn_like(obs_reshaped[:, :, joint_pos_start:joint_pos_end]).clamp(-3, 3)
                obs_reshaped[:, :, joint_pos_start:joint_pos_end] += noise * self.joint_pos_noise

            # 4. 关节速度噪声 (q˙t): num_joints dims
            if self.joint_vel_noise > 0.0:
                noise = torch.randn_like(obs_reshaped[:, :, joint_vel_start:joint_vel_end]).clamp(-3, 3)
                obs_reshaped[:, :, joint_vel_start:joint_vel_end] += noise * self.joint_vel_noise

            # 5. 动作历史噪声 (ahist_t): action_dim dims
            if self.action_noise > 0.0:
                noise = torch.randn_like(obs_reshaped[:, :, action_start:action_end]).clamp(-3, 3)
                obs_reshaped[:, :, action_start:action_end] += noise * self.action_noise

            # 展平回原始形状
            obs = obs_reshaped.view(self.num_envs, -1)

        return obs


class ref_motion_windowed(RobotTrackObservation):
    """
    论文规定的参考运动窗口观察 (Reference Motion Window)

    根据 TWIST 论文，参考运动包含 [p̂t, θ̂t, q̂t]_{t-10:t+10}，共 21 帧窗口。
    - p̂t: 参考根节点位置 (3 dims)
    - θ̂t: 参考根节点方向 (旋转矩阵前两行: 2×3=6 dims)
    - q̂t: 参考关节位置 (num_joints dims)

    总维度: 21 frames × (3 + 6 + num_joints) dims
    对于 G1 (23 joints): 21 × (3 + 6 + 23) = 21 × 32 = 672 dims

    Args:
        past_frames: 过去帧数（论文中为 10）
        future_frames: 未来帧数（论文中为 10）
        coordinate_frame: 坐标系选择 ('robot_root' 或 'motion_root')
        ref_root_pos_noise: 参考根节点位置噪声标准差 (p̂t)
        ref_root_ori_noise: 参考根节点方向噪声标准差 (θ̂t)
        ref_joint_pos_noise: 参考关节位置噪声标准差 (q̂t)
    """
    def __init__(
        self,
        past_frames: int = 10,
        future_frames: int = 10,
        coordinate_frame: str = 'robot_root',  # 'robot_root' or 'motion_root'
        ref_root_pos_noise: float = 0.0,
        ref_root_ori_noise: float = 0.0,
        ref_joint_pos_noise: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.window_length = past_frames + 1 + future_frames  # t-10 to t+10 = 21 frames
        self.coordinate_frame = coordinate_frame

        # 细粒度噪声配置
        self.ref_root_pos_noise = ref_root_pos_noise
        self.ref_root_ori_noise = ref_root_ori_noise
        self.ref_joint_pos_noise = ref_joint_pos_noise

        # 延迟初始化
        self.num_joints = None
        self.ref_motion_dim = None

        # 历史缓冲区（用于存储过去的参考运动）
        self.past_buffer = None
        # 当前帧的参考运动窗口
        self.ref_motion_window_b = None
        self._initialized = False

    def _lazy_init(self):
        """延迟初始化"""
        if self._initialized:
            return

        # 获取关节数量
        self.num_joints = len(self.command_manager.dataset.joint_names)

        # 计算单帧参考运动观察维度
        # p̂t (3) + θ̂t (2x3=6) + q̂t (num_joints)
        self.ref_motion_dim = 3 + 6 + self.num_joints

        # 初始化过去帧缓冲区: [num_envs, past_frames, ref_motion_dim]
        self.past_buffer = torch.zeros(
            self.num_envs, self.past_frames, self.ref_motion_dim,
            device=self.device
        )

        # 初始化完整窗口缓冲区: [num_envs, window_length, ref_motion_dim]
        self.ref_motion_window_b = torch.zeros(
            self.num_envs, self.window_length, self.ref_motion_dim,
            device=self.device
        )

        self._initialized = True

    def reset(self, env_ids):
        """重置指定环境的缓冲区"""
        if self._initialized:
            self.past_buffer[env_ids] = 0.0
            self.ref_motion_window_b[env_ids] = 0.0

    def update(self):
        """更新参考运动窗口"""
        self._lazy_init()

        # === 1. 构造当前帧的参考运动 ===
        ref_root_pos_w = self.command_manager.ref_root_pos_w  # [num_envs, 3]
        ref_root_quat_w = self.command_manager.ref_root_quat_w  # [num_envs, 4]
        ref_joint_pos = self.command_manager.current_ref_motion.joint_pos  # [num_envs, num_joints]

        # 转换为旋转矩阵（只取前两行）
        ref_root_rot_mat = matrix_from_quat(ref_root_quat_w)  # [num_envs, 3, 3]
        ref_root_ori = ref_root_rot_mat[:, :2, :].reshape(self.num_envs, 6)  # [num_envs, 6]

        # 拼接当前帧
        current_ref = torch.cat([
            ref_root_pos_w,   # 3 dims
            ref_root_ori,     # 6 dims
            ref_joint_pos,    # num_joints dims
        ], dim=1)  # [num_envs, ref_motion_dim]

        # === 2. 获取未来帧的参考运动 ===
        # 从 command_manager 获取未来参考数据
        ref_root_pos_future_w = self.command_manager.ref_root_pos_future_w  # [num_envs, num_future_steps, 3]
        ref_root_quat_future_w = self.command_manager.ref_root_quat_future_w  # [num_envs, num_future_steps, 4]
        ref_joint_pos_future = self.command_manager.ref_joint_pos_future_  # [num_envs, num_future_steps, num_joints]

        # 只取我们需要的未来帧数
        num_available_future = ref_root_pos_future_w.shape[1]
        num_future_to_use = min(self.future_frames, num_available_future)

        # 构造未来帧
        future_ref_list = []
        for i in range(num_future_to_use):
            future_root_pos = ref_root_pos_future_w[:, i, :]  # [num_envs, 3]
            future_root_quat = ref_root_quat_future_w[:, i, :]  # [num_envs, 4]
            future_joint_pos = ref_joint_pos_future[:, i, :]  # [num_envs, num_joints]

            # 转换方向为旋转矩阵
            future_root_rot_mat = matrix_from_quat(future_root_quat)
            future_root_ori = future_root_rot_mat[:, :2, :].reshape(self.num_envs, 6)

            future_ref = torch.cat([
                future_root_pos,    # 3 dims
                future_root_ori,    # 6 dims
                future_joint_pos,   # num_joints dims
            ], dim=1)  # [num_envs, ref_motion_dim]

            future_ref_list.append(future_ref.unsqueeze(1))

        # 如果可用未来帧数不足，用当前帧填充
        for i in range(num_future_to_use, self.future_frames):
            future_ref_list.append(current_ref.unsqueeze(1))

        future_ref = torch.cat(future_ref_list, dim=1)  # [num_envs, future_frames, ref_motion_dim]

        # === 3. 组合完整窗口: [过去帧, 当前帧, 未来帧] ===
        self.ref_motion_window_b = torch.cat([
            self.past_buffer,                      # [num_envs, past_frames, ref_motion_dim]
            current_ref.unsqueeze(1),              # [num_envs, 1, ref_motion_dim]
            future_ref,                            # [num_envs, future_frames, ref_motion_dim]
        ], dim=1)  # [num_envs, window_length, ref_motion_dim]

        # === 4. 转换到指定坐标系 ===
        if self.coordinate_frame == 'robot_root':
            # 转换到机器人根节点坐标系
            robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, :]  # [num_envs, 1, 3]
            robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :]  # [num_envs, 1, 4]

            # 转换位置
            ref_pos_w = self.ref_motion_window_b[:, :, :3]  # [num_envs, window_length, 3]
            ref_pos_b = quat_apply_inverse(robot_root_quat_w, ref_pos_w - robot_root_pos_w)
            self.ref_motion_window_b[:, :, :3] = ref_pos_b

            # 方向和关节位置保持不变（相对观察）

        # === 5. 更新过去帧缓冲区 ===
        # 将历史向前移动一帧
        self.past_buffer[:, :-1, :] = self.past_buffer[:, 1:, :].clone()
        # 将当前帧添加到过去帧缓冲区的最新位置
        self.past_buffer[:, -1, :] = current_ref

    def compute(self):
        """返回完整的参考运动窗口观察，为不同组件应用细粒度噪声"""
        self._lazy_init()

        # 获取展平的窗口: [num_envs, window_length * ref_motion_dim]
        obs = self.ref_motion_window_b.view(self.num_envs, -1)

        # 应用细粒度噪声到不同组件
        # 组件在 ref_motion_dim 中的顺序: [root_pos(3), root_ori(6), joint_pos(n)]
        if any([self.ref_root_pos_noise > 0, self.ref_root_ori_noise > 0, self.ref_joint_pos_noise > 0]):

            # 重塑为 [num_envs, window_length, ref_motion_dim] 以便分别处理每个组件
            obs_reshaped = obs.view(self.num_envs, self.window_length, self.ref_motion_dim)

            # 计算每个组件的起始索引
            root_pos_start = 0
            root_pos_end = 3
            root_ori_start = 3
            root_ori_end = 9
            joint_pos_start = 9
            joint_pos_end = 9 + self.num_joints

            # 1. 参考根节点位置噪声 (p̂t): 3 dims
            if self.ref_root_pos_noise > 0.0:
                noise = torch.randn_like(obs_reshaped[:, :, root_pos_start:root_pos_end]).clamp(-3, 3)
                obs_reshaped[:, :, root_pos_start:root_pos_end] += noise * self.ref_root_pos_noise

            # 2. 参考根节点方向噪声 (θ̂t): 6 dims
            if self.ref_root_ori_noise > 0.0:
                noise = torch.randn_like(obs_reshaped[:, :, root_ori_start:root_ori_end]).clamp(-3, 3)
                obs_reshaped[:, :, root_ori_start:root_ori_end] += noise * self.ref_root_ori_noise

            # 3. 参考关节位置噪声 (q̂t): num_joints dims
            if self.ref_joint_pos_noise > 0.0:
                noise = torch.randn_like(obs_reshaped[:, :, joint_pos_start:joint_pos_end]).clamp(-3, 3)
                obs_reshaped[:, :, joint_pos_start:joint_pos_end] += noise * self.ref_joint_pos_noise

            # 展平回原始形状
            obs = obs_reshaped.view(self.num_envs, -1)

        return obs
