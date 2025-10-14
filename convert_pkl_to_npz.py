#!/usr/bin/env python3
"""
将GMR的pkl格式运动数据转换为HDMI项目使用的npz格式

PKL格式（GMR）包含：
- fps: 帧率
- root_pos: 根部位置 [T, 3]
- root_rot: 根部旋转四元数 [T, 4] (xyzw格式)
- dof_pos: 关节位置 [T, N_joints]
- local_body_pos: 局部身体位置 [T, N_bodies, 3]
- link_body_list: 身体链接列表

NPZ格式（HDMI）包含：
- body_pos_w: 世界坐标系下的身体位置 [T, N_bodies, 3]
- body_quat_w: 世界坐标系下的身体四元数 [T, N_bodies, 4] (wxyz格式)
- joint_pos: 关节位置 [T, N_joints]
- body_lin_vel_w: 身体线速度 [T, N_bodies, 3]
- body_ang_vel_w: 身体角速度 [T, N_bodies, 3]
- joint_vel: 关节速度 [T, N_joints]
- object_contact: 物体接触信息 [T, 1] (可选)

同时需要meta.json文件包含：
- body_names: 身体名称列表
- joint_names: 关节名称列表
- fps: 帧率
"""

import pickle
import numpy as np
import json
import argparse
import os
from pathlib import Path
from scipy.spatial.transform import Rotation


def compute_velocities(positions, dt):
    """计算速度（使用中心差分）"""
    velocities = np.zeros_like(positions)
    
    # 前向差分（第一帧）
    velocities[0] = (positions[1] - positions[0]) / dt
    
    # 中心差分（中间帧）
    for i in range(1, len(positions) - 1):
        velocities[i] = (positions[i + 1] - positions[i - 1]) / (2 * dt)
    
    # 后向差分（最后一帧）
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    
    return velocities


def quaternion_to_angular_velocity(quaternions, dt):
    """从四元数序列计算角速度"""
    angular_velocities = np.zeros((len(quaternions), 3))
    
    for i in range(len(quaternions) - 1):
        q1 = Rotation.from_quat(quaternions[i])
        q2 = Rotation.from_quat(quaternions[i + 1])
        
        # 计算相对旋转
        q_rel = q2 * q1.inv()
        
        # 转换为角速度
        rotvec = q_rel.as_rotvec()
        angular_velocities[i] = rotvec / dt
    
    # 最后一帧使用前一帧的角速度
    angular_velocities[-1] = angular_velocities[-2]
    
    return angular_velocities


def compute_forward_kinematics(root_rot_wxyz, motion_data, N_bodies, T, robot_name):
    """
    使用前向运动学计算每个身体部位的旋转
    
    这是一个简化的前向运动学实现。在实际应用中，需要：
    1. 机器人的运动学模型（关节层次结构）
    2. 每个关节的局部旋转
    3. 从根部到每个身体部位的变换链
    """
    print("Warning: Forward kinematics computation is simplified")
    print("For accurate results, use a proper robotics library like PyBullet or MuJoCo")
    
    body_quat_w = np.zeros((T, N_bodies, 4))
    
    # 简化的运动学链假设（仅作为示例）
    if robot_name == "unitree_g1":
        # G1机器人的简化运动学链
        # 这里只是示例，实际需要完整的运动学模型
        for t in range(T):
            # 根部（pelvis）
            body_quat_w[t, 0] = root_rot_wxyz[t]  # pelvis
            
            # 躯干部分 - 受腰部关节影响
            if N_bodies > 13:  # torso_link
                # 假设腰部关节影响躯干旋转
                waist_rotation = get_joint_rotation_contribution(motion_data, t, ["waist_yaw", "waist_roll", "waist_pitch"])
                torso_rot = combine_rotations(root_rot_wxyz[t], waist_rotation)
                body_quat_w[t, 13] = torso_rot
            
            # 其他身体部位使用根部旋转（简化）
            for b in range(N_bodies):
                if body_quat_w[t, b].sum() == 0:  # 如果还没有设置
                    body_quat_w[t, b] = root_rot_wxyz[t]
    else:
        # 对于其他机器人，使用根部旋转
        for t in range(T):
            for b in range(N_bodies):
                body_quat_w[t, b] = root_rot_wxyz[t]
    
    return body_quat_w


def get_joint_rotation_contribution(motion_data, time_idx, joint_names):
    """
    从dof_pos计算关节对旋转的贡献
    
    注意：这需要机器人的运动学模型才能准确计算
    """
    if "dof_pos" not in motion_data:
        return np.array([0, 0, 0, 1])  # 单位四元数
    
    dof_pos = motion_data["dof_pos"]
    if time_idx >= len(dof_pos):
        return np.array([0, 0, 0, 1])
    
    # 简化实现：仅考虑腰部关节的影响
    # 实际需要完整的运动学模型来正确计算
    joint_angles = dof_pos[time_idx]
    
    # 假设腰部关节是前几个关节（这是简化假设）
    if len(joint_angles) >= 3:
        # 使用前3个关节角度作为腰部旋转的近似
        yaw = joint_angles[0] if len(joint_angles) > 12 else 0    # waist_yaw
        roll = joint_angles[1] if len(joint_angles) > 13 else 0   # waist_roll  
        pitch = joint_angles[2] if len(joint_angles) > 14 else 0  # waist_pitch
        
        # 转换欧拉角到四元数 (简化，实际需要正确的关节轴向)
        from scipy.spatial.transform import Rotation
        euler_rot = Rotation.from_euler('zxy', [yaw, roll, pitch])
        quat_xyzw = euler_rot.as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # 转为wxyz
    
    return np.array([1, 0, 0, 0])  # 单位四元数


def combine_rotations(base_rot, additional_rot):
    """
    组合两个旋转（四元数乘法）
    """
    # 转换为Rotation对象进行乘法
    base_r = Rotation.from_quat([base_rot[1], base_rot[2], base_rot[3], base_rot[0]])  # wxyz to xyzw
    add_r = Rotation.from_quat([additional_rot[1], additional_rot[2], additional_rot[3], additional_rot[0]])
    
    combined = base_r * add_r
    combined_quat = combined.as_quat()  # xyzw
    
    return np.array([combined_quat[3], combined_quat[0], combined_quat[1], combined_quat[2]])  # 转回wxyz


def verify_coordinate_conversion(local_body_pos, body_pos_w, root_pos, frame_idx=0):
    """
    验证坐标系转换的正确性
    """
    print(f"\n=== 坐标转换验证 (第{frame_idx}帧) ===")
    
    # 检查根部位置
    root_local = local_body_pos[frame_idx, 0]  # GMR中根部应该是[0,0,0]或接近0
    root_world = body_pos_w[frame_idx, 0]      # 转换后的根部世界位置
    expected_root = root_pos[frame_idx]        # 期望的根部位置
    
    print(f"根部位置检查:")
    print(f"  GMR local_body_pos[{frame_idx}, 0]: {root_local}")
    print(f"  转换后 body_pos_w[{frame_idx}, 0]: {root_world}")
    print(f"  期望的 root_pos[{frame_idx}]: {expected_root}")
    print(f"  误差: {np.linalg.norm(root_world - expected_root):.6f}")
    
    # 检查相对距离保持
    if local_body_pos.shape[1] > 1:
        local_dist = np.linalg.norm(local_body_pos[frame_idx, 1] - local_body_pos[frame_idx, 0])
        world_dist = np.linalg.norm(body_pos_w[frame_idx, 1] - body_pos_w[frame_idx, 0])
        print(f"\n相对距离保持检查:")
        print(f"  局部坐标系中距离: {local_dist:.6f}")
        print(f"  世界坐标系中距离: {world_dist:.6f}")
        print(f"  距离误差: {abs(local_dist - world_dist):.6f}")
    
    return np.linalg.norm(root_world - expected_root) < 1e-3


def convert_pkl_to_npz(pkl_path, output_dir, robot_name="unitree_g1"):
    """
    将PKL格式转换为NPZ格式
    
    Args:
        pkl_path: PKL文件路径
        output_dir: 输出目录
        robot_name: 机器人名称，用于生成默认的关节和身体名称
    """
    
    # 加载PKL数据
    print(f"加载PKL文件: {pkl_path}")
    with open(pkl_path, "rb") as f:
        motion_data = pickle.load(f)
    
    # 提取基本信息
    fps = motion_data["fps"]
    dt = 1.0 / fps
    
    root_pos = motion_data["root_pos"]  # [T, 3]
    root_rot = motion_data["root_rot"]  # [T, 4] xyzw格式
    dof_pos = motion_data["dof_pos"]    # [T, N_joints]
    local_body_pos = motion_data["local_body_pos"]  # [T, N_bodies, 3]
    
    print(f"数据形状:")
    print(f"  root_pos: {root_pos.shape}")
    print(f"  root_rot: {root_rot.shape}")
    print(f"  dof_pos: {dof_pos.shape}")
    print(f"  local_body_pos: {local_body_pos.shape}")
    print(f"  fps: {fps}")
    
    # 检查PKL文件中是否包含额外的旋转数据
    available_keys = list(motion_data.keys())
    print(f"PKL文件包含的数据键: {available_keys}")
    
    rotation_keys = [k for k in available_keys if 'rot' in k.lower() or 'quat' in k.lower()]
    if rotation_keys:
        print(f"发现旋转相关数据: {rotation_keys}")
    else:
        print("未发现额外的旋转数据，将使用简化方法")
    
    T, N_bodies, _ = local_body_pos.shape
    N_joints = dof_pos.shape[1]
    
    # 转换四元数格式从xyzw到wxyz
    root_rot_wxyz = root_rot[:, [3, 0, 1, 2]]
    
    # 构建世界坐标系下的身体位置和旋转
    body_pos_w = np.zeros((T, N_bodies, 3))
    body_quat_w = np.zeros((T, N_bodies, 4))
    
    # 根部旋转矩阵
    root_rotations = Rotation.from_quat(root_rot)  # xyzw格式
    
    # 尝试从PKL数据中获取更详细的旋转信息
    if "body_quat_w" in motion_data:
        # 如果PKL文件包含身体四元数信息，直接使用
        pkl_body_quat = motion_data["body_quat_w"]  # 假设是wxyz格式
        if pkl_body_quat.shape[-1] == 4 and len(pkl_body_quat.shape) == 3:
            body_quat_w = pkl_body_quat
        else:
            print("Warning: body_quat_w format not recognized, using root rotation for all bodies")
            for t in range(T):
                for b in range(N_bodies):
                    body_quat_w[t, b] = root_rot_wxyz[t]
    elif "joint_rot" in motion_data or "body_rot" in motion_data:
        # 如果有关节旋转信息，进行前向运动学计算
        print("Found rotation data, attempting forward kinematics...")
        body_quat_w = compute_forward_kinematics(
            root_rot_wxyz, motion_data, N_bodies, T, robot_name
        )
    else:
        # 简化假设：所有身体部位使用根部旋转（不准确但可用）
        print("Warning: No detailed rotation data found, using simplified assumption")
        print("All body parts will have the same root rotation (may cause visualization issues)")
        for t in range(T):
            for b in range(N_bodies):
                body_quat_w[t, b] = root_rot_wxyz[t]
    
    # 将GMR的local_body_pos转换为LocoMujoco的body_pos_w
    print("转换坐标系: GMR local_body_pos → LocoMujoco body_pos_w")
    for t in range(T):
        # GMR local_body_pos: 相对于根部的局部坐标
        # LocoMujoco body_pos_w: 世界坐标系中的绝对位置
        # 转换公式: body_pos_w = root_rotation @ local_body_pos + root_pos
        
        rotated_local_pos = root_rotations[t].apply(local_body_pos[t])
        body_pos_w[t] = rotated_local_pos + root_pos[t:t+1]  # 广播加法
    
    # 验证转换正确性
    conversion_ok = verify_coordinate_conversion(local_body_pos, body_pos_w, root_pos, frame_idx=0)
    if not conversion_ok:
        print("Warning: 坐标转换可能存在问题，请检查数据")
    
    # 计算速度
    print("计算速度...")
    body_lin_vel_w = compute_velocities(body_pos_w, dt)
    joint_vel = compute_velocities(dof_pos, dt)
    
    # 计算角速度
    body_ang_vel_w = np.zeros((T, N_bodies, 3))
    for b in range(N_bodies):
        body_ang_vel_w[:, b] = quaternion_to_angular_velocity(
            body_quat_w[:, b, [1, 2, 3, 0]], dt  # 转换为xyzw格式计算
        )
    
    # 生成默认的身体和关节名称
    if robot_name == "unitree_g1":
        # G1机器人的默认关节名称
        joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", 
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]
        
        body_names = [
            "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
            "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
            "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
            "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
            "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
            "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
            "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
            "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link"
        ]
    else:
        # 生成通用名称
        joint_names = [f"joint_{i}" for i in range(N_joints)]
        body_names = [f"body_{i}" for i in range(N_bodies)]
    
    # 调整名称列表长度以匹配数据
    joint_names = joint_names[:N_joints] + [f"joint_{i}" for i in range(len(joint_names), N_joints)]
    body_names = body_names[:N_bodies] + [f"body_{i}" for i in range(len(body_names), N_bodies)]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存NPZ文件
    npz_path = os.path.join(output_dir, "motion.npz")
    print(f"保存NPZ文件: {npz_path}")
    
    np.savez_compressed(
        npz_path,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        joint_pos=dof_pos,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_vel=joint_vel
    )
    
    # 保存meta.json文件
    meta_path = os.path.join(output_dir, "meta.json")
    print(f"保存meta文件: {meta_path}")
    
    meta_data = {
        "body_names": body_names,
        "joint_names": joint_names,
        "fps": float(fps)
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=4)
    
    print(f"转换完成!")
    print(f"输出文件:")
    print(f"  - {npz_path}")
    print(f"  - {meta_path}")
    print(f"数据统计:")
    print(f"  - 帧数: {T}")
    print(f"  - 身体数量: {N_bodies}")
    print(f"  - 关节数量: {N_joints}")
    print(f"  - 帧率: {fps} Hz")
    print(f"  - 持续时间: {T/fps:.2f} 秒")


def main():
    parser = argparse.ArgumentParser(description="将GMR的PKL格式运动数据转换为HDMI项目的NPZ格式")
    parser.add_argument("--pkl_path", type=str, required=True, help="输入的PKL文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--robot_name", type=str, default="unitree_g1", 
                       help="机器人名称 (unitree_g1, 或其他)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(f"PKL文件不存在: {args.pkl_path}")
    
    convert_pkl_to_npz(args.pkl_path, args.output_dir, args.robot_name)


if __name__ == "__main__":
    main()
