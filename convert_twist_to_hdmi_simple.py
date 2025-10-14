#!/usr/bin/env python3
"""
将 TWIST SkeletonMotion 格式转换为 HDMI NPZ 格式（简化版本 - 直接使用原始数据）

这个版本只做最基本的格式转换，不进行任何角度转换或坐标系转换。
直接使用 TWIST 的原始数据：
- global_translation → body_pos_w
- global_rotation → body_quat_w
- linear_velocity → body_lin_vel_w
- dof_vels → joint_vel
- local_rotation 转换为简单的关节角度

注意：这要求 TWIST 和 HDMI 的坐标系和关节定义必须一致，否则可能需要后续调整。
"""

import numpy as np
import torch
import json
import sys
import os
from pathlib import Path
import yaml
from tqdm import tqdm
import argparse

# 添加 TWIST 的 pose 模块到路径
TWIST_PATH = "/home/ubuntu/DATA2/workspace/xmh/TWIST-master"
sys.path.insert(0, os.path.join(TWIST_PATH, "pose"))

try:
    from pose.poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
    print("✓ Successfully imported TWIST poselib")
except ImportError as e:
    print(f"✗ Failed to import TWIST poselib: {e}")
    print(f"  Make sure TWIST is installed at: {TWIST_PATH}")
    sys.exit(1)

# HDMI 的标准关节和身体名称
from active_adaptation.utils.motion import unitree_joint_names, unitree_body_names


def compute_angular_velocity(quat, fps):
    """
    从四元数序列计算角速度（使用有限差分）

    Args:
        quat: [T, 4] 四元数序列 (wxyz格式)
        fps: 帧率

    Returns:
        [T, 3] 角速度
    """
    dt = 1.0 / fps

    if len(quat) < 2:
        return np.zeros((len(quat), 3))

    # 使用简单的有限差分
    q1 = quat[:-1]
    q2 = quat[1:]

    # 计算角速度: ω ≈ 2/dt * [q1* ⊗ q2].xyz
    ang_vel = (2.0 / dt) * np.stack([
        q1[:, 0]*q2[:, 1] - q1[:, 1]*q2[:, 0] - q1[:, 2]*q2[:, 3] + q1[:, 3]*q2[:, 2],
        q1[:, 0]*q2[:, 2] + q1[:, 1]*q2[:, 3] - q1[:, 2]*q2[:, 0] - q1[:, 3]*q2[:, 1],
        q1[:, 0]*q2[:, 3] - q1[:, 1]*q2[:, 2] + q1[:, 2]*q2[:, 1] - q1[:, 3]*q2[:, 0]
    ], axis=-1)

    # 最后一帧使用倒数第二帧的速度
    ang_vel = np.vstack([ang_vel, ang_vel[-1:]])
    return ang_vel


def quaternion_to_simple_joint_angles(local_quat):
    """
    将四元数转换为简单的关节角度（使用轴角表示）

    这是一个简化的转换，假设关节主要绕一个轴旋转。
    对于更复杂的关节，可能需要更精确的转换。

    Args:
        local_quat: [T, N_joints, 4] 局部旋转四元数 (wxyz)

    Returns:
        [T, N_joints] 关节角度（弧度）
    """
    T, N = local_quat.shape[0], local_quat.shape[1]

    # 计算旋转角度：angle = 2 * arccos(w)
    w = local_quat[:, :, 0]  # [T, N]
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))

    # 计算旋转轴方向（用于确定角度符号）
    xyz = local_quat[:, :, 1:]  # [T, N, 3]
    axis_y = xyz[:, :, 1]  # 假设主要绕 Y 轴旋转

    # 根据轴的方向调整角度符号
    signed_angle = angle * np.sign(axis_y)

    return signed_angle


def convert_twist_motion_to_hdmi(
    npy_file: str,
    output_dir: str,
    target_fps: int = 50,
    verbose: bool = True
):
    """
    转换单个 TWIST motion 文件到 HDMI 格式（简化版本 - 直接使用原始数据）

    Args:
        npy_file: TWIST 的 .npy 文件路径
        output_dir: 输出目录
        target_fps: 目标帧率（HDMI 默认 50）
        verbose: 是否打印详细信息
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Converting: {npy_file}")
        print(f"{'='*60}")

    # 加载 SkeletonMotion
    motion = SkeletonMotion.from_file(npy_file)

    # 获取基本信息
    T = motion.tensor.shape[0]
    fps = motion.fps
    N_joints = motion.num_joints

    if verbose:
        print(f"  Original: {T} frames @ {fps} FPS, {N_joints} joints")

    # 如果帧率不匹配，进行插值
    if fps != target_fps:
        T_new = int(T * target_fps / fps)
        if verbose:
            print(f"  Resampling: {T} frames @ {fps} FPS -> {T_new} frames @ {target_fps} FPS")

        from scipy.interpolate import interp1d

        old_times = np.arange(T) / fps
        new_times = np.arange(T_new) / target_fps
        new_times = new_times[new_times <= old_times[-1]]
        T_new = len(new_times)

        # 插值 root translation
        root_trans_interp = interp1d(old_times, motion.root_translation.numpy(), axis=0, kind='linear')
        new_root_trans = root_trans_interp(new_times)

        # 插值 local rotation（线性插值后归一化）
        local_rot = motion.local_rotation.numpy()
        new_local_rot = np.zeros((T_new, N_joints, 4))
        for j in range(N_joints):
            for k in range(4):
                rot_interp = interp1d(old_times, local_rot[:, j, k], kind='linear')
                new_local_rot[:, j, k] = rot_interp(new_times)

        # 归一化四元数
        new_local_rot = new_local_rot / np.linalg.norm(new_local_rot, axis=-1, keepdims=True)

        # 创建新的 SkeletonMotion
        new_motion = SkeletonMotion.from_skeleton_state(
            SkeletonState.from_rotation_and_root_translation(
                skeleton_tree=motion.skeleton_tree,
                r=torch.from_numpy(new_local_rot),
                t=torch.from_numpy(new_root_trans),
                is_local=True
            ),
            fps=target_fps
        )
        motion = new_motion
        T = T_new
        fps = target_fps

    if verbose:
        print(f"  Final: {T} frames @ {fps} FPS")

    # ===== 直接提取 TWIST 原始数据（不做转换） =====

    # 1. Body positions (世界坐标系)
    body_pos_w = motion.global_translation.numpy()  # [T, N_joints, 3]

    # 2. Body orientations (世界坐标系四元数 - wxyz格式)
    body_quat_w = motion.global_rotation.numpy()  # [T, N_joints, 4]

    # 3. Body linear velocities (世界坐标系)
    body_lin_vel_w = motion.linear_velocity.numpy()  # [T, N_joints, 3]

    # 4. 计算角速度（从四元数）
    body_ang_vel_w = np.zeros_like(body_pos_w)  # [T, N_joints, 3]
    for i in range(N_joints):
        body_ang_vel_w[:, i] = compute_angular_velocity(body_quat_w[:, i], fps)

    # 5. Joint positions (关节角度)
    # 从 local_rotation 转换为简单的关节角度
    local_rotation = motion.local_rotation.numpy()  # [T, N_joints, 4]

    # 跳过 root（第一个关节），只转换实际的关节
    joint_angles = quaternion_to_simple_joint_angles(local_rotation[:, 1:])  # [T, N_joints-1]

    # 6. Joint velocities (关节速度)
    joint_vel = motion.dof_vels.numpy()  # [T, N_dofs]

    if verbose:
        print(f"  Extracted TWIST data shapes:")
        print(f"    body_pos_w: {body_pos_w.shape}")
        print(f"    body_quat_w: {body_quat_w.shape}")
        print(f"    body_lin_vel_w: {body_lin_vel_w.shape}")
        print(f"    body_ang_vel_w: {body_ang_vel_w.shape}")
        print(f"    joint_angles: {joint_angles.shape}")
        print(f"    joint_vel: {joint_vel.shape}")

    # ===== 调整数组大小以匹配 HDMI 格式 =====

    target_n_bodies = len(unitree_body_names)
    target_n_joints = len(unitree_joint_names)

    # 调整 body 相关数组
    if N_joints < target_n_bodies:
        # Pad with zeros
        pad_width = target_n_bodies - N_joints
        body_pos_w = np.pad(body_pos_w, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
        body_quat_w = np.pad(body_quat_w, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
        body_quat_w[:, N_joints:, 0] = 1.0  # 设置为单位四元数 (1, 0, 0, 0)
        body_lin_vel_w = np.pad(body_lin_vel_w, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
        body_ang_vel_w = np.pad(body_ang_vel_w, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
    elif N_joints > target_n_bodies:
        # Truncate
        body_pos_w = body_pos_w[:, :target_n_bodies]
        body_quat_w = body_quat_w[:, :target_n_bodies]
        body_lin_vel_w = body_lin_vel_w[:, :target_n_bodies]
        body_ang_vel_w = body_ang_vel_w[:, :target_n_bodies]

    # 调整 joint 相关数组
    joint_pos = joint_angles  # 使用转换后的关节角度

    if joint_pos.shape[1] < target_n_joints:
        pad_width = target_n_joints - joint_pos.shape[1]
        joint_pos = np.pad(joint_pos, ((0, 0), (0, pad_width)), mode='constant')
    elif joint_pos.shape[1] > target_n_joints:
        joint_pos = joint_pos[:, :target_n_joints]

    if joint_vel.shape[1] < target_n_joints:
        pad_width = target_n_joints - joint_vel.shape[1]
        joint_vel = np.pad(joint_vel, ((0, 0), (0, pad_width)), mode='constant')
    elif joint_vel.shape[1] > target_n_joints:
        joint_vel = joint_vel[:, :target_n_joints]

    # ===== 准备 HDMI 格式数据 =====
    motion_data = {
        "body_pos_w": body_pos_w.astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32),
        "body_lin_vel_w": body_lin_vel_w.astype(np.float32),
        "body_ang_vel_w": body_ang_vel_w.astype(np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
    }

    if verbose:
        print(f"  HDMI format data shapes:")
        for key, val in motion_data.items():
            print(f"    {key}: {val.shape}")

    # ===== 保存文件 =====
    os.makedirs(output_dir, exist_ok=True)

    # 保存 NPZ
    output_npz = os.path.join(output_dir, "motion.npz")
    np.savez_compressed(output_npz, **motion_data)

    if verbose:
        print(f"  ✓ Saved motion.npz ({os.path.getsize(output_npz) / 1024:.1f} KB)")

    # 获取 TWIST 的实际关节和身体名称
    skeleton = motion.skeleton_tree
    twist_body_names = [skeleton.node_names[i] for i in range(min(N_joints, target_n_bodies))]
    twist_joint_names = [f"joint_{i}" for i in range(min(N_joints-1, target_n_joints))]  # 简化的关节名称

    # 创建 meta.json
    meta = {
        "body_names": twist_body_names,
        "joint_names": twist_joint_names,
        "fps": float(fps),
        "source": "TWIST SkeletonMotion",
        "original_file": str(npy_file),
        "original_fps": float(motion.fps) if fps != motion.fps else float(fps),
        "n_frames": int(T),
        "n_bodies": body_pos_w.shape[1],
        "n_joints": joint_pos.shape[1]
    }

    output_meta = os.path.join(output_dir, "meta.json")
    with open(output_meta, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"  ✓ Saved meta.json")
        print(f"  ✓ Total output size: {(os.path.getsize(output_npz) + os.path.getsize(output_meta)) / 1024:.1f} KB")

    return output_dir


def convert_twist_yaml_to_hdmi(
    yaml_file: str,
    output_base_dir: str,
    target_fps: int = 50,
    max_motions: int = None
):
    """
    转换整个 TWIST dataset（从 YAML 配置）到 HDMI 格式

    Args:
        yaml_file: TWIST 的 YAML 配置文件
        output_base_dir: HDMI 数据输出根目录
        target_fps: 目标帧率
        max_motions: 最大转换数量（None = 全部）
    """
    print(f"\n{'='*80}")
    print(f"TWIST to HDMI Converter (Simplified - Direct Data Transfer)")
    print(f"{'='*80}")
    print(f"Input YAML: {yaml_file}")
    print(f"Output dir: {output_base_dir}")
    print(f"Target FPS: {target_fps}")
    print(f"{'='*80}\n")

    # 加载 YAML 配置
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    root_path = config.get('root_path', '')
    motions_config = config.get('motions', [])

    # 解析 motions
    if isinstance(motions_config, dict):
        motions = []
        for key, val in motions_config.items():
            if key == 'root':
                continue
            if isinstance(val, dict):
                motions.append({
                    'file': key,
                    'weight': val.get('weight', 1.0),
                    'description': val.get('description', '')
                })
            else:
                motions.append({'file': key, 'weight': 1.0, 'description': ''})
    else:
        motions = motions_config

    if max_motions is not None:
        motions = motions[:max_motions]

    print(f"Found {len(motions)} motions in YAML config")
    if max_motions:
        print(f"Converting first {max_motions} motions only")
    print()

    success_count = 0
    failed_files = []

    for i, motion_entry in enumerate(tqdm(motions, desc="Converting motions")):
        if isinstance(motion_entry, dict):
            motion_file = motion_entry['file']
        else:
            motion_file = motion_entry

        # 构建完整路径
        if not motion_file.endswith('.npy'):
            motion_file = motion_file + '.npy'

        full_path = os.path.join(root_path, motion_file)

        # 检查文件是否存在
        if not os.path.exists(full_path):
            print(f"\n⚠ Warning: File not found: {full_path}")
            failed_files.append(full_path)
            continue

        # 输出目录
        motion_name = Path(motion_file).stem
        output_dir = os.path.join(output_base_dir, f"{motion_name}")

        try:
            convert_twist_motion_to_hdmi(
                full_path,
                output_dir,
                target_fps=target_fps,
                verbose=False
            )
            success_count += 1
        except Exception as e:
            print(f"\n✗ Error converting {full_path}:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            failed_files.append(full_path)
            continue

    # 打印总结
    print(f"\n{'='*80}")
    print(f"Conversion Summary")
    print(f"{'='*80}")
    print(f"Total motions: {len(motions)}")
    print(f"Successfully converted: {success_count}")
    print(f"Failed: {len(failed_files)}")

    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")

    print(f"\n✓ Output directory: {output_base_dir}")
    print(f"{'='*80}\n")

    return output_base_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert TWIST SkeletonMotion to HDMI NPZ format (Simplified - Direct Transfer)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This simplified version directly uses TWIST's raw data without coordinate transformations:
- global_translation → body_pos_w
- global_rotation → body_quat_w
- linear_velocity → body_lin_vel_w
- angular_velocity (computed) → body_ang_vel_w
- local_rotation (simplified) → joint_pos
- dof_vels → joint_vel

Examples:
  # Convert entire TWIST dataset
  python convert_twist_to_hdmi_simple.py \\
      --yaml_config ../TWIST-master/legged_gym/motion_data_configs/twist_dataset.yaml \\
      --output data/motion/g1/twist_simple

  # Convert single motion file
  python convert_twist_to_hdmi_simple.py \\
      --input /path/to/motion.npy \\
      --output data/motion/g1/single_motion

  # Convert first 10 motions with custom FPS
  python convert_twist_to_hdmi_simple.py \\
      --yaml_config twist_dataset.yaml \\
      --output data/motion/g1/twist \\
      --target_fps 60 \\
      --max_motions 10
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--yaml_config", help="TWIST dataset YAML config file")
    group.add_argument("--input", help="Single TWIST .npy motion file")

    parser.add_argument("--output", required=True, help="Output directory for HDMI format")
    parser.add_argument("--target_fps", type=int, default=50, help="Target FPS (default: 50)")
    parser.add_argument("--max_motions", type=int, help="Maximum number of motions to convert")

    args = parser.parse_args()

    if args.yaml_config:
        convert_twist_yaml_to_hdmi(
            args.yaml_config,
            args.output,
            target_fps=args.target_fps,
            max_motions=args.max_motions
        )
    else:
        convert_twist_motion_to_hdmi(
            args.input,
            args.output,
            target_fps=args.target_fps,
            verbose=True
        )


if __name__ == "__main__":
    main()
