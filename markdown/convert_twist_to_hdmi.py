#!/usr/bin/env python3
"""
将 TWIST SkeletonMotion 格式转换为 HDMI NPZ 格式

TWIST 格式：
- .npy 文件包含 SkeletonMotion 对象
- _key_bodies.npy 包含关键身体局部位置
- YAML 配置列出所有运动文件

HDMI 格式：
- motion.npz 包含所有运动数据
- meta.json 包含元数据
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
    from pose.poselib.poselib.core.rotation3d import quat_mul_norm, quat_inverse
    print("✓ Successfully imported TWIST poselib")
except ImportError as e:
    print(f"✗ Failed to import TWIST poselib: {e}")
    print(f"  Make sure TWIST is installed at: {TWIST_PATH}")
    sys.exit(1)

# HDMI 的标准关节和身体名称
from active_adaptation.utils.motion import unitree_joint_names, unitree_body_names


def quat_to_ang_vel(quat, fps):
    """
    从四元数序列计算角速度（使用有限差分）

    Args:
        quat: [T, 4] 四元数序列 (wxyz格式)
        fps: 帧率

    Returns:
        [T, 3] 角速度
    """
    dt = 1.0 / fps

    # 使用中心差分
    if len(quat) < 2:
        return np.zeros((len(quat), 3))

    q1 = quat[:-1]
    q2 = quat[1:]

    # 计算角速度: ω = 2/dt * [q1* ⊗ q2].xyz / q1.w
    # 更稳定的方法：使用四元数微分公式
    ang_vel = (2.0 / dt) * np.stack([
        q1[:, 0]*q2[:, 1] - q1[:, 1]*q2[:, 0] - q1[:, 2]*q2[:, 3] + q1[:, 3]*q2[:, 2],
        q1[:, 0]*q2[:, 2] + q1[:, 1]*q2[:, 3] - q1[:, 2]*q2[:, 0] - q1[:, 3]*q2[:, 1],
        q1[:, 0]*q2[:, 3] - q1[:, 1]*q2[:, 2] + q1[:, 2]*q2[:, 1] - q1[:, 3]*q2[:, 0]
    ], axis=-1)

    # 最后一帧使用倒数第二帧的速度
    ang_vel = np.vstack([ang_vel, ang_vel[-1:]])
    return ang_vel


def local_rotation_to_dof_pos(local_rot, dof_body_ids, dof_offsets):
    """
    将局部旋转转换为 DOF 位置（关节角度）

    Args:
        local_rot: [T, N_joints, 4] 局部旋转四元数
        dof_body_ids: DOF 对应的 body ID 列表
        dof_offsets: DOF 偏移量列表

    Returns:
        [T, N_dof] DOF 位置
    """
    T = local_rot.shape[0]
    num_dof = dof_offsets[-1]
    dof_pos = np.zeros((T, num_dof), dtype=np.float32)

    for j in range(len(dof_body_ids)):
        body_id = dof_body_ids[j]
        joint_offset = dof_offsets[j]
        joint_size = dof_offsets[j + 1] - joint_offset

        joint_q = local_rot[:, body_id]  # [T, 4]

        if joint_size == 3:
            # 3-DOF joint: 使用指数映射
            # 简化：使用轴角表示
            angle = 2.0 * np.arccos(np.clip(joint_q[:, 0], -1.0, 1.0))
            axis = joint_q[:, 1:4]
            axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
            axis_norm = np.where(axis_norm < 1e-6, 1.0, axis_norm)
            axis = axis / axis_norm
            exp_map = angle[:, None] * axis
            dof_pos[:, joint_offset:joint_offset + joint_size] = exp_map

        elif joint_size == 1:
            # 1-DOF joint: 绕 Y 轴旋转
            angle = 2.0 * np.arccos(np.clip(joint_q[:, 0], -1.0, 1.0))
            axis = joint_q[:, 1:4]
            # 投影到 Y 轴
            joint_theta = angle * np.sign(axis[:, 1])
            # 归一化到 [-π, π]
            joint_theta = np.arctan2(np.sin(joint_theta), np.cos(joint_theta))
            dof_pos[:, joint_offset] = joint_theta

        elif joint_size == 2:
            # 2-DOF joint
            angle = 2.0 * np.arccos(np.clip(joint_q[:, 0], -1.0, 1.0))
            axis = joint_q[:, 1:4]
            axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
            axis_norm = np.where(axis_norm < 1e-6, 1.0, axis_norm)
            axis = axis / axis_norm
            exp_map = angle[:, None] * axis
            dof_pos[:, joint_offset:joint_offset + joint_size] = exp_map[:, :2]
        else:
            raise ValueError(f"Unsupported joint size: {joint_size}")

    return dof_pos


def convert_twist_motion_to_hdmi(
    npy_file: str,
    output_dir: str,
    key_bodies_file: str = None,
    target_fps: int = 50,
    verbose: bool = True
):
    """
    转换单个 TWIST motion 文件到 HDMI 格式

    Args:
        npy_file: TWIST 的 .npy 文件路径
        output_dir: 输出目录
        key_bodies_file: _key_bodies.npy 文件路径（可选）
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
        # 重新采样到目标帧率
        T_new = int(T * target_fps / fps)
        if verbose:
            print(f"  Resampling: {T} frames @ {fps} FPS -> {T_new} frames @ {target_fps} FPS")

        # 使用 SkeletonMotion 的时间索引进行插值
        old_times = np.arange(T) / fps
        new_times = np.arange(T_new) / target_fps

        # 限制新时间不超过原始时间范围
        new_times = new_times[new_times <= old_times[-1]]
        T_new = len(new_times)

        # 创建新的 motion 对象（在新的时间点采样）
        from scipy.interpolate import interp1d

        # 插值 root translation
        root_trans_interp = interp1d(old_times, motion.root_translation.numpy(), axis=0, kind='linear')
        new_root_trans = root_trans_interp(new_times)

        # 插值 local rotation (使用 slerp)
        # 简化：使用线性插值（slerp 太慢）
        local_rot = motion.local_rotation.numpy()
        new_local_rot = np.zeros((T_new, N_joints, 4))
        for j in range(N_joints):
            for k in range(4):
                rot_interp = interp1d(old_times, local_rot[:, j, k], kind='linear')
                new_local_rot[:, j, k] = rot_interp(new_times)

        # 归一化四元数
        new_local_rot = new_local_rot / np.linalg.norm(new_local_rot, axis=-1, keepdims=True)

        # 创建新的 SkeletonState
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

    # 提取数据
    # 1. Global translation (body positions)
    global_translation = motion.global_translation.numpy()  # [T, N_joints, 3]

    # 2. Global rotation (body orientations)
    global_rotation = motion.global_rotation.numpy()  # [T, N_joints, 4] (wxyz)

    # 3. Local rotation
    local_rotation = motion.local_rotation.numpy()  # [T, N_joints, 4] (wxyz)

    # 4. Root translation and velocity
    root_translation = motion.root_translation.numpy()  # [T, 3]

    if verbose:
        print(f"  Extracted data shapes:")
        print(f"    global_translation: {global_translation.shape}")
        print(f"    global_rotation: {global_rotation.shape}")
        print(f"    local_rotation: {local_rotation.shape}")

    # 计算速度（使用有限差分）
    body_lin_vel_w = np.zeros_like(global_translation)
    if T > 1:
        body_lin_vel_w[:-1] = (global_translation[1:] - global_translation[:-1]) * fps
        body_lin_vel_w[-1] = body_lin_vel_w[-2]  # 最后一帧

    # 计算角速度
    body_ang_vel_w = np.zeros_like(global_translation)
    for i in range(N_joints):
        body_ang_vel_w[:, i] = quat_to_ang_vel(global_rotation[:, i], fps)

    # 转换为 DOF positions (joint angles)
    # TWIST 使用特定的 DOF 配置，这里需要根据 G1 机器人的配置
    # 简化处理：使用 G1 的标准配置

    # G1 的 DOF 配置（23 DOFs）
    # 腿部：6 DOF x 2 = 12
    # 腰部：3 DOF = 3
    # 手臂：4 DOF x 2 = 8
    # 总共 23 DOF

    # 定义 dof_body_ids 和 dof_offsets（根据 G1 配置）
    # 注意：这需要与 TWIST 的骨架结构对应
    dof_body_ids = list(range(1, 24))  # 身体 1-23（跳过 root）
    dof_offsets = list(range(24))  # 每个身体一个 DOF（简化）

    # 实际的 G1 配置更复杂，这里简化处理
    # 直接使用 local_rotation 的前 23 个关节
    joint_pos = np.zeros((T, 23), dtype=np.float32)

    # 简化：将四元数转换为轴角，取第一个分量
    for i in range(min(23, N_joints - 1)):  # 跳过 root
        quat = local_rotation[:, i + 1]  # body 0 是 root
        # 转换为轴角
        angle = 2.0 * np.arccos(np.clip(quat[:, 0], -1.0, 1.0))
        joint_pos[:, i] = angle

    # 计算关节速度
    joint_vel = np.zeros_like(joint_pos)
    if T > 1:
        joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) * fps
        joint_vel[-1] = joint_vel[-2]

    # 确保数组维度正确
    if N_joints < len(unitree_body_names):
        # Pad with zeros
        pad_width = len(unitree_body_names) - N_joints
        global_translation = np.pad(global_translation, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
        global_rotation = np.pad(global_rotation, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
        global_rotation[:, N_joints:, 0] = 1.0  # 设置为单位四元数
        body_lin_vel_w = np.pad(body_lin_vel_w, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
        body_ang_vel_w = np.pad(body_ang_vel_w, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
    elif N_joints > len(unitree_body_names):
        # Truncate
        global_translation = global_translation[:, :len(unitree_body_names)]
        global_rotation = global_rotation[:, :len(unitree_body_names)]
        body_lin_vel_w = body_lin_vel_w[:, :len(unitree_body_names)]
        body_ang_vel_w = body_ang_vel_w[:, :len(unitree_body_names)]

    # 确保关节数正确
    if joint_pos.shape[1] < len(unitree_joint_names):
        pad_width = len(unitree_joint_names) - joint_pos.shape[1]
        joint_pos = np.pad(joint_pos, ((0, 0), (0, pad_width)), mode='constant')
        joint_vel = np.pad(joint_vel, ((0, 0), (0, pad_width)), mode='constant')
    elif joint_pos.shape[1] > len(unitree_joint_names):
        joint_pos = joint_pos[:, :len(unitree_joint_names)]
        joint_vel = joint_vel[:, :len(unitree_joint_names)]

    # 准备 HDMI 格式数据
    motion_data = {
        "body_pos_w": global_translation.astype(np.float32),
        "body_quat_w": global_rotation.astype(np.float32),
        "body_lin_vel_w": body_lin_vel_w.astype(np.float32),
        "body_ang_vel_w": body_ang_vel_w.astype(np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
    }

    if verbose:
        print(f"  HDMI format data shapes:")
        for key, val in motion_data.items():
            print(f"    {key}: {val.shape}")

    # 保存 NPZ
    os.makedirs(output_dir, exist_ok=True)
    output_npz = os.path.join(output_dir, "motion.npz")
    np.savez_compressed(output_npz, **motion_data)

    if verbose:
        print(f"  ✓ Saved motion.npz ({os.path.getsize(output_npz) / 1024:.1f} KB)")

    # 创建 meta.json
    meta = {
        "body_names": unitree_body_names[:global_translation.shape[1]],
        "joint_names": unitree_joint_names[:joint_pos.shape[1]],
        "fps": float(fps)
    }

    output_meta = os.path.join(output_dir, "meta.json")
    with open(output_meta, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"  ✓ Saved meta.json")

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
    print(f"TWIST to HDMI Converter")
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

    # 解析 motions（可能是列表或字典）
    if isinstance(motions_config, dict):
        # 字典格式（新版）
        motions = []
        for key, val in motions_config.items():
            if key == 'root':
                continue
            if isinstance(val, dict):
                motions.append({
                    'file': key,
                    'weight': val.get('weight', 1.0),
                    'difficulty': val.get('difficulty', 0),
                    'description': val.get('description', '')
                })
            else:
                motions.append({'file': key, 'weight': 1.0, 'difficulty': 0, 'description': ''})
    else:
        # 列表格式（旧版）
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
            weight = motion_entry.get('weight', 1.0)
            description = motion_entry.get('description', '')
        else:
            motion_file = motion_entry
            weight = 1.0
            description = ''

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
        output_dir = os.path.join(output_base_dir, f"motion_{i:03d}_{motion_name}")

        try:
            convert_twist_motion_to_hdmi(
                full_path,
                output_dir,
                target_fps=target_fps,
                verbose=False  # 批量转换时不打印详细信息
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
        description="Convert TWIST SkeletonMotion format to HDMI NPZ format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert entire TWIST dataset
  python convert_twist_to_hdmi.py \\
      --yaml ../TWIST-master/legged_gym/motion_data_configs/twist_dataset.yaml \\
      --output data/motion/g1/twist

  # Convert single motion file
  python convert_twist_to_hdmi.py \\
      --npy /path/to/motion.npy \\
      --output data/motion/g1/single_motion

  # Convert first 10 motions only
  python convert_twist_to_hdmi.py \\
      --yaml twist_dataset.yaml \\
      --output data/motion/g1/twist_subset \\
      --max-motions 10
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--yaml", help="TWIST dataset YAML config file")
    group.add_argument("--npy", help="Single TWIST .npy motion file")

    parser.add_argument("--output", required=True, help="Output directory for HDMI format")
    parser.add_argument("--fps", type=int, default=50, help="Target FPS (default: 50)")
    parser.add_argument("--max-motions", type=int, help="Maximum number of motions to convert")

    args = parser.parse_args()

    if args.yaml:
        convert_twist_yaml_to_hdmi(
            args.yaml,
            args.output,
            target_fps=args.fps,
            max_motions=args.max_motions
        )
    else:
        convert_twist_motion_to_hdmi(
            args.npy,
            args.output,
            target_fps=args.fps,
            verbose=True
        )


if __name__ == "__main__":
    main()
