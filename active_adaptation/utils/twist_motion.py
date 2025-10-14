import torch
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from tensordict import TensorClass, MemoryMappedTensor
# from tensordict import tensorclass, MemoryMappedTensor
from typing import List, Union
from scipy.spatial.transform import Rotation as sRot, Slerp
from isaaclab.utils.string import resolve_matching_names
from omegaconf import ListConfig
import re
import os
import glob
import pickle

unitree_joint_names =  [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]

unitree_body_names = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

def lerp(ts_target, ts_source, x):
    return np.stack([np.interp(ts_target, ts_source, x[:, i]) for i in range(x.shape[1])], axis=-1)


def slerp(ts_target, ts_source, quat):
    # time dim: 0
    # batch dim: 1:-1
    # quat dim: -1
    # for each batch dim, do the slerp
    batch_shape = quat.shape[1:-1]
    quat_dim = quat.shape[-1]

    steps_target = ts_target.shape[0]
    steps_source = ts_source.shape[0]

    quat = quat.reshape(steps_source, -1, quat_dim)

    batch_size = int(np.prod(batch_shape, initial=1))
    out = np.empty((steps_target, batch_size, quat_dim))
    for i in range(batch_size):
        s = Slerp(ts_source, sRot.from_quat(quat[:, i, [1, 2, 3, 0]])) # quat first to quat last
        out[:, i, :] = s(ts_target).as_quat()[..., [3, 0, 1, 2]] # quat last to quat first
    out = out.reshape(steps_target, *batch_shape, quat_dim)
    return out


def interpolate(motion, source_fps: int, target_fps: int):
    if source_fps != target_fps:
        # 必须插值的字段
        required_keys = ["body_pos_w", "body_lin_vel_w", "body_quat_w", "body_ang_vel_w", "joint_pos", "joint_vel"]
        # 可选的 GMR 字段
        optional_keys = ["root_pos", "root_rot", "local_body_pos"]

        # 检查未知的字段
        all_known_keys = set(required_keys + optional_keys)
        extra_keys = set(motion.keys()) - all_known_keys
        if extra_keys:
            print(f"  [Warning] Skipping interpolation for unknown keys: {extra_keys}")

        T = motion["joint_pos"].shape[0]
        # 使用 linspace 确保准确的样本数量
        duration = (T - 1) / source_fps  # 总时长
        ts_source = np.linspace(0, duration, T)  # 确保恰好 T 个样本

        # 计算目标帧数
        T_target = int(duration * target_fps) + 1
        ts_target = np.linspace(0, duration, T_target)

        # 插值必需字段
        motion["body_pos_w"] = lerp(ts_target, ts_source, motion["body_pos_w"].reshape(T, -1)).reshape(len(ts_target), -1, 3)
        motion["body_lin_vel_w"] = lerp(ts_target, ts_source, motion["body_lin_vel_w"].reshape(T, -1)).reshape(len(ts_target), -1, 3)
        motion["body_quat_w"] = slerp(ts_target, ts_source, motion["body_quat_w"])
        motion["body_ang_vel_w"] = lerp(ts_target, ts_source, motion["body_ang_vel_w"].reshape(T, -1)).reshape(len(ts_target), -1, 3)
        motion["joint_pos"] = lerp(ts_target, ts_source, motion["joint_pos"])
        motion["joint_vel"] = lerp(ts_target, ts_source, motion["joint_vel"])

        # 插值可选的 GMR 字段
        if "root_pos" in motion:
            motion["root_pos"] = lerp(ts_target, ts_source, motion["root_pos"])
        if "root_rot" in motion:
            motion["root_rot"] = slerp(ts_target, ts_source, motion["root_rot"].reshape(T, 1, 4)).reshape(len(ts_target), 4)
        if "local_body_pos" in motion:
            motion["local_body_pos"] = lerp(ts_target, ts_source, motion["local_body_pos"].reshape(T, -1)).reshape(len(ts_target), -1, 3)

    return motion

def quat_to_angular_velocity(quat: torch.Tensor, fps: float) -> torch.Tensor:
    """Convert quaternion sequence to angular velocities using finite differences.
    
    Args:
        quat: Quaternion sequence of shape [T, ..., 4] where ... represents arbitrary batch dimensions
        fps: Frame rate for computing the time derivative
    
    Returns:
        Angular velocities of shape [T-1, ..., 3]
    """
    dt = 1.0 / fps
    
    # Get q1 and q2 for consecutive timesteps
    q1 = quat[:-1]  # [T-1, ..., 4]
    q2 = quat[1:]   # [T-1, ..., 4]
    
    # Compute angular velocities using the formula
    # ω = 2/dt * [q1w*q2x - q1x*q2w - q1y*q2z + q1z*q2y,
    #             q1w*q2y + q1x*q2z - q1y*q2w - q1z*q2x,
    #             q1w*q2z - q1x*q2y + q1y*q2x - q1z*q2w]
    
    ang_vel = (2.0 / dt) * torch.stack([
        q1[..., 0]*q2[..., 1] - q1[..., 1]*q2[..., 0] - q1[..., 2]*q2[..., 3] + q1[..., 3]*q2[..., 2],
        q1[..., 0]*q2[..., 2] + q1[..., 1]*q2[..., 3] - q1[..., 2]*q2[..., 0] - q1[..., 3]*q2[..., 1],
        q1[..., 0]*q2[..., 3] - q1[..., 1]*q2[..., 2] + q1[..., 2]*q2[..., 1] - q1[..., 3]*q2[..., 0]
    ], dim=-1)
    
    return ang_vel


# @tensorclass
class TwistMotionData(TensorClass):
    motion_id: torch.Tensor
    step: torch.Tensor

    # HDMI 标准字段
    body_pos_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor

    # GMR 额外字段（可选）
    root_pos: torch.Tensor = None      # [N, 3] 根位置
    root_rot: torch.Tensor = None      # [N, 4] 根旋转（四元数 wxyz）
    local_body_pos: torch.Tensor = None  # [N, num_bodies, 3] 局部身体位置

class TwistMotionDataset:
    def __init__(
        self,
        body_names: List[str],
        joint_names: List[str],
        motion_paths: List[Path],
        starts: List[int],
        ends: List[int],
        data: TwistMotionData,
    ):
        self.body_names = body_names
        self.joint_names = joint_names
        self.motion_paths = motion_paths
        self.starts = torch.as_tensor(starts)
        self.ends = torch.as_tensor(ends)
        self.lengths = self.ends - self.starts
        self.data = data
        self.device = data.device
    
    def to(self, device: torch.device):
        self.data = self.data.to(device)
        self.starts = self.starts.to(device)
        self.ends = self.ends.to(device)
        self.lengths = self.lengths.to(device)
        self.device = device
        return self

    @classmethod
    def create_from_path(cls, root_path: str | List[str], isaac_joint_names: List[str] | None = None, target_fps: int = 50, memory_mapped: bool = False):

        
        if isinstance(root_path, ListConfig) or isinstance(root_path, list):
            # 如果是列表,取第一个路径
            root_path = root_path[0] if root_path else "."
        
        if not isinstance(root_path, str):
            raise ValueError(f"Invalid root_path type: {type(root_path)}")
        
        # 递归查找所有.pkl文件
        motion_paths = glob.glob(os.path.join(root_path, "**/*.pkl"), recursive=True)
        motion_paths = [Path(p) for p in motion_paths]
        
        print(f"Found {len(motion_paths)} .pkl files in {root_path}")
        
        if not motion_paths:
            raise RuntimeError(f"No .pkl files found in {root_path}")
        
        # 读取所有motion数据
        motions = []
        total_length = 0
        body_names = None
        joint_names = None
        #dict_keys(['fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list'])
        for motion_path in tqdm(motion_paths, desc="Loading motion files"):
            try:
                with open(motion_path, "rb") as f:
                    motion_data = pickle.load(f)

                # 提取数据
                fps = motion_data["fps"]

                # 检查是 GMR 格式还是 HDMI 格式
                if "root_pos" in motion_data and "dof_pos" in motion_data:
                    # === GMR 格式 ===
                    # 需要从 GMR 格式转换为 HDMI 格式
                    print(f"  [GMR Format] {motion_path.name}")

                    root_pos = np.array(motion_data["root_pos"])  # [T, 3]
                    root_rot = np.array(motion_data["root_rot"])  # [T, 4] wxyz
                    dof_pos = np.array(motion_data["dof_pos"])    # [T, 29]
                    local_body_pos = np.array(motion_data["local_body_pos"])  # [T, N_bodies, 3]

                    T = root_pos.shape[0]
                    N_bodies = local_body_pos.shape[1]

                    # 1. 计算速度（有限差分）
                    dt = 1.0 / fps

                    # Root 线速度
                    root_lin_vel = np.zeros_like(root_pos)
                    root_lin_vel[:-1] = (root_pos[1:] - root_pos[:-1]) / dt
                    root_lin_vel[-1] = root_lin_vel[-2]  # 最后一帧复制前一帧

                    # Root 角速度（使用 TWIST 方式计算）
                    # TWIST 方法：root_ang_vel = fps * quat_to_exp_map(quat_diff(q_t, q_{t+1}))
                    # 然后应用 19 点平滑窗口
                    root_ang_vel = np.zeros_like(root_pos)

                    # 计算四元数差分并转换为 exponential map
                    for i in range(T-1):
                        q0 = root_rot[i]    # wxyz
                        q1 = root_rot[i+1]  # wxyz

                        # 1. quat_diff: dq = q1 * conj(q0)
                        q0_conj = np.array([q0[0], -q0[1], -q0[2], -q0[3]])  # conjugate
                        dq = np.array([
                            q0_conj[0]*q1[0] - q0_conj[1]*q1[1] - q0_conj[2]*q1[2] - q0_conj[3]*q1[3],
                            q0_conj[0]*q1[1] + q0_conj[1]*q1[0] + q0_conj[2]*q1[3] - q0_conj[3]*q1[2],
                            q0_conj[0]*q1[2] - q0_conj[1]*q1[3] + q0_conj[2]*q1[0] + q0_conj[3]*q1[1],
                            q0_conj[0]*q1[3] + q0_conj[1]*q1[2] - q0_conj[2]*q1[1] + q0_conj[3]*q1[0]
                        ])

                        # 2. quat_to_exp_map: exp_map = angle * axis
                        # 提取角度和轴
                        min_theta = 1e-5
                        sin_theta = np.sqrt(1 - dq[0] * dq[0])
                        angle = 2 * np.arccos(np.clip(dq[0], -1, 1))

                        # 归一化角度到 [-π, π]
                        angle = ((angle + np.pi) % (2 * np.pi)) - np.pi

                        if abs(sin_theta) > min_theta:
                            axis = dq[1:] / sin_theta
                        else:
                            axis = np.array([0.0, 0.0, 1.0])  # 默认轴
                            angle = 0.0

                        # 3. 转换为角速度：ω = fps * exp_map
                        exp_map = angle * axis
                        root_ang_vel[i] = fps * exp_map

                    # 最后一帧复制前一帧
                    root_ang_vel[-1] = root_ang_vel[-2]

                    # 4. TWIST 使用 19 点平滑窗口（移动平均）
                    # 使用卷积实现高效平滑
                    box_pts = 19
                    if T >= box_pts:
                        from scipy.ndimage import convolve1d
                        kernel = np.ones(box_pts) / box_pts
                        for axis_idx in range(3):
                            root_ang_vel[:, axis_idx] = convolve1d(
                                root_ang_vel[:, axis_idx],
                                kernel,
                                mode='nearest'  # 边界处理
                            )

                    # Joint 速度
                    joint_vel = np.zeros_like(dof_pos)
                    joint_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) / dt
                    joint_vel[-1] = joint_vel[-2]

                    # 2. 构建世界坐标系的 body 位置和四元数
                    # local_body_pos 是局部坐标系，需要转换到世界坐标系
                    # body_pos_w = root_pos + R(root_rot) @ local_body_pos
                    from scipy.spatial.transform import Rotation as R

                    body_pos_w = np.zeros((T, N_bodies, 3))
                    body_quat_w = np.zeros((T, N_bodies, 4))  # wxyz

                    for t in range(T):
                        # 转换四元数格式：wxyz -> xyzw (scipy)
                        root_quat_xyzw = root_rot[t, [1, 2, 3, 0]]
                        R_root = R.from_quat(root_quat_xyzw)

                        for b in range(N_bodies):
                            # 世界坐标位置
                            body_pos_w[t, b] = root_pos[t] + R_root.apply(local_body_pos[t, b])
                            # 世界坐标旋转（假设 body 与 root 同向）
                            body_quat_w[t, b] = root_rot[t]  # 简化假设

                    # 3. 计算 body 速度
                    body_lin_vel_w = np.zeros_like(body_pos_w)
                    body_lin_vel_w[:-1] = (body_pos_w[1:] - body_pos_w[:-1]) / dt
                    body_lin_vel_w[-1] = body_lin_vel_w[-2]

                    body_ang_vel_w = np.tile(root_ang_vel[:, np.newaxis, :], (1, N_bodies, 1))

                    # 构建 HDMI 格式的 motion
                    motion = {
                        "local_body_pos": local_body_pos,
                        "joint_pos": dof_pos,
                        "joint_vel": joint_vel,
                        "root_pos": root_pos,
                        "root_rot": root_rot,


                        
                        "body_pos_w": body_pos_w,
                        "body_lin_vel_w": body_lin_vel_w,
                        "body_quat_w": body_quat_w,
                        "body_ang_vel_w": body_ang_vel_w,
                        
                    }

                    # 获取 body 和 joint 名称
                    if body_names is None:
                        body_names = motion_data.get("link_body_list", unitree_body_names)
                        joint_names = unitree_joint_names  # GMR 没有 joint_names，使用默认

                elif "body_pos_w" in motion_data:
                    # === HDMI 格式 ===
                    print(f"  [HDMI Format] {motion_path.name}")

                    # 转换为numpy数组
                    motion = {
                        "body_pos_w": np.array(motion_data["body_pos_w"]),
                        "body_lin_vel_w": np.array(motion_data["body_lin_vel_w"]),
                        "body_quat_w": np.array(motion_data["body_quat_w"]),
                        "body_ang_vel_w": np.array(motion_data["body_ang_vel_w"]),
                        "joint_pos": np.array(motion_data["joint_pos"]),
                        "joint_vel": np.array(motion_data["joint_vel"]),
                    }

                    # 获取body和joint名称(假设所有文件相同)
                    if body_names is None:
                        body_names = motion_data.get("body_names", unitree_body_names)
                        joint_names = motion_data.get("joint_names", unitree_joint_names)

                else:
                    raise ValueError(f"Unknown motion format in {motion_path}")

                # 插值到目标帧率
                motion = interpolate(motion, source_fps=fps, target_fps=target_fps)

                total_length += motion["body_pos_w"].shape[0]
                motions.append(motion)
                    
            except Exception as e:
                import traceback
                print(f"Failed to load {motion_path}: {e}")
                print(f"  Full traceback:")
                traceback.print_exc()
                continue
        
        if not motions:
            raise RuntimeError(f"No valid motion data loaded from {root_path}")
        
        # 重新映射关节顺序
        if isaac_joint_names is not None:
            share_joint_names = [name for name in joint_names if name in isaac_joint_names]
            src_joint_indices = [joint_names.index(name) for name in share_joint_names]
            dst_joint_indices = [isaac_joint_names.index(name) for name in share_joint_names]

            more_joint_names = [name for name in joint_names if name not in isaac_joint_names]
            src_more_joint_indices = [joint_names.index(name) for name in more_joint_names]
            dst_more_joint_indices = [len(isaac_joint_names) + i for i in range(len(more_joint_names))]

            joint_names = isaac_joint_names + more_joint_names
            src_joint_indices = src_joint_indices + src_more_joint_indices
            dst_joint_indices = dst_joint_indices + dst_more_joint_indices

            for motion in motions:
                joint_pos = np.zeros((motion["joint_pos"].shape[0], len(joint_names)))
                joint_vel = np.zeros((motion["joint_vel"].shape[0], len(joint_names)))
                joint_pos[:, dst_joint_indices] = motion["joint_pos"][:, src_joint_indices]
                joint_vel[:, dst_joint_indices] = motion["joint_vel"][:, src_joint_indices]
                motion["joint_pos"] = joint_pos
                motion["joint_vel"] = joint_vel
        
        # 创建tensor
        TensorClass = MemoryMappedTensor if memory_mapped else torch

        step: torch.Tensor = TensorClass.empty(total_length, dtype=int)
        motion_id: torch.Tensor = TensorClass.empty(total_length, dtype=int)
        body_pos_w: torch.Tensor = TensorClass.empty(total_length, len(body_names), 3)
        body_lin_vel_w: torch.Tensor = TensorClass.empty(total_length, len(body_names), 3)
        body_quat_w: torch.Tensor = TensorClass.empty(total_length, len(body_names), 4)
        body_ang_vel_w: torch.Tensor = TensorClass.empty(total_length, len(body_names), 3)
        joint_pos: torch.Tensor = TensorClass.empty(total_length, len(joint_names))
        joint_vel: torch.Tensor = TensorClass.empty(total_length, len(joint_names))

        # 检查是否有 GMR 额外字段
        has_gmr_fields = any("root_pos" in m for m in motions)
        root_pos_tensor = None
        root_rot_tensor = None
        local_body_pos_tensor = None

        if has_gmr_fields:
            # 确定 N_bodies（从第一个包含 local_body_pos 的 motion 中获取）
            N_bodies = next((m["local_body_pos"].shape[1] for m in motions if "local_body_pos" in m), len(body_names))

            root_pos_tensor = TensorClass.empty(total_length, 3)
            root_rot_tensor = TensorClass.empty(total_length, 4)
            local_body_pos_tensor = TensorClass.empty(total_length, N_bodies, 3)

        start_idx = 0
        starts = []
        ends = []

        for i, motion in enumerate(motions):
            motion_length = motion["body_pos_w"].shape[0]
            step[start_idx: start_idx + motion_length] = torch.arange(motion_length)
            motion_id[start_idx:start_idx + motion_length] = i

            # Body and joint positions
            body_pos_w[start_idx:start_idx + motion_length] = torch.as_tensor(motion["body_pos_w"])
            body_lin_vel_w[start_idx:start_idx + motion_length] = torch.as_tensor(motion["body_lin_vel_w"])
            body_quat_w[start_idx:start_idx + motion_length] = torch.as_tensor(motion["body_quat_w"])
            body_ang_vel_w[start_idx:start_idx + motion_length] = torch.as_tensor(motion["body_ang_vel_w"])
            joint_pos[start_idx:start_idx + motion_length] = torch.as_tensor(motion["joint_pos"])
            joint_vel[start_idx:start_idx + motion_length] = torch.as_tensor(motion["joint_vel"])

            # 填充 GMR 额外字段（如果存在）
            if has_gmr_fields and "root_pos" in motion:
                root_pos_tensor[start_idx:start_idx + motion_length] = torch.as_tensor(motion["root_pos"])
                root_rot_tensor[start_idx:start_idx + motion_length] = torch.as_tensor(motion["root_rot"])
                if "local_body_pos" in motion:
                    local_body_pos_tensor[start_idx:start_idx + motion_length] = torch.as_tensor(motion["local_body_pos"])

            starts.append(start_idx)
            start_idx += motion_length
            ends.append(start_idx)
        
        kwargs = {
            "motion_id": motion_id,
            "step": step,
            "body_pos_w": body_pos_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_quat_w": body_quat_w,
            "body_ang_vel_w": body_ang_vel_w,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "root_pos": root_pos_tensor,
            "root_rot": root_rot_tensor,
            "local_body_pos": local_body_pos_tensor,
            "batch_size": [total_length]
        }

        data = TwistMotionData(**kwargs)

        return cls(
            body_names=body_names,
            joint_names=joint_names,
            motion_paths=motion_paths,
            starts=starts,
            ends=ends,
            data=data,
        )

    @property
    def num_motions(self):
        return len(self.starts)
    
    @property
    def num_steps(self):
        return len(self.data)

    def get_slice(self, motion_ids: torch.Tensor, starts: torch.Tensor, steps: Union[int, torch.Tensor] = 1) -> TwistMotionData:
        if isinstance(steps, int):
            steps = torch.arange(steps, device=self.device)
        idx = (self.starts[motion_ids] + starts).unsqueeze(1) + steps.unsqueeze(0)
        idx.clamp_max_(self.ends.unsqueeze(1)[motion_ids] - 1)
        return self.data[idx] # shape: [len(motion_ids), len(steps), ...]

    def find_joints(self, joint_names, preserve_order: bool=False):
        return resolve_matching_names(joint_names, self.joint_names, preserve_order)

    def find_bodies(self, body_names, preserve_order: bool=False):
        return resolve_matching_names(body_names, self.body_names, preserve_order)
