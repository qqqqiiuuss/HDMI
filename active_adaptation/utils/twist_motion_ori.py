

import os
import yaml
import pickle
import logging
from pathlib import Path
from typing import List, Union, Tuple
from tqdm import tqdm

import torch
import numpy as np
from tensordict import TensorClass

logger = logging.getLogger(__name__)


def smooth(x: torch.Tensor, box_pts: int, device: torch.device) -> torch.Tensor:
    """
    使用 box filter 进行 1D 卷积平滑

    Args:
        x: 输入张量 (T, C)
        box_pts: 卷积核大小
        device: 设备

    Returns:
        平滑后的张量 (T, C)
    """
    box = torch.ones(box_pts, device=device) / box_pts
    num_channels = x.shape[1]
    x_reshaped = x.T.unsqueeze(0)  # (1, C, T)
    smoothed = torch.nn.functional.conv1d(
        x_reshaped,
        box.view(1, 1, -1).expand(num_channels, 1, -1),
        groups=num_channels,
        padding='same'
    )
    return smoothed.squeeze(0).T  # (T, C)


def quat_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    计算两个四元数之间的差分

    Args:
        q1: 第一个四元数 (*, 4) [w, x, y, z]
        q2: 第二个四元数 (*, 4) [w, x, y, z]

    Returns:
        差分四元数 (*, 4)
    """
    # q_diff = q1^(-1) * q2
    q1_conj = q1.clone()
    q1_conj[..., 1:] *= -1  # 共轭

    # 四元数乘法
    w1, x1, y1, z1 = q1_conj[..., 0], q1_conj[..., 1], q1_conj[..., 2], q1_conj[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quat_to_exp_map(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为指数映射（exponential map）

    Args:
        q: 四元数 (*, 4) [w, x, y, z]

    Returns:
        指数映射 (*, 3)
    """
    angle = 2 * torch.acos(torch.clamp(q[..., 0], -1.0, 1.0))
    axis = q[..., 1:] / (torch.sin(angle.unsqueeze(-1) / 2) + 1e-8)
    exp_map = axis * angle.unsqueeze(-1)
    return exp_map


def slerp(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    四元数球面线性插值（Spherical Linear Interpolation）

    Args:
        q1: 起始四元数 (N, 4) [w, x, y, z]
        q2: 结束四元数 (N, 4) [w, x, y, z]
        t: 插值权重 (N,) 或 (N, 1)

    Returns:
        插值后的四元数 (N, 4)
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)

    # 计算点积
    dot = (q1 * q2).sum(dim=-1, keepdim=True)

    # 如果点积为负，翻转一个四元数以选择最短路径
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)

    # 如果四元数非常接近，使用线性插值
    theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta = torch.sin(theta)

    # SLERP 公式
    w1 = torch.sin((1 - t) * theta) / (sin_theta + 1e-8)
    w2 = torch.sin(t * theta) / (sin_theta + 1e-8)

    # 处理接近的情况（使用线性插值）
    close_mask = (sin_theta < 1e-6)
    w1 = torch.where(close_mask, 1 - t, w1)
    w2 = torch.where(close_mask, t, w2)

    return w1 * q1 + w2 * q2


class TwistMotionData(TensorClass):
    """
    TWIST 风格的 Motion 数据类

    与 HDMI 的 MotionData 兼容，但使用 TWIST 的数据格式
    """
    motion_id: torch.Tensor      # [N] motion ID
    step: torch.Tensor           # [N] 当前帧索引

    # Root state (TWIST 格式)
    root_pos: torch.Tensor       # [N, 3] 根位置
    root_rot: torch.Tensor       # [N, 4] 根旋转（四元数 wxyz）
    root_vel: torch.Tensor       # [N, 3] 根线速度（TWIST 计算）
    root_ang_vel: torch.Tensor   # [N, 3] 根角速度（TWIST 计算）

    # Joint state
    dof_pos: torch.Tensor        # [N, num_dof] DOF 位置
    dof_vel: torch.Tensor        # [N, num_dof] DOF 速度（TWIST 计算）

    # Key body state (TWIST 格式 - 局部坐标系)
    local_key_body_pos: torch.Tensor  # [N, num_key_bodies, 3] 关键点局部位置


class TwistMotionDataset:
    """
    TWIST Motion Dataset for HDMI

    这个类实现了与 HDMI MotionDataset 兼容的接口，
    但内部使用 TWIST 的 motion 处理逻辑。

    主要特性：
    1. 保留 TWIST 的 19 点平滑滤波
    2. 保留 TWIST 的速度计算方法（有限差分 + 平滑）
    3. 保留 TWIST 的数据拼接方式（单一大张量）
    4. 提供与 HDMI 兼容的 get_slice 接口

    Args:
        body_names: 身体链接名称列表（从 pkl 中读取）
        joint_names: 关节名称列表（从 pkl 中读取）
        motion_paths: Motion 文件路径列表
        starts: 每个 motion 的起始帧索引
        ends: 每个 motion 的结束帧索引
        lengths: 每个 motion 的长度（秒）
        data: TwistMotionData 对象
    """

    def __init__(
        self,
        body_names: List[str],
        joint_names: List[str],
        motion_paths: List[str],
        starts: torch.Tensor,
        ends: torch.Tensor,
        lengths: torch.Tensor,
        data: TwistMotionData,
    ):
        self.body_names = body_names
        self.joint_names = joint_names
        self.motion_paths = motion_paths
        self.starts = starts
        self.ends = ends
        self.lengths = lengths
        self.data = data
        self.device = data.dof_pos.device

    def to(self, device: torch.device):
        """移动数据到指定设备"""
        self.data = self.data.to(device)
        self.starts = self.starts.to(device)
        self.ends = self.ends.to(device)
        self.lengths = self.lengths.to(device)
        self.device = device
        return self

    @property
    def num_motions(self) -> int:
        """返回 motion 数量"""
        return len(self.starts)

    @property
    def num_steps(self) -> int:
        """返回总帧数"""
        return len(self.data)

    @classmethod
    def create_from_yaml(
        cls,
        yaml_path: str,
        device: torch.device = torch.device("cuda:0"),
        smooth_window: int = 19,
    ):
        """
        从 YAML 配置文件加载 TWIST motion data

        Args:
            yaml_path: YAML 配置文件路径
            device: 目标设备
            smooth_window: 平滑窗口大小（TWIST 默认 19）

        Returns:
            TwistMotionDataset 实例

        YAML 格式示例:
        ```yaml
        root_path: "/path/to/motions"
        motions:
          - file: "walk.pkl"
          - file: "run.pkl"
        ```
        注意：weight 字段会被忽略，所有 motion 平等对待（均匀采样）
        """
        logger.info(f"Loading TWIST motions from {yaml_path}")

        # 读取 YAML 配置
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        root_path = config["root_path"]
        motion_list = config["motions"]

        # 准备数据容器
        motion_paths = []

        motion_root_pos_list = []
        motion_root_rot_list = []
        motion_root_vel_list = []
        motion_root_ang_vel_list = []
        motion_dof_pos_list = []
        motion_dof_vel_list = []
        motion_local_body_pos_list = []

        motion_num_frames = []
        motion_lengths = []

        body_names = None
        joint_names = None

        # 加载每个 motion
        for motion_entry in tqdm(motion_list, desc="Loading TWIST motions"):
            motion_file = os.path.join(root_path, motion_entry["file"])
            # 注意：忽略 weight 字段，所有 motion 平等对待

            try:
                with open(motion_file, "rb") as f:
                    motion_data = pickle.load(f)

                fps = motion_data["fps"]
                dt = 1.0 / fps

                # 读取原始数据
                root_pos = torch.tensor(motion_data["root_pos"], dtype=torch.float32, device=device)
                root_rot = torch.tensor(motion_data["root_rot"], dtype=torch.float32, device=device)
                dof_pos = torch.tensor(motion_data["dof_pos"], dtype=torch.float32, device=device)
                local_body_pos = torch.tensor(motion_data["local_body_pos"], dtype=torch.float32, device=device)

                if body_names is None:
                    body_names = motion_data.get("link_body_list", [])
                    joint_names = motion_data.get("joint_names", [])

                num_frames = root_pos.shape[0]
                curr_len = dt * (num_frames - 1)

                # === TWIST 特有处理：计算速度并平滑 ===

                # 1. 计算根线速度
                root_vel = torch.zeros_like(root_pos)
                root_vel[:-1, :] = fps * (root_pos[1:, :] - root_pos[:-1, :])
                root_vel[-1, :] = root_vel[-2, :]
                root_vel = smooth(root_vel, smooth_window, device=device)

                # 2. 计算根角速度
                root_ang_vel = torch.zeros_like(root_pos)
                root_drot = quat_diff(root_rot[:-1], root_rot[1:])
                root_ang_vel[:-1, :] = fps * quat_to_exp_map(root_drot)
                root_ang_vel[-1, :] = root_ang_vel[-2, :]
                root_ang_vel = smooth(root_ang_vel, smooth_window, device=device)

                # 3. 计算 DOF 速度
                dof_vel = torch.zeros_like(dof_pos)
                dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
                dof_vel[-1, :] = dof_vel[-2, :]
                dof_vel = smooth(dof_vel, smooth_window, device=device)

                # 存储数据
                motion_root_pos_list.append(root_pos)
                motion_root_rot_list.append(root_rot)
                motion_root_vel_list.append(root_vel)
                motion_root_ang_vel_list.append(root_ang_vel)
                motion_dof_pos_list.append(dof_pos)
                motion_dof_vel_list.append(dof_vel)
                motion_local_body_pos_list.append(local_body_pos)

                motion_num_frames.append(num_frames)
                motion_lengths.append(curr_len)
                motion_paths.append(motion_file)

            except Exception as e:
                logger.error(f"Failed to load {motion_file}: {e}")
                continue

        if not motion_paths:
            raise RuntimeError(f"No motions loaded from {yaml_path}")

        # 转换为张量
        num_frames_tensor = torch.tensor(motion_num_frames, dtype=torch.long, device=device)
        lengths_tensor = torch.tensor(motion_lengths, dtype=torch.float32, device=device)

        # === TWIST 特有处理：拼接所有 motion 到单一张量 ===
        root_pos_all = torch.cat(motion_root_pos_list, dim=0)
        root_rot_all = torch.cat(motion_root_rot_list, dim=0)
        root_vel_all = torch.cat(motion_root_vel_list, dim=0)
        root_ang_vel_all = torch.cat(motion_root_ang_vel_list, dim=0)
        dof_pos_all = torch.cat(motion_dof_pos_list, dim=0)
        dof_vel_all = torch.cat(motion_dof_vel_list, dim=0)
        local_body_pos_all = torch.cat(motion_local_body_pos_list, dim=0)

        total_frames = root_pos_all.shape[0]

        # 计算每个 motion 的起始索引
        lengths_shifted = num_frames_tensor.roll(1)
        lengths_shifted[0] = 0
        starts = lengths_shifted.cumsum(0)
        ends = starts + num_frames_tensor

        # 创建 motion_id 和 step 张量
        motion_id = torch.zeros(total_frames, dtype=torch.long, device=device)
        step = torch.zeros(total_frames, dtype=torch.long, device=device)

        for i in range(len(motion_paths)):
            motion_id[starts[i]:ends[i]] = i
            step[starts[i]:ends[i]] = torch.arange(num_frames_tensor[i], device=device)

        # 创建 TwistMotionData
        data = TwistMotionData(
            motion_id=motion_id,
            step=step,
            root_pos=root_pos_all,
            root_rot=root_rot_all,
            root_vel=root_vel_all,
            root_ang_vel=root_ang_vel_all,
            dof_pos=dof_pos_all,
            dof_vel=dof_vel_all,
            local_key_body_pos=local_body_pos_all,
            batch_size=[total_frames]
        )

        logger.info(f"Loaded {len(motion_paths)} motions with {total_frames} total frames")
        logger.info(f"Using uniform sampling (all motions treated equally)")

        return cls(
            body_names=body_names,
            joint_names=joint_names,
            motion_paths=motion_paths,
            starts=starts,
            ends=ends,
            lengths=lengths_tensor,
            data=data,
        )

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: Union[int, torch.Tensor] = 1
    ) -> TwistMotionData:
        """
        获取指定 motion 的切片数据（与 HDMI 兼容的接口）

        Args:
            motion_ids: Motion ID 张量 (N,)
            starts: 每个 motion 的起始帧 (N,)
            steps: 帧数或帧索引张量

        Returns:
            TwistMotionData: 形状为 [N, len(steps), ...] 的数据
        """
        if isinstance(steps, int):
            steps = torch.arange(steps, device=self.device)

        # 计算全局索引
        idx = (self.starts[motion_ids].unsqueeze(1) + starts.unsqueeze(1)) + steps.unsqueeze(0)

        # 限制在有效范围内
        # 注意：min 和 max 都必须是 Tensor 类型（不能混用 int 和 Tensor）
        min_idx = torch.zeros_like(idx)
        max_idx = self.ends[motion_ids].unsqueeze(1) - 1
        idx = torch.clamp(idx, min_idx, max_idx)

        return self.data[idx]  # [len(motion_ids), len(steps), ...]

    

    def find_joints(self, joint_names: List[str], preserve_order: bool = False) -> Tuple[List[int], List[str]]:
        """查找关节索引（与 HDMI 兼容）"""
        indices = []
        names = []
        for name in joint_names:
            if name in self.joint_names:
                indices.append(self.joint_names.index(name))
                names.append(name)
        return indices, names

    def find_bodies(self, body_names: List[str], preserve_order: bool = False) -> Tuple[List[int], List[str]]:
        """查找身体索引（与 HDMI 兼容）"""
        indices = []
        names = []
        for name in body_names:
            if name in self.body_names:
                indices.append(self.body_names.index(name))
                names.append(name)
        return indices, names

    def sample_motions(self, n: int) -> torch.Tensor:
        """
        采样 motion IDs（均匀随机采样，每个 motion 平等对待）

        Args:
            n: 采样数量

        Returns:
            Motion IDs (n,)
        """
        return torch.randint(0, self.num_motions, (n,), device=self.device)

    def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """
        为指定 motion 采样随机时间

        Args:
            motion_ids: Motion IDs (N,)

        Returns:
            时间（秒） (N,)
        """
        phase = torch.rand(motion_ids.shape, device=self.device)
        return self.lengths[motion_ids] * phase
