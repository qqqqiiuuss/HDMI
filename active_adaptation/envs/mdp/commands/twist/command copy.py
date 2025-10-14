"""
TWIST Motion Tracking Command Module

这个模块实现了 TWIST 风格的运动跟踪命令管理器。

主要特性:
1. 使用 TwistMotionDataset 加载 PKL 格式的运动数据
2. 支持多运动采样（每个环境独立采样）
3. 提供 20 步未来参考运动
4. 支持 9 个关键点跟踪

作者: HDMI-TWIST Integration
日期: 2025.10.11
"""

from active_adaptation.envs.mdp.base import Command
from active_adaptation.utils.twist_motion import TwistMotionDataset, TwistMotionData

from typing import List, Dict, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.assets import Articulation

import torch
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz, quat_mul

torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)


class TwistMotionTracking(Command):
    """
    TWIST 运动跟踪命令类

    该类负责从 TWIST motion dataset 中加载参考运动轨迹，并为机器人提供跟踪目标。
    与 HDMI 的 RobotTracking 不同，它使用 TwistMotionDataset 并保留 TWIST 的处理流程。

    Args:
        env: 环境实例
        data_path: TWIST motion YAML 配置文件路径
        key_body_names: 关键点名称列表（TWIST 默认 9 个）
        root_body_name: 根身体名称
        pose_range: 位置和姿态的随机化范围
        velocity_range: 速度的随机化范围
        init_joint_pos_noise: 初始关节位置噪声
        init_joint_vel_noise: 初始关节速度噪声
        future_steps: 未来时间步列表（TWIST 使用 20 步）
    """

    def __init__(
        self,
        env,
        data_path: str,  # YAML 配置文件路径
        key_body_names: List[str] = None,
        root_body_name: str = "pelvis",
        pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2)
        },
        velocity_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (-0.2, 0.2),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5)
        },
        init_joint_pos_noise: float = 0.1,
        init_joint_vel_noise: float = 0.1,
        future_steps: List[int] = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        use_runtime_interpolation: bool = False,  # 新增：是否使用运行时插值
        target_fps: float = 50.0,  # 新增：目标FPS（用于运行时插值）
    ):
        super().__init__(env)

        # 创建 TWIST motion dataset
        print(f"[TwistMotionTracking] Loading TWIST motions from: {data_path}")
        self.dataset = TwistMotionDataset.create_from_yaml(
            yaml_path=data_path,
            device=self.device,
            smooth_window=19  # TWIST 使用 19 点平滑
        )

        # TWIST 默认 9 个关键点（G1 机器人）
        if key_body_names is None:
            key_body_names = [
                "left_hand_marker",
                "right_hand_marker",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_knee_link",
                "right_knee_link",
                "left_elbow_link",
                "right_elbow_link",
                "head_link"
            ]

        # 查找关键点索引
        self.key_body_indices_robot = []
        self.key_body_names_matched = []

        for pattern in key_body_names:
            indices, names = self.asset.find_bodies(pattern)
            if indices:
                self.key_body_indices_robot.append(indices[0])
                self.key_body_names_matched.append(names[0])

        self.num_key_bodies = len(self.key_body_indices_robot)

        print(f"[TwistMotionTracking] Tracking {self.num_key_bodies} key bodies:")
        for name in self.key_body_names_matched:
            print(f"  - {name}")

        # TWIST motion 数据中的关键点已经是预先计算好的 local_key_body_pos
        # 不需要从 dataset.body_names 中查找索引
        # 我们假设 motion 数据中的 local_key_body_pos 已经是 9 个关键点的顺序
        print(f"[TwistMotionTracking] Using pre-computed local_key_body_pos from motion data")

        # 根身体名称（用于重置）
        self.root_body_name = root_body_name

        # 机器人关节映射：尝试匹配，如果不匹配则使用默认顺序
        asset_joint_names = self.asset.joint_names
        self.asset_joint_idx_motion = []

        print(f"[TwistMotionTracking] Mapping {len(asset_joint_names)} robot joints to dataset...")
        print(f"  Robot joints: {asset_joint_names[:5]}...")

        if hasattr(self.dataset, 'joint_names') and self.dataset.joint_names:
            print(f"  Dataset joints ({len(self.dataset.joint_names)}): {self.dataset.joint_names[:5]}...")

            # 尝试匹配关节名称
            matched_count = 0
            for i, joint_name in enumerate(asset_joint_names):
                try:
                    idx = self.dataset.joint_names.index(joint_name)
                    self.asset_joint_idx_motion.append(idx)
                    matched_count += 1
                except ValueError:
                    # 如果找不到，使用 -1 标记（后续处理时跳过）
                    self.asset_joint_idx_motion.append(-1)
                    print(f"  [Warning] Joint '{joint_name}' not found in dataset, will use default value")

            print(f"  ✓ Matched {matched_count}/{len(asset_joint_names)} joints")
            print(f"  ✗ {len(asset_joint_names) - matched_count} joints will use default values")
        else:
            # 没有关节名称信息，假设前 N 个顺序一致
            print(f"  [Warning] No joint names in dataset, assuming first {len(asset_joint_names)} are sequential")
            self.asset_joint_idx_motion = list(range(len(asset_joint_names)))

        # 初始化张量
        with torch.device(self.device):
            # 未来时间步
            self.future_steps = torch.tensor(future_steps)
            self.num_future_steps = len(future_steps)

            # 运动状态
            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_len = torch.zeros(self.num_envs, dtype=torch.long)  # 运动长度（帧数）
            self.motion_starts = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_ends = torch.zeros(self.num_envs, dtype=torch.long)
            self.t = torch.zeros(self.num_envs, dtype=torch.long)  # 帧索引（离散模式）或步数（插值模式）
            self.motion_time = torch.zeros(self.num_envs, dtype=torch.float32)  # 运行时插值时使用

        # 随机化参数
        pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.pose_range = torch.tensor(pose_range_list, device=self.device)

        velocity_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.velocity_range = torch.tensor(velocity_range_list, device=self.device)

        self.init_joint_pos_noise = init_joint_pos_noise
        self.init_joint_vel_noise = init_joint_vel_noise

        # 运行时插值配置
        self.use_runtime_interpolation = use_runtime_interpolation
        self.target_fps = target_fps
        self.dt = 1.0 / target_fps  # 时间步长（秒）

        # 初始化
        self.first_sample = True
        self.update()

        print(f"[TwistMotionTracking] Initialized successfully!")
        print(f"  - Num motions: {self.dataset.num_motions}")
        print(f"  - Total frames: {self.dataset.num_steps}")
        print(f"  - Future steps: {len(future_steps)}")
        print(f"  - Runtime interpolation: {'Enabled' if use_runtime_interpolation else 'Disabled'}")
        if use_runtime_interpolation:
            print(f"  - Target FPS: {target_fps}")

    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        """
        为指定环境采样运动数据

        Args:
            env_ids: 需要采样的环境ID列表
        """
        if self.first_sample:
            # 使用均匀采样（所有 motion 平等对待）
            motion_ids = self.dataset.sample_motions(len(env_ids))
            self.motion_ids[env_ids] = motion_ids
            # 将运动长度从帧数获取（ends - starts）
            motion_len = self.dataset.ends[motion_ids] - self.dataset.starts[motion_ids]
            self.motion_len[env_ids] = motion_len
            self.motion_starts[env_ids] = self.dataset.starts[motion_ids]
            self.motion_ends[env_ids] = self.dataset.ends[motion_ids]
            self.first_sample = False
        else:
            motion_len = self.motion_len[env_ids]

        # 随机选择开始时间
        max_len = motion_len - self.future_steps[-1]
        start_phase = torch.rand(len(env_ids), device=self.device)
        start_t = (start_phase * max_len).long()

        # 非训练模式：从开始位置开始
        if not self.env.training:
            start_t.fill_(0)

        self.t[env_ids] = start_t

        # 如果使用运行时插值，初始化时间
        if self.use_runtime_interpolation:
            # 从帧索引转换为时间（秒）
            self.motion_time[env_ids] = start_t.float() * self.dt

    def sample_init(self, env_ids: torch.Tensor) -> None:
        """
        采样并初始化指定环境的机器人状态

        Args:
            env_ids: 需要初始化的环境ID列表
        """
        self._sample_motions(env_ids)

        # 从运动数据中获取重置状态
        self._motion_reset: TwistMotionData = self.dataset.get_slice(
            self.motion_ids[env_ids],
            self.t[env_ids],
            1
        ).squeeze(1)

        # 提取根身体状态
        init_root_pos = self._motion_reset.root_pos
        init_root_quat = self._motion_reset.root_rot
        init_root_lin_vel = self._motion_reset.root_vel
        # TWIST 没有直接的 root_ang_vel，使用零值
        init_root_ang_vel = torch.zeros_like(init_root_lin_vel)

        # 位置和姿态随机化
        rand_samples = sample_uniform(
            self.pose_range[:, 0],
            self.pose_range[:, 1],
            (len(env_ids), 6),
            device=self.device
        )
        if not self.env.training:
            rand_samples.fill_(0.0)

        positions = init_root_pos + self.env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = quat_mul(init_root_quat, orientations_delta)

        # 速度随机化
        rand_samples = sample_uniform(
            self.velocity_range[:, 0],
            self.velocity_range[:, 1],
            (len(env_ids), 6),
            device=self.device
        )
        if not self.env.training:
            rand_samples.fill_(0.0)

        velocities = torch.cat([init_root_lin_vel, init_root_ang_vel], dim=-1) + rand_samples

        # 写入仿真
        self.asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)

        # 初始化关节状态
        # 方案：只映射 motion 数据中存在的关节，其他使用默认值
        num_asset_joints = len(self.asset.joint_names)
        init_joint_pos = self.asset.data.default_joint_pos[env_ids].clone()  # 从默认值开始
        init_joint_vel = torch.zeros(len(env_ids), num_asset_joints, device=self.device)

        # 只更新 motion 数据中存在的关节
        for robot_idx, motion_idx in enumerate(self.asset_joint_idx_motion):
            if motion_idx >= 0 and motion_idx < self._motion_reset.dof_pos.shape[1]:
                # 有效映射：从 motion 数据获取值
                init_joint_pos[:, robot_idx] = self._motion_reset.dof_pos[:, motion_idx]
                init_joint_vel[:, robot_idx] = self._motion_reset.dof_vel[:, motion_idx]
            # 否则保持默认值（已经在上面设置）

        # 添加关节噪声
        joint_pos_noise = sample_uniform(-1, 1, init_joint_pos.shape, device=self.device) * self.init_joint_pos_noise
        joint_vel_noise = sample_uniform(-1, 1, init_joint_vel.shape, device=self.device) * self.init_joint_vel_noise

        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        # 限制在有效范围内
        joint_pos_limits = self.asset.data.soft_joint_pos_limits[env_ids]  # [len(env_ids), num_joints, 2]
        joint_vel_limits = self.asset.data.soft_joint_vel_limits[env_ids]  # [len(env_ids), num_joints]

        # 使用 torch.clamp 而不是 clamp_，因为形状不完全匹配
        init_joint_pos = torch.clamp(init_joint_pos, joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        init_joint_vel = torch.clamp(init_joint_vel, -joint_vel_limits, joint_vel_limits)

        # 写入仿真
        self.asset.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids)

    @property
    def finished(self):
        """检查运动是否完全结束"""
        return (self.t >= self.motion_len).unsqueeze(1)

    def update(self):
        """
        更新运动跟踪状态

        每步调用此方法以：
        1. 获取未来参考运动数据
        2. 更新机器人和参考状态
        3. 推进时间步
        """
        if self.use_runtime_interpolation:
            # === 运行时插值模式 ===
            # 使用 calc_motion_frame 获取当前插值帧
            result = self.dataset.calc_motion_frame(
                self.motion_ids,
                self.motion_time
            )
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, local_key_body_pos = result

            # 创建当前参考运动
            self.current_ref_motion = TwistMotionData(
                motion_id=self.motion_ids,
                step=self.t,  # 步数（不是帧索引）
                root_pos=root_pos,
                root_rot=root_rot,
                root_vel=root_vel,
                root_ang_vel=root_ang_vel,
                dof_pos=dof_pos,
                dof_vel=dof_vel,
                local_key_body_pos=local_key_body_pos,
                batch_size=[self.num_envs]
            )

            # 获取未来参考运动（仍用离散索引，因为未来帧插值增益不大）
            self.future_ref_motion = self.dataset.get_slice(
                self.motion_ids,
                self.t,
                steps=self.future_steps
            )

            # 推进时间
            self.motion_time += self.dt
            self.t += 1
        else:
            # === 离散帧模式（原始实现） ===
            # 获取未来参考运动数据
            self.future_ref_motion = self.dataset.get_slice(
                self.motion_ids,
                self.t,
                steps=self.future_steps
            )
            # 形状: [num_envs, len(future_steps), ...]

            # 获取当前参考运动（第一步）
            self.current_ref_motion: TwistMotionData = TwistMotionData(
                motion_id=self.future_ref_motion.motion_id[:, 0],
                step=self.future_ref_motion.step[:, 0],
                root_pos=self.future_ref_motion.root_pos[:, 0],
                root_rot=self.future_ref_motion.root_rot[:, 0],
                root_vel=self.future_ref_motion.root_vel[:, 0],
                root_ang_vel=self.future_ref_motion.root_ang_vel[:, 0],
                dof_pos=self.future_ref_motion.dof_pos[:, 0],
                dof_vel=self.future_ref_motion.dof_vel[:, 0],
                local_key_body_pos=self.future_ref_motion.local_key_body_pos[:, 0]
            )

            # 推进时间步
            self.t += 1

    def get_current_ref_motion(self) -> TwistMotionData:
        """获取当前参考运动（供观察和奖励使用）"""
        return self.current_ref_motion

    def debug_draw(self):
        """绘制调试信息（暂时为空）"""
        pass
