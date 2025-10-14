"""
机器人运动跟踪命令模块

该模块实现了机器人运动跟踪功能，包括：
1. RobotTracking: 基础机器人运动跟踪类
2. RobotObjectTracking: 带物体交互的机器人运动跟踪类

主要功能：
- 从运动数据集中加载参考运动轨迹
- 为机器人提供运动跟踪目标
- 支持物体交互和接触检测
- 提供可视化和调试功能
"""

from active_adaptation.envs.mdp.base import Command
from active_adaptation.utils.twist_motion import TwistMotionDataset, TwistMotionData

from typing import List, Dict, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from isaaclab.assets import Articulation, RigidObject

import torch
import numpy as np
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz, quat_mul, quat_apply, quat_apply_inverse
from tensordict import TensorDict
from active_adaptation.utils.math import batchify

# 对四元数操作函数进行批处理优化
quat_apply = batchify(quat_apply)
quat_apply_inverse = batchify(quat_apply_inverse)

# 设置PyTorch打印选项，便于调试
torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)

class TwistMotionTracking(Command):
    """
    机器人运动跟踪命令类
    
    该类负责从运动数据集中加载参考运动轨迹，并为机器人提供跟踪目标。
    支持多种模式：采样运动、重放运动、记录运动等。
    
    Args:
        env: 环境实例
        data_path: 运动数据文件路径，可以是单个文件或文件列表
        tracking_keypoint_names: 需要跟踪的关键点名称列表
        tracking_joint_names: 需要跟踪的关节名称列表
        root_body_name: 根身体名称，默认为"pelvis"
        reset_range: 重置时间范围，None表示使用整个运动长度
        pose_range: 位置和姿态的随机化范围
        velocity_range: 速度的随机化范围
        init_joint_pos_noise: 初始关节位置噪声
        init_joint_vel_noise: 初始关节速度噪声
        future_steps: 未来时间步列表，用于观察
        call_update: 是否在初始化时调用update方法
        sample_motion: 是否采样运动
        replay_motion: 是否重放运动
        record_motion: 是否记录运动
    """
    def __init__(
        self, env, data_path: List[str] | str,
        tracking_keypoint_names: List[str] = None,
        key_body_names: List[str] = None,
        tracking_joint_names: List[str] = None,
        # reset parameters
        root_body_name: str = "pelvis",
        reset_range: Tuple[float, float] | None = None,
        pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        velocity_range: Dict[str, Tuple[float, float]] = {
            "x": (-0., 0.),
            "y": (-0., 0.),
            "z": (-0., 0.),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        init_joint_pos_noise: float = 0.0,
        init_joint_vel_noise: float = 0.0,
        # observation parameters
        future_steps: List[int] = [1, 2, 8, 16],
        call_update: bool = True,
        sample_motion: bool = False,
        replay_motion: bool = False,
        record_motion: bool = False,
    ):
        # 导入相关模块（确保模块被加载，即使未直接使用）
        from . import observations  # noqa: F401
        from . import rewards  # noqa: F401
        from . import randomizations  # noqa: F401
        from . import terminations  # noqa: F401
        super().__init__(env)

        # 处理参数兼容性：key_body_names 和 tracking_keypoint_names
        if key_body_names is not None:
            tracking_keypoint_names = key_body_names
        if tracking_keypoint_names is None:
            raise ValueError("Must provide either 'tracking_keypoint_names' or 'key_body_names'")

        # 处理 tracking_joint_names 默认值
        if tracking_joint_names is None:
            tracking_joint_names = self.asset.joint_names

        # 创建运动数据集，将数据加载到指定设备
        self.dataset = TwistMotionDataset.create_from_path(
            data_path,
            isaac_joint_names=self.asset.joint_names,
            target_fps=int(1/self.env.step_dt)
        ).to(self.device)

        # 设置跟踪身体和关节名称，用于观察和终止条件
        self.tracking_keypoint_names = self.asset.find_bodies(tracking_keypoint_names)[1]
        self.tracking_body_indices_motion = [self.dataset.body_names.index(name) for name in self.tracking_keypoint_names]
        self.tracking_body_indices_asset = [self.asset.body_names.index(name) for name in self.tracking_keypoint_names]

        self.tracking_joint_names = self.asset.find_joints(tracking_joint_names)[1]
        self.tracking_joint_indices_motion = [self.dataset.joint_names.index(name) for name in self.tracking_joint_names]
        self.tracking_joint_indices_asset = [self.asset.joint_names.index(name) for name in self.tracking_joint_names]

        # 记录跟踪身体和关节的数量
        self.num_tracking_bodies = len(self.tracking_body_indices_asset)
        self.num_tracking_joints = len(self.tracking_joint_indices_asset)
        self.num_future_steps = len(future_steps)

        # 获取根身体和关节在运动数据中的索引，用于重置
        self.root_body_name = root_body_name
        self.root_body_idx_motion = self.dataset.body_names.index(root_body_name)
        
        # 获取资产关节名称在运动数据中的索引映射
        asset_joint_names = self.asset.joint_names
        self.asset_joint_idx_motion = [self.dataset.joint_names.index(joint_name) for joint_name in asset_joint_names]

        # 在指定设备上初始化张量
        with torch.device(self.device):
            # 环境状态标记
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            # 未来时间步张量
            self.future_steps = torch.tensor(future_steps)

            # 运动相关状态变量
            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long)  # 当前运动ID
            self.motion_len = torch.zeros(self.num_envs, dtype=torch.long)  # 运动长度
            self.motion_starts = torch.zeros(self.num_envs, dtype=torch.long)  # 运动开始时间
            self.motion_ends = torch.zeros(self.num_envs, dtype=torch.long)  # 运动结束时间
            self.t = torch.zeros(self.num_envs, dtype=torch.long)  # 当前时间步
            self.replay_motion_t = torch.zeros(self.num_envs, dtype=torch.long)  # 重放运动时间

            # 评估时间步（随机选择）
            self.eval_t = torch.randint(0, self.dataset.lengths[0], (self.num_envs,), device=self.device)

        # 重置参数
        self.reset_range = reset_range

        # 位置和姿态随机化范围
        pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.pose_range = torch.tensor(pose_range_list, device=self.device)
        # 速度随机化范围
        velocity_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.velocity_range = torch.tensor(velocity_range_list, device=self.device)

        # 关节噪声参数
        self.init_joint_pos_noise = init_joint_pos_noise
        self.init_joint_vel_noise = init_joint_vel_noise

        # 运动模式标志
        self.first_sample_motion = True
        self.sample_motion = sample_motion
        self.replay_motion = replay_motion
        self.record_motion = record_motion

        # 重放运动模式：禁用随机化
        if self.replay_motion:
            self.pose_range.fill_(0.0)
            self.init_joint_pos_noise = 0.0
            self.init_joint_vel_noise = 0.0
        
        # 记录运动模式：只支持单环境，禁用随机化
        if self.record_motion:
            assert self.num_envs == 1, "record_motion only supports num_envs=1"
            self.pose_range.fill_(0.0)
            self.init_joint_pos_noise = 0.0
            self.init_joint_vel_noise = 0.0

        # 初始化调试绘制和更新
        if call_update:
            self._init_debug_draw()
            self.update()
            if self.record_motion:
                self.motion_frames = []
        
    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        """
        为指定环境采样运动数据
        
        Args:
            env_ids: 需要采样的环境ID列表
        """
        if self.sample_motion or self.first_sample_motion:
            # 为每个环境采样运动ID和开始时间
            motion_ids = torch.randint(0, self.dataset.num_motions, size=(len(env_ids),), device=self.device)
            self.motion_ids[env_ids] = motion_ids
            self.motion_len[env_ids] = motion_len = self.dataset.lengths[motion_ids]
            self.motion_starts[env_ids] = self.dataset.starts[motion_ids]
            self.motion_ends[env_ids] = self.dataset.ends[motion_ids]
            self.first_sample_motion = False
        else:
            motion_len = self.motion_len[env_ids]

        # 确定开始时间
        if self.reset_range is None:
            # 随机选择开始时间，确保有足够的未来步数
            max_len = motion_len - self.future_steps[-1]
            start_phase = torch.rand(len(env_ids), device=self.device)
            start_t = (start_phase * max_len).long()
        else:
            # 使用指定的重置范围
            start_t = torch.randint(*self.reset_range, (len(env_ids),), device=self.device)
            
        # 非训练模式或记录模式：从开始位置开始
        if not self.env.training or self.record_motion:
            start_t.fill_(0)

        # 重放运动模式：循环播放
        if self.replay_motion:
            self.replay_motion_t[env_ids] = (self.replay_motion_t[env_ids] + 1) % motion_len
            start_t = self.replay_motion_t[env_ids]

        self.t[env_ids] = start_t


    def sample_init(self, env_ids: torch.Tensor) -> None:
        """
        采样并初始化指定环境的机器人状态
        
        Args:
            env_ids: 需要初始化的环境ID列表
        """
        self._sample_motions(env_ids)

        # 从运动数据中获取重置状态
        self._motion_reset: TwistMotionData = self.dataset.get_slice(self.motion_ids[env_ids], self.t[env_ids], 1).squeeze(1)
        # 形状: [len(env_ids), num_bodies/num_joints, 3/4/...]
        
        motion = self._motion_reset
        # 提取根身体状态
        init_root_pos = motion.body_pos_w[:, self.root_body_idx_motion]
        init_root_quat = motion.body_quat_w[:, self.root_body_idx_motion]
        init_root_lin_vel = motion.body_lin_vel_w[:, self.root_body_idx_motion]
        init_root_ang_vel = motion.body_ang_vel_w[:, self.root_body_idx_motion]

        # 位置和姿态随机化
        rand_samples = sample_uniform(self.pose_range[:, 0], self.pose_range[:, 1], (len(env_ids), 6), device=self.device)
        if not self.env.training:
            rand_samples.fill_(0.0)
        positions = init_root_pos + self.env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = quat_mul(init_root_quat, orientations_delta)

        # 速度随机化
        rand_samples = sample_uniform(self.velocity_range[:, 0], self.velocity_range[:, 1], (len(env_ids), 6), device=self.device)
        if not self.env.training:
            rand_samples.fill_(0.0)
        velocities = torch.cat([init_root_lin_vel, init_root_ang_vel], dim=-1) + rand_samples

        # 将根身体状态写入仿真
        self.asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)

        # 初始化关节状态
        init_joint_pos = motion.joint_pos[:, self.asset_joint_idx_motion]
        init_joint_vel = motion.joint_vel[:, self.asset_joint_idx_motion]

        # 添加关节噪声
        joint_pos_noise = sample_uniform(-1, 1, (init_joint_pos.shape[0], init_joint_pos.shape[1]), device=self.device) * self.init_joint_pos_noise
        joint_vel_noise = sample_uniform(-1, 1, (init_joint_vel.shape[0], init_joint_vel.shape[1]), device=self.device) * self.init_joint_vel_noise

        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        # 限制关节状态在有效范围内
        joint_pos_limits = self.asset.data.soft_joint_pos_limits[env_ids]
        joint_vel_limits = self.asset.data.soft_joint_vel_limits[env_ids]
        init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        # 将关节状态写入仿真
        self.asset.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids)

        # 记录运动模式：保存之前的运动数据
        if self.record_motion:
            if len(self.motion_frames) > 0:
                self._save_motion()
                self.motion_frames = []
    
    def _save_motion(self):
        """
        保存记录的运动数据到文件
        
        将收集的运动帧数据保存为npz格式，并生成相应的元数据文件
        """
        motion_data: TensorDict = torch.cat(self.motion_frames, dim=0)
        motion_data = motion_data[25:].numpy()  # 跳过前25帧
        moton_meta = {
            "joint_names": self.asset.joint_names,
            "body_names": self.asset.body_names,
            "fps": int(1/self.env.step_dt),
        }
        save_dir = "record_motion"
        motion_data_path = f"{save_dir}/motion.npz"
        motion_meta_path = f"{save_dir}/meta.json"
        import os
        import json
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(motion_data_path, **motion_data)
        with open(motion_meta_path, "w") as f:
            json.dump(moton_meta, f, indent=4)
        print(f"Saved recorded motion to {motion_data_path} and {motion_meta_path}")
        breakpoint()
            

    @property
    def success(self):
        """
        检查运动是否成功完成（到达倒数第二步）
        
        Returns:
            torch.Tensor: 形状为[num_envs, 1]的布尔张量，表示每个环境是否成功
        """
        return (self.t >= self.motion_len - 1).unsqueeze(1)
    
    @property
    def finished(self):
        """
        检查运动是否完全结束
        
        Returns:
            torch.Tensor: 形状为[num_envs, 1]的布尔张量，表示每个环境是否结束
        """
        if self.replay_motion:
            return torch.ones(self.num_envs, 1, dtype=bool, device=self.device)
        return (self.t >= self.motion_len).unsqueeze(1)

    def update(self):
        """
        更新运动跟踪状态
        
        每步调用此方法以：
        1. 记录当前运动帧（如果启用记录模式）
        2. 获取未来参考运动数据
        3. 更新机器人和参考状态
        4. 更新可视化标记
        5. 推进时间步
        """
        # 记录运动帧（如果启用记录模式）
        if hasattr(self, "motion_frames"):
            motion_frame = {}
            motion_frame["body_pos_w"] = self.asset.data.body_link_pos_w.cpu()
            motion_frame["body_quat_w"] = self.asset.data.body_link_quat_w.cpu()
            motion_frame["body_lin_vel_w"] = self.asset.data.body_com_lin_vel_w.cpu()
            motion_frame["body_ang_vel_w"] = self.asset.data.body_com_ang_vel_w.cpu()
            motion_frame["joint_pos"] = self.asset.data.joint_pos.cpu()
            motion_frame["joint_vel"] = self.asset.data.joint_vel.cpu()
            self.motion_frames.append(TensorDict(motion_frame, batch_size=[1]))
            
        # 获取未来参考运动数据，用于智能体观察
        self.future_ref_motion = self.dataset.get_slice(self.motion_ids, self.t, steps=self.future_steps)
        # 形状: [num_envs, len(future_steps), num_bodies/num_joints, 3/4/...]

        # 观察：未来参考身体和关节状态
        self.ref_body_pos_future_w = self.future_ref_motion.body_pos_w[..., self.tracking_body_indices_motion, :] + self.env.scene.env_origins[:, None, None, :]
        self.ref_body_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.tracking_body_indices_motion, :]
        self.ref_body_quat_future_w = self.future_ref_motion.body_quat_w[..., self.tracking_body_indices_motion, :]
        self.ref_body_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[..., self.tracking_body_indices_motion, :]
        self.ref_joint_pos_future_ = self.future_ref_motion.joint_pos[..., self.tracking_joint_indices_motion]
        self.ref_joint_vel_future_ = self.future_ref_motion.joint_vel[..., self.tracking_joint_indices_motion]
        self.ref_root_pos_future_w = self.future_ref_motion.body_pos_w[..., self.root_body_idx_motion, :] + self.env.scene.env_origins[:, None, :]
        self.ref_root_quat_future_w = self.future_ref_motion.body_quat_w[..., self.root_body_idx_motion, :]
        self.ref_root_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.root_body_idx_motion, :]
        self.ref_root_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[..., self.root_body_idx_motion, :]

        # 奖励：当前机器人身体和关节状态
        self.robot_body_pos_w = self.asset.data.body_link_pos_w[:, self.tracking_body_indices_asset]
        self.robot_body_lin_vel_w = self.asset.data.body_com_lin_vel_w[:, self.tracking_body_indices_asset]
        self.robot_body_quat_w = self.asset.data.body_link_quat_w[:, self.tracking_body_indices_asset]
        self.robot_body_ang_vel_w = self.asset.data.body_com_ang_vel_w[:, self.tracking_body_indices_asset]
        self.robot_joint_pos = self.asset.data.joint_pos[:, self.tracking_joint_indices_asset]
        self.robot_joint_vel = self.asset.data.joint_vel[:, self.tracking_joint_indices_asset]
        self.robot_root_pos_w = self.asset.data.root_link_pos_w
        self.robot_root_quat_w = self.asset.data.root_link_quat_w

        # 奖励：当前参考身体和关节状态
        self.current_ref_motion: TwistMotionData = self.future_ref_motion[:, 0]
        self.ref_body_pos_w = self.ref_body_pos_future_w[:, 0]
        self.ref_body_lin_vel_w = self.ref_body_lin_vel_future_w[:, 0]
        self.ref_body_quat_w = self.ref_body_quat_future_w[:, 0]
        self.ref_body_ang_vel_w = self.ref_body_ang_vel_future_w[:, 0]
        self.ref_joint_pos = self.ref_joint_pos_future_[:, 0]
        self.ref_joint_vel = self.ref_joint_vel_future_[:, 0]
        self.ref_root_pos_w = self.ref_root_pos_future_w[:, 0]
        self.ref_root_quat_w = self.ref_root_quat_future_w[:, 0]
        # 形状: [num_envs, num_future_steps, num_tracking_bodies, xxx]

        # 更新Isaac Lab可视化标记
        if self.env.backend == "isaac":
            self.all_marker_pos_w[0] = self.robot_body_pos_w
            self.all_marker_pos_w[1] = self.ref_body_pos_w
            # self.all_marker_pos_w[0] = self.ref_body_pos_future_w[:, 0]
            # self.all_marker_pos_w[1] = self.ref_body_pos_future_w[:, -1]

        # 推进时间步
        self.t += 1
    
    def _init_debug_draw(self):
        """
        初始化调试绘制功能
        
        创建可视化标记，用于在Isaac Lab中显示机器人和参考关键点
        """
        if self.env.backend != "isaac":
            return
        
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Keypoints",
            markers={
                "robot": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0)  # 绿色表示机器人
                    ),
                ),
                "reference": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0)  # 红色表示参考
                    ),
                ),
            },
        )
        self.vis_markers = VisualizationMarkers(vis_markers_cfg)
        num_ref_markers = self.num_envs * self.num_tracking_bodies
        self.marker_indices = [0] * num_ref_markers + [1] * num_ref_markers
        self.all_marker_pos_w = torch.zeros(2, self.num_envs, self.num_tracking_bodies, 3, device=self.device)

    def debug_draw(self):
        """
        绘制调试信息
        
        在Isaac Lab中可视化机器人和参考关键点位置
        """
        if self.env.backend != "isaac":
            return

        # 重放运动模式：隐藏标记
        if self.replay_motion:
            self.all_marker_pos_w.fill_(-1000)
        
        # 形状: [2, num_envs, num_tracking_bodies, 3]
        self.vis_markers.visualize(
            translations=self.all_marker_pos_w.reshape(-1, 3),
            marker_indices=self.marker_indices,
        )

        # 可选：绘制从机器人到目标关键点的向量
        # robot_keypoints_w = self.all_marker_pos_w[0].reshape(-1, 3)
        # target_keypoints_w = self.all_marker_pos_w[1].reshape(-1, 3)
        # self.env.debug_draw.vector(
        #     robot_keypoints_w,
        #     target_keypoints_w - robot_keypoints_w,
        #     color=(0, 0, 1, 1)
        # )
