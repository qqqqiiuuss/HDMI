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
from active_adaptation.utils.motion import MotionDataset, MotionData

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

class RobotTracking(Command):
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
        tracking_keypoint_names: List[str],
        tracking_joint_names: List[str],
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
        
        # 创建运动数据集，将数据加载到指定设备
        self.dataset = MotionDataset.create_from_path(
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
        self._motion_reset: MotionData = self.dataset.get_slice(self.motion_ids[env_ids], self.t[env_ids], 1).squeeze(1)
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
        self.current_ref_motion: MotionData = self.future_ref_motion[:, 0]
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

class RobotObjectTracking(RobotTracking):
    """
    带物体交互的机器人运动跟踪命令类
    
    继承自RobotTracking，增加了物体交互功能，包括：
    - 物体位置跟踪
    - 接触检测和奖励
    - 末端执行器与物体的交互
    
    Args:
        extra_object_names: 额外物体名称列表
        object_asset_name: 场景中物体的资产名称
        object_body_name: 定义接触目标位置的身体名称
        object_joint_name: 要跟踪的物体关节名称（可选）
        object_pose_range: 物体位置和姿态的随机化范围
        object_init_joint_pos_noise: 物体初始关节位置噪声
        object_init_joint_vel_noise: 物体初始关节速度噪声
        contact_eef_body_name: 接触末端执行器身体名称列表
        contact_frc_eef_body_name: 接触力末端执行器身体名称列表
        contact_target_pos_offset: 从物体到接触目标位置的偏移
        contact_eef_pos_offset: 从末端执行器的偏移
        **kwargs: 传递给父类的其他参数
    """
    def __init__(
        self,
        extra_object_names: List[str],
        object_asset_name: str, # for finding the object in the scene
        object_body_name: str, # for the body that defines the contact target position
        object_joint_name: str | None = None, # object joint to track
        # for reset
        object_pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        object_init_joint_pos_noise: float = 0.1, 
        object_init_joint_vel_noise: float = 0.1,
        # for contact rewards
        contact_eef_body_name: List[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        contact_frc_eef_body_name: List[str | List[str]] = ["left_wrist_(roll|pitch|yaw)_link", "right_wrist_(roll|pitch|yaw)_link"],
        ## offset from object to contact target position
        contact_target_pos_offset: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        ## offset from end effector
        contact_eef_pos_offset: List[Tuple[float, float, float]] = [(0.1, 0.0, 0.0), (0.1, 0.0, 0.0)],
        **kwargs
    ):
        super().__init__(**kwargs, call_update=False)

        # 获取额外物体对象
        self.extra_objects: List[Articulation | RigidObject] = [self.env.scene[name] for name in extra_object_names]
        self.extra_object_body_id_motion = [self.dataset.body_names.index(name) for name in extra_object_names]

        # 设置主要物体对象
        self.object_asset_name = object_asset_name
        if object_joint_name is None:
            # 刚体对象
            self.object = self.env.scene.rigid_objects[object_asset_name]
            self.object_joint_idx_motion = None
            self.object_joint_idx_asset = None
        else:
            # 关节对象
            self.object = self.env.scene.articulations[object_asset_name]
            self.object_joint_idx_motion = self.dataset.joint_names.index(object_joint_name)
            self.object_joint_idx_asset = self.object.joint_names.index(object_joint_name)
        
        # 获取物体身体索引
        self.object_body_id_asset = self.object.body_names.index(object_body_name)
        self.object_body_id_motion = self.dataset.body_names.index(object_asset_name)

        # 物体姿态随机化范围
        pose_range_list = [object_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.object_pose_range = torch.tensor(pose_range_list, device=self.device)
        self.object_init_joint_pos_noise = object_init_joint_pos_noise
        self.object_init_joint_vel_noise = object_init_joint_vel_noise

        # 重放或记录模式：禁用物体随机化
        if self.replay_motion or self.record_motion:
            self.object_pose_range.fill_(0.0)
            self.object_init_joint_pos_noise = 0.0
            self.object_init_joint_vel_noise = 0.0

        # 设置接触身体索引
        assert len(contact_eef_body_name) == len(contact_target_pos_offset) == len(contact_eef_pos_offset), \
            "contact_eef_body_name, contact_target_pos_offset, and contact_eef_pos_offset must have the same length"
        self.num_eefs = len(contact_eef_body_name)
        self.contact_eef_body_indices_asset = [self.asset.body_names.index(name) for name in contact_eef_body_name]

        # 设置接触力传感器
        self.eef_filtered_sensor: List[List[ContactSensor]] = []
        self.eef_filtered_sensor_indices: List[List[int]] = []
        for eef_name in contact_eef_body_name:
            eef_names = self.asset.find_bodies(eef_name)[1]
            sensors_for_this_eef = []
            sensor_indices_for_this_eef = []
            for eef_name in eef_names:
                eef_sensor_name = f"{eef_name}_{object_asset_name}_contact_forces"
                eef_sensor_filtered = self.env.scene.sensors[eef_sensor_name]
                sensors_for_this_eef.append(eef_sensor_filtered)
                sensor_indices_for_this_eef.append(eef_sensor_filtered.body_names.index(eef_name))
            self.eef_filtered_sensor.append(sensors_for_this_eef)
            self.eef_filtered_sensor_indices.append(sensor_indices_for_this_eef)

        # 初始化接触相关张量
        with torch.device(self.device):
            self.contact_target_pos_offset = torch.tensor(contact_target_pos_offset, device=self.device).repeat(self.num_envs, 1, 1)
            self.contact_eef_pos_offset = torch.tensor(contact_eef_pos_offset, device=self.device).repeat(self.num_envs, 1, 1)

            self.contact_target_pos_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
            self.contact_eef_pos_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)

            self.eef_contact_forces_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
            self.eef_contact_forces_b = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
        
        # 处理物体缩放
        scale = getattr(self.object.cfg.spawn, "scale", None)
        if not isinstance(scale, torch.Tensor):
            scale_tensor = torch.ones(self.num_envs, 3)
            if scale is None:
                pass
            elif isinstance(scale, float):
                scale_tensor[:] = scale
            elif isinstance(scale, tuple):
                scale_tensor[:] = torch.tensor(scale)
            else:
                raise ValueError(f"Invalid scale type: {type(scale)}")
            scale = scale_tensor
        self.contact_target_pos_offset *= scale.unsqueeze(1).to(self.device)

        # 加载物体接触数据
        motion_paths = self.dataset.motion_paths
        assert len(motion_paths) == 1, "Only one motion path is supported for RobotObjectTracking"
        motion_data = np.load(motion_paths[0], allow_pickle=True)
        object_contact = motion_data["object_contact"]
        self._object_contact = torch.tensor(object_contact, device=self.device, dtype=torch.bool)
        # if self._object_contact.shape[1] == 1:
        #     # expand to num_eefs
        #     self._object_contact = self._object_contact.repeat(1, self.num_eefs)
        # # shape: [num_steps, num_eefs/1]

        # 初始化调试绘制和更新
        self._init_debug_draw()
        self.update()
        if self.record_motion:
            self.motion_frames = []
    
    def sample_init(self, env_ids):
        """
        采样并初始化指定环境的机器人和物体状态
        
        Args:
            env_ids: 需要初始化的环境ID列表
        """
        super().sample_init(env_ids)
         
        # 初始化主要物体位置和姿态
        init_object_pos = self._motion_reset.body_pos_w[:, self.object_body_id_motion]
        init_object_quat = self._motion_reset.body_quat_w[:, self.object_body_id_motion]

        # 物体位置和姿态随机化
        rand_samples = sample_uniform(self.object_pose_range[:, 0], self.object_pose_range[:, 1], (len(env_ids), 6), device=self.device)

        init_object_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        init_object_quat = quat_mul(init_object_quat, orientations_delta)
        
        # 设置物体根状态
        init_object_state_w = self.object.data.default_root_state[env_ids]
        init_object_state_w[:, 0:3] = init_object_pos + self.env.scene.env_origins[env_ids]
        init_object_state_w[:, 3:7] = init_object_quat
        init_object_state_w[:, 7:]  = 0.0  # 零速度

        self.object.write_root_link_pose_to_sim(init_object_state_w[:, 0:7], env_ids=env_ids)
        self.object.write_root_com_velocity_to_sim(init_object_state_w[:, 7:], env_ids=env_ids)

        # 初始化额外物体
        for object_, object_body_id_motion in zip(self.extra_objects, self.extra_object_body_id_motion):
            init_object_pos = self._motion_reset.body_pos_w[:, object_body_id_motion]
            init_object_quat = self._motion_reset.body_quat_w[:, object_body_id_motion]
            
            init_object_pos += rand_samples[:, 0:3]
            init_object_quat = quat_mul(init_object_quat, orientations_delta)

            init_object_state_w = object_.data.default_root_state[env_ids]
            init_object_state_w[:, 0:3] = init_object_pos + self.env.scene.env_origins[env_ids]
            init_object_state_w[:, 3:7] = init_object_quat
            init_object_state_w[:, 7:]  = 0.0  # 零速度

            object_.write_root_link_pose_to_sim(init_object_state_w[:, 0:7], env_ids=env_ids)
            object_.write_root_com_velocity_to_sim(init_object_state_w[:, 7:], env_ids=env_ids)

        # 调试：计算物体在机器人坐标系中的位置（已注释）
        # robot_pos_w = self.asset.data.root_link_pos_w[env_ids]
        # robot_quat_w = self.asset.data.root_link_quat_w[env_ids]
        # object_pos_b = quat_apply_inverse(robot_quat_w, (init_object_pos + self.env.scene.env_origins[env_ids]) - robot_pos_w)
        # from isaaclab.utils.math import quat_conjugate
        # object_quat_b = quat_mul(quat_conjugate(robot_quat_w), init_object_quat)
        # print(f"Object initial position in robot frame: {object_pos_b}, orientation: {object_quat_b}")

        # 初始化物体关节状态（如果存在）
        if self.object_joint_idx_asset is not None:
            init_joint_pos = self._motion_reset.joint_pos[:, self.object_joint_idx_motion].unsqueeze(1)
            init_joint_vel = self._motion_reset.joint_vel[:, self.object_joint_idx_motion].unsqueeze(1)

            # 添加关节噪声
            joint_pos_noise = sample_uniform(-1, 1, (init_joint_pos.shape[0], init_joint_pos.shape[1]), device=self.device) * self.object_init_joint_pos_noise
            joint_vel_noise = sample_uniform(-1, 1, (init_joint_vel.shape[0], init_joint_vel.shape[1]), device=self.device) * self.object_init_joint_vel_noise
            
            init_joint_pos += joint_pos_noise
            init_joint_vel += joint_vel_noise
            
            # 限制关节状态在有效范围内
            joint_pos_limits = self.object.data.soft_joint_pos_limits[env_ids]
            joint_vel_limits = self.object.data.soft_joint_vel_limits[env_ids]
            init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
            init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

            self.object.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids, joint_ids=[self.object_joint_idx_asset])

    def _save_motion(self):
        """
        保存记录的运动数据到文件（包含物体接触信息）
        
        将收集的运动帧数据和物体接触信息保存为npz格式，并生成相应的元数据文件
        """
        motion_data: TensorDict = torch.cat(self.motion_frames, dim=0)
        motion_data = motion_data[25:].numpy()  # 跳过前25帧
        motion_data["object_contact"] = self._object_contact[25:].cpu().numpy()  # 添加物体接触信息
        moton_meta = {
            "joint_names": self.asset.joint_names,
            "body_names": self.asset.body_names + [self.object_asset_name],  # 包含物体名称
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

    def update(self):
        """
        更新带物体交互的运动跟踪状态
        
        继承父类的update方法，并添加物体相关的状态更新：
        1. 记录物体运动帧数据
        2. 更新物体位置和姿态
        3. 更新接触目标位置
        4. 计算接触力
        """
        super().update()
        
        # 记录物体运动帧数据
        if hasattr(self, "motion_frames"):
            motion_frame = self.motion_frames[-1]
            # 添加物体数据到运动帧
            object_pos_w = self.object.data.body_link_pos_w[:, self.object_body_id_asset].cpu()
            object_quat_w = self.object.data.body_link_quat_w[:, self.object_body_id_asset].cpu()
            object_lin_vel_w = self.object.data.body_com_lin_vel_w[:, self.object_body_id_asset].cpu()
            object_ang_vel_w = self.object.data.body_com_ang_vel_w[:, self.object_body_id_asset].cpu()
            motion_frame["body_pos_w"] = torch.cat([motion_frame["body_pos_w"], object_pos_w.unsqueeze(1)], dim=1)
            motion_frame["body_quat_w"] = torch.cat([motion_frame["body_quat_w"], object_quat_w.unsqueeze(1)], dim=1)
            motion_frame["body_lin_vel_w"] = torch.cat([motion_frame["body_lin_vel_w"], object_lin_vel_w.unsqueeze(1)], dim=1)
            motion_frame["body_ang_vel_w"] = torch.cat([motion_frame["body_ang_vel_w"], object_ang_vel_w.unsqueeze(1)], dim=1)

        # 更新参考物体位置和姿态
        self.ref_object_pos_future_w = self.future_ref_motion.body_pos_w[..., self.object_body_id_motion, :] + self.env.scene.env_origins[:, None, :]
        self.ref_object_quat_future_w = self.future_ref_motion.body_quat_w[..., self.object_body_id_motion, :]
        self.ref_object_pos_w = self.ref_object_pos_future_w[:, 0]
        self.ref_object_quat_w = self.ref_object_quat_future_w[:, 0]
        self.object_pos_w = self.object.data.root_link_pos_w
        self.object_quat_w = self.object.data.root_link_quat_w

        # 更新物体关节状态（如果存在）
        if self.object_joint_idx_asset is not None:
            self.ref_object_joint_pos_future = self.future_ref_motion.joint_pos[..., self.object_joint_idx_motion]
            self.ref_object_joint_vel_future = self.future_ref_motion.joint_vel[..., self.object_joint_idx_motion]
            self.ref_object_joint_pos = self.ref_object_joint_pos_future[:, 0]
            self.ref_object_joint_vel = self.ref_object_joint_vel_future[:, 0]
            self.object_joint_pos = self.object.data.joint_pos[:, self.object_joint_idx_asset]
            self.object_joint_vel = self.object.data.joint_vel[:, self.object_joint_idx_asset]
            
        # 更新物体接触状态
        idx = (self.motion_starts + self.t).unsqueeze(1) + self.future_steps.unsqueeze(0)
        idx.clamp_max_(self.motion_ends.unsqueeze(1) - 1)
        self.ref_object_contact_future = self._object_contact[idx]
        self.ref_object_contact = self.ref_object_contact_future[:, 0]
        
        # 更新接触目标位置和末端执行器位置
        object_pos_w = self.object.data.body_link_pos_w[:, self.object_body_id_asset]
        object_quat_w = self.object.data.body_link_quat_w[:, self.object_body_id_asset]
        self.contact_target_pos_w[:] = object_pos_w.unsqueeze(1) + quat_apply(object_quat_w.unsqueeze(1), self.contact_target_pos_offset)
        
        eef_pos_w = self.asset.data.body_link_pos_w[:, self.contact_eef_body_indices_asset]
        eef_quat_w = self.asset.data.body_link_quat_w[:, self.contact_eef_body_indices_asset]
        self.contact_eef_pos_w[:] = eef_pos_w + quat_apply(eef_quat_w, self.contact_eef_pos_offset)
        
        # 计算末端执行器接触力
        self.eef_contact_forces_w.zero_()
        for eef_idx, (eef_sensors, eef_sensor_indices) in enumerate(zip(self.eef_filtered_sensor, self.eef_filtered_sensor_indices)):
            for eef_sensor, eef_sensor_id in zip(eef_sensors, eef_sensor_indices):
                self.eef_contact_forces_w[:, eef_idx] += eef_sensor.data.force_matrix_w[:, eef_sensor_id, 0]

        # 将接触力转换到物体坐标系
        self.eef_contact_forces_b[:] = quat_apply_inverse(object_quat_w.unsqueeze(1), self.eef_contact_forces_w)

    def _init_debug_draw(self):
        """
        初始化带物体交互的调试绘制功能
        
        继承父类的调试绘制功能，并添加末端执行器接触标记
        """
        super()._init_debug_draw()

        if self.env.backend != "isaac":
            return
        
        from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
        import isaaclab.sim as sim_utils
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path="/World/EefContact",
            markers={
                "left": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.3),  # 青绿色表示左末端执行器
                        metallic=1.0,
                    )
                ),
                "right": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.3, 1.0),  # 蓝色表示右末端执行器
                        metallic=1.0,
                    )
                ),
            }
        )
        self.eef_contact_markers = VisualizationMarkers(vis_markers_cfg)
        self.eef_contact_markers_indices = [0, 1] * (self.num_envs * self.num_eefs)
        self.eef_contact_markers_pos_w = torch.zeros(self.num_envs, 2, self.num_eefs, 3)

    def debug_draw(self):
        """
        绘制带物体交互的调试信息
        
        继承父类的调试绘制功能，并添加：
        1. 末端执行器和接触目标位置标记
        2. 接触力向量可视化
        3. 从末端执行器到接触目标的向量
        """
        super().debug_draw()

        if self.env.backend != "isaac":
            return
        
        # 设置末端执行器接触标记位置
        self.eef_contact_markers_pos_w[:, 0, :, :] = self.contact_eef_pos_w
        self.eef_contact_markers_pos_w[:, 1, :, :] = self.contact_target_pos_w
        # 超出范围的接触隐藏标记
        out_of_range_mask = ~self.ref_object_contact[:, None, :, None].expand_as(self.eef_contact_markers_pos_w)
        self.eef_contact_markers_pos_w[out_of_range_mask] = -1000.0
        
        self.eef_contact_markers.visualize(
            translations=self.eef_contact_markers_pos_w.view(-1, 3),
            marker_indices=self.eef_contact_markers_indices,
        )

        # 可视化接触力向量
        self.env.debug_draw.vector(
            self.contact_eef_pos_w.reshape(-1, 3),
            self.eef_contact_forces_w.reshape(-1, 3) / 80,  # 缩放力向量以便可视化
            color=(1.0, 1.0, 1.0, 1.0),  # 白色
            size=4.0,
        )

        # 绘制从末端执行器到接触目标的向量
        self.env.debug_draw.vector(
            self.contact_eef_pos_w.view(-1, 3),
            (self.contact_target_pos_w - self.contact_eef_pos_w).view(-1, 3),
            color=(0, 1, 0, 1),  # 绿色
            size=4.0,
        )
