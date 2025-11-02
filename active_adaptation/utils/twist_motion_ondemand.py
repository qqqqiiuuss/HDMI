"""
On-Demand Motion Dataset with Hard Negative Mining

按需加载 Motion 数据集，支持：
1. On-Demand Loading: 只加载当前 batch 需要的 motion
2. Hard Negative Mining: 根据成功率动态调整采样权重
3. Motion Filtering: 过滤无法学习的 motion
4. 显存可控: 4096 环境只需 2-3GB 显存

使用方法:
    dataset = OnDemandTwistMotionDataset.create_from_yaml(
        "path/to/dataset.yaml",
        device="cuda",
        enable_hnm=True
    )

    # 采样 motion IDs
    motion_ids = dataset.sample_motions(4096)

    # 加载当前 batch
    dataset.load_batch(motion_ids)

    # 获取数据
    data = dataset.get_slice(motion_ids, time_steps, steps)

    # 更新 HNM 权重
    dataset.update_hnm(motion_ids, success_flags)
"""

import torch
import numpy as np
import pickle
import yaml
from pathlib import Path
from typing import List, Union, Dict
from tqdm import tqdm
from collections import OrderedDict

# 导入原有的 TwistMotionData
from .twist_motion import TwistMotionData, interpolate


class OnDemandTwistMotionDataset:
    """按需加载的 Motion 数据集，支持 Hard Negative Mining"""

    def __init__(
        self,
        motion_paths: List[Path],
        body_names: List[str],
        joint_names: List[str],
        device: str = "cuda",
        # Hard Negative Mining 配置
        enable_hnm: bool = True,
        hnm_alpha: float = 1.5,
        hnm_beta: float = 0.7,
        hnm_min_weight: float = 1e-6,
        hnm_boost_unsampled: float = 1.1,
        # Motion Filtering 配置
        hnm_filter_enabled: bool = True,
        hnm_min_attempts: int = 200,
        hnm_max_failure_rate: float = 0.99,
        # 缓存管理配置
        max_cache_size: int = 3000,  # 最大缓存的 motion 数量
    ):
        """
        Args:
            motion_paths: 所有 motion 文件路径列表
            body_names: Body 名称列表
            joint_names: Joint 名称列表
            device: 设备 (cuda/cpu)
            enable_hnm: 是否启用 Hard Negative Mining
            hnm_alpha: 失败样本权重乘数 (>1.0)
            hnm_beta: 成功样本权重乘数 (<1.0)
            hnm_min_weight: 最小权重（保证全覆盖）
            hnm_boost_unsampled: 未采样 motion 权重提升倍数
            hnm_filter_enabled: 是否过滤不可能的 motion
            hnm_min_attempts: 过滤前最少尝试次数
            hnm_max_failure_rate: 过滤阈值（失败率）
        """
        self.device = device
        self.num_motions = len(motion_paths)
        self.motion_paths = motion_paths
        self.body_names = body_names
        self.joint_names = joint_names
        self.max_cache_size = max_cache_size

        # 如果数据集很小，直接全部加载到 GPU（不使用 on-demand loading）
        self.preload_all = (self.num_motions < 100)

        print(f"📊 Initializing OnDemandTwistMotionDataset")
        print(f"   Total motions: {self.num_motions}")
        print(f"   Device: {device}")
        print(f"   HNM enabled: {enable_hnm}")
        print(f"   Max cache size: {max_cache_size}")
        if self.preload_all:
            print(f"   ⚡ Small dataset detected - preloading all {self.num_motions} motions to GPU")

        # ==================== Hard Negative Mining ====================
        # 必须在 _scan_metadata() 之前初始化，因为扫描时需要访问这些属性
        self.enable_hnm = enable_hnm
        self.hnm_alpha = hnm_alpha
        self.hnm_beta = hnm_beta
        self.hnm_min_weight = hnm_min_weight
        self.hnm_boost_unsampled = hnm_boost_unsampled
        self.hnm_filter_enabled = hnm_filter_enabled
        self.hnm_min_attempts = hnm_min_attempts
        self.hnm_max_failure_rate = hnm_max_failure_rate

        if self.enable_hnm:
            # 采样权重（初始均匀分布）
            self.motion_weights = torch.ones(self.num_motions, device=device) / self.num_motions

            # 成功率统计
            self.success_count = torch.zeros(self.num_motions, device=device)
            self.attempt_count = torch.zeros(self.num_motions, device=device)
            self.success_rate = torch.zeros(self.num_motions, device=device)

            # 过滤标记
            self.filtered_mask = torch.ones(self.num_motions, dtype=torch.bool, device=device)

            print(f"✅ Hard Negative Mining initialized")
            print(f"   Alpha (failure boost): {hnm_alpha}")
            print(f"   Beta (success reduce): {hnm_beta}")
            print(f"   Min weight: {hnm_min_weight}")

        # ==================== 元数据 ====================
        # 元数据需要在同一设备上，避免设备不匹配
        self.motion_lengths = torch.zeros(self.num_motions, dtype=torch.long, device=device)
        self.motion_fps = torch.zeros(self.num_motions, dtype=torch.float32, device=device)
        self.starts = torch.zeros(self.num_motions, dtype=torch.long, device=device)
        self.ends = torch.zeros(self.num_motions, dtype=torch.long, device=device)

        # 快速扫描元数据（不加载实际数据）
        print(f"🔍 Scanning metadata...")
        self._scan_metadata()

        self.lengths = self.motion_lengths  # 兼容原有 API

        # ==================== GPU 缓存（当前 batch）====================
        # LRU 缓存管理
        self.current_batch_data = OrderedDict()  # {motion_id: motion_data_dict}
        self.motion_access_time = {}  # {motion_id: access_counter}
        self.access_counter = 0  # 全局访问计数器

        # 如果数据集很小，预加载所有数据到 GPU
        if self.preload_all:
            all_motion_ids = list(range(self.num_motions))
            print(f"🔄 Preloading all {self.num_motions} motions to GPU...")
            self.load_batch(all_motion_ids)
            print(f"✅ Preloaded {len(self.current_batch_data)} motions")

        print(f"✅ OnDemandTwistMotionDataset initialized")

    def _scan_metadata(self):
        """快速扫描所有 motion 的元数据（不加载实际数据）"""
        start_idx = 0

        for i, path in enumerate(tqdm(self.motion_paths, desc="Scanning metadata")):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)

                    # 检查格式
                    if 'body_pos_w' in data:
                        length = data['body_pos_w'].shape[0]
                    elif 'root_pos' in data:
                        length = data['root_pos'].shape[0]
                    else:
                        print(f"⚠️  Unknown format: {path.name}")
                        length = 0

                    fps = data.get('fps', 50.0)

                    self.motion_lengths[i] = length
                    self.motion_fps[i] = fps
                    self.starts[i] = start_idx
                    self.ends[i] = start_idx + length
                    start_idx += length

            except Exception as e:
                print(f"❌ Failed to scan {path.name}: {e}")
                self.motion_lengths[i] = 0
                self.motion_fps[i] = 0
                self.starts[i] = start_idx
                self.ends[i] = start_idx

                # 如果启用 HNM，标记为无效并设置权重为 0
                if self.enable_hnm:
                    self.filtered_mask[i] = False
                    self.motion_weights[i] = 0.0

        # 重新归一化权重（排除无效的 motions）
        if self.enable_hnm:
            valid_mask = self.motion_lengths > 0
            if valid_mask.sum() > 0:
                self.motion_weights = self.motion_weights * valid_mask.float()
                self.motion_weights = self.motion_weights / self.motion_weights.sum()
            else:
                print("❌ ERROR: No valid motions found!")

        num_valid = (self.motion_lengths > 0).sum().item()
        num_invalid = self.num_motions - num_valid

        print(f"✅ Scanned {self.num_motions} motions")
        print(f"   Total frames: {start_idx}")
        print(f"   Valid motions: {num_valid}")
        if num_invalid > 0:
            print(f"   ⚠️  Invalid motions: {num_invalid} (will be filtered out)")

    def sample_motions(self, n: int) -> torch.Tensor:
        """根据 Hard Negative Mining 权重采样 motion IDs

        Args:
            n: 采样数量（通常是 num_envs）

        Returns:
            motion_ids: [n] 采样的 motion ID
        """
        if self.enable_hnm:
            # 过滤掉被标记为无效的 motion
            valid_weights = self.motion_weights * self.filtered_mask.float()
            valid_weights = valid_weights / valid_weights.sum()

            # 加权采样
            motion_ids = torch.multinomial(
                valid_weights,
                num_samples=n,
                replacement=True
            )
        else:
            # 均匀随机采样
            motion_ids = torch.randint(0, self.num_motions, (n,), device=self.device)

        return motion_ids

    def load_batch(
        self,
        motion_ids: Union[torch.Tensor, np.ndarray],
        target_fps: int = 50,
        isaac_joint_names: List[str] = None
    ):
        """加载一个 batch 需要的所有 motion 到 GPU（增量加载）

        Args:
            motion_ids: [num_unique_motions] 当前 batch 需要的 unique motion IDs (可以是 torch.Tensor 或 numpy.ndarray)
            target_fps: 目标帧率（用于插值）
            isaac_joint_names: Isaac 关节名称（用于重新映射）
        """
        # 1. 转换为 numpy array（如果是 tensor）
        if isinstance(motion_ids, torch.Tensor):
            unique_ids = motion_ids.cpu().numpy()
        else:
            unique_ids = np.asarray(motion_ids)

        # 2. 更新访问时间（LRU 追踪）
        for mid in unique_ids:
            mid_int = int(mid)
            if mid_int in self.current_batch_data:
                self.motion_access_time[mid_int] = self.access_counter
                self.access_counter += 1

        # 3. 找出需要新加载的 motion IDs（增量加载）
        new_motion_ids = [mid for mid in unique_ids if int(mid) not in self.current_batch_data]

        if len(new_motion_ids) == 0:
            # 所有 motions 都已在缓存中，无需加载
            return

        # 4. 检查缓存大小，如果超过限制则淘汰最少使用的 motions
        num_to_evict = len(self.current_batch_data) + len(new_motion_ids) - self.max_cache_size
        if num_to_evict > 0:
            # 按访问时间排序，移除最旧的
            sorted_motions = sorted(self.motion_access_time.items(), key=lambda x: x[1])
            motions_to_remove = [mid for mid, _ in sorted_motions[:num_to_evict]]

            for mid in motions_to_remove:
                del self.current_batch_data[mid]
                del self.motion_access_time[mid]

            print(f"🗑️  Evicted {num_to_evict} motions from cache (LRU)")

        print(f"🔄 Loading {len(new_motion_ids)} new motions (cache: {len(self.current_batch_data)} -> {len(self.current_batch_data) + len(new_motion_ids)})")

        # 3. 从磁盘加载每个新的 unique motion
        for motion_id in tqdm(new_motion_ids, desc="Loading motions", leave=False):
            mid = int(motion_id)
            path = self.motion_paths[mid]

            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)

                # 检查格式并转换
                motion_dict = self._parse_motion_data(data, target_fps)

                # 重新映射关节（如果需要）
                if isaac_joint_names is not None:
                    motion_dict = self._remap_joints(motion_dict, isaac_joint_names)

                # 保存到缓存（保留为 numpy，减少显存占用）
                self.current_batch_data[mid] = motion_dict

                # 更新访问时间
                self.motion_access_time[mid] = self.access_counter
                self.access_counter += 1

            except Exception as e:
                print(f"❌ Failed to load motion {mid} ({path.name}): {e}")
                # 标记为无效并永久过滤
                if self.enable_hnm:
                    self.filtered_mask[mid] = False
                    # 将权重设为 0，避免再次采样
                    self.motion_weights[mid] = 0.0
                    # 重新归一化权重
                    if self.motion_weights.sum() > 0:
                        self.motion_weights = self.motion_weights / self.motion_weights.sum()

                # 设置 motion_lengths 为 0，标记为无效
                self.motion_lengths[mid] = 0

        print(f"✅ Cache now has {len(self.current_batch_data)} motions (max: {self.max_cache_size})")

    def _parse_motion_data(self, data: Dict, target_fps: int) -> Dict:
        """解析并插值 motion 数据

        Args:
            data: 从 pkl 文件加载的原始数据
            target_fps: 目标帧率

        Returns:
            motion_dict: 包含所有字段的字典（numpy 格式）
        """
        fps = data.get('fps', 50.0)

        # 检查格式
        if 'body_pos_w' in data:
            # HDMI 格式
            motion = {
                'body_pos_w': np.array(data['body_pos_w']),
                'body_lin_vel_w': np.array(data['body_lin_vel_w']),
                'body_quat_w': np.array(data['body_quat_w']),
                'body_ang_vel_w': np.array(data['body_ang_vel_w']),
                'joint_pos': np.array(data['joint_pos']),
                'joint_vel': np.array(data['joint_vel']),
            }
        elif 'root_pos' in data and 'dof_pos' in data:
            # GMR 格式 - 需要转换
            motion = self._convert_gmr_to_hdmi(data)
        else:
            raise ValueError("Unknown motion format")

        # 插值到目标帧率
        if abs(fps - target_fps) > 0.1:
            motion = interpolate(motion, source_fps=fps, target_fps=target_fps)

        return motion

    def _convert_gmr_to_hdmi(self, data: Dict) -> Dict:
        """将 GMR 格式转换为 HDMI 格式

        这是简化版本，完整实现在 twist_motion.py
        """
        # 导入转换逻辑（避免循环导入）
        from scipy.spatial.transform import Rotation as R

        root_pos = np.array(data['root_pos'])
        root_rot = np.array(data['root_rot'])[:, [3, 0, 1, 2]]  # wxyz
        dof_pos = np.array(data['dof_pos'])
        local_body_pos = np.array(data['local_body_pos'])

        T = root_pos.shape[0]
        N_bodies = local_body_pos.shape[1]
        fps = data.get('fps', 50.0)
        dt = 1.0 / fps

        # 计算速度
        root_lin_vel = np.zeros_like(root_pos)
        root_lin_vel[:-1] = (root_pos[1:] - root_pos[:-1]) / dt
        root_lin_vel[-1] = root_lin_vel[-2]

        joint_vel = np.zeros_like(dof_pos)
        joint_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) / dt
        joint_vel[-1] = joint_vel[-2]

        # 计算角速度（TWIST 完整实现）
        root_ang_vel = np.zeros((T, 3))

        # 使用 TWIST 方法：root_ang_vel = fps * quat_to_exp_map(quat_diff(q_t, q_{t+1}))
        for i in range(T - 1):
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
            min_theta = 1e-5
            # 防止浮点误差导致负数
            sin_theta_sq = max(0.0, 1 - dq[0] * dq[0])
            sin_theta = np.sqrt(sin_theta_sq)
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

        # 构建世界坐标系的 body 位置
        body_pos_w = np.zeros((T, N_bodies, 3))
        body_quat_w = np.zeros((T, N_bodies, 4))

        for t in range(T):
            root_quat_xyzw = root_rot[t, [1, 2, 3, 0]]
            R_root = R.from_quat(root_quat_xyzw)

            for b in range(N_bodies):
                body_pos_w[t, b] = root_pos[t] + R_root.apply(local_body_pos[t, b])
                body_quat_w[t, b] = root_rot[t]

        body_lin_vel_w = np.zeros_like(body_pos_w)
        body_lin_vel_w[:-1] = (body_pos_w[1:] - body_pos_w[:-1]) / dt
        body_lin_vel_w[-1] = body_lin_vel_w[-2]

        body_ang_vel_w = np.tile(root_ang_vel[:, np.newaxis, :], (1, N_bodies, 1))

        return {
            'body_pos_w': body_pos_w,
            'body_lin_vel_w': body_lin_vel_w,
            'body_quat_w': body_quat_w,
            'body_ang_vel_w': body_ang_vel_w,
            'joint_pos': dof_pos,
            'joint_vel': joint_vel,
        }

    def _remap_joints(self, motion: Dict, isaac_joint_names: List[str]) -> Dict:
        """重新映射关节顺序（完整实现）

        Args:
            motion: 包含 joint_pos 和 joint_vel 的字典
            isaac_joint_names: Isaac 期望的关节顺序

        Returns:
            重新映射后的 motion
        """
        if isaac_joint_names is None:
            return motion

        # 1. 找到共享关节（在两个列表中都存在的关节）
        share_joint_names = [name for name in self.joint_names if name in isaac_joint_names]
        src_joint_indices = [self.joint_names.index(name) for name in share_joint_names]
        dst_joint_indices = [isaac_joint_names.index(name) for name in share_joint_names]

        # 2. 找到额外关节（在 motion 中有但 isaac 中没有的）
        more_joint_names = [name for name in self.joint_names if name not in isaac_joint_names]
        src_more_joint_indices = [self.joint_names.index(name) for name in more_joint_names]
        dst_more_joint_indices = [len(isaac_joint_names) + i for i in range(len(more_joint_names))]

        # 3. 合并索引
        all_src_indices = src_joint_indices + src_more_joint_indices
        all_dst_indices = dst_joint_indices + dst_more_joint_indices
        new_joint_names = isaac_joint_names + more_joint_names

        # 4. 重新映射
        T = motion['joint_pos'].shape[0]
        new_joint_pos = np.zeros((T, len(new_joint_names)))
        new_joint_vel = np.zeros((T, len(new_joint_names)))

        new_joint_pos[:, all_dst_indices] = motion['joint_pos'][:, all_src_indices]
        new_joint_vel[:, all_dst_indices] = motion['joint_vel'][:, all_src_indices]

        motion['joint_pos'] = new_joint_pos
        motion['joint_vel'] = new_joint_vel

        return motion

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        time_steps: torch.Tensor,
        steps: Union[int, List[int], torch.Tensor]
    ) -> TwistMotionData:
        """获取多个 motion 的时间切片

        Args:
            motion_ids: [N] 要获取的 motion IDs
            time_steps: [N] 每个 motion 的起始时间步
            steps: int 或 List[int]，要获取的未来步数

        Returns:
            切片数据 TwistMotionData
        """
        # 确保输入在正确的设备上（可能来自不同设备）
        motion_ids = motion_ids.to(self.device)
        time_steps = time_steps.to(self.device)

        # 确保所有需要的 motion 都已加载
        unique_ids = torch.unique(motion_ids)
        missing_ids = [mid.item() for mid in unique_ids if mid.item() not in self.current_batch_data]

        if missing_ids:
            print(f"⚠️  Warning: {len(missing_ids)} motions not loaded!")
            print(f"    This should not happen. Call load_batch() first.")
            raise RuntimeError(f"{len(missing_ids)} motions not in current batch")

        # 准备存储结果
        N = len(motion_ids)

        if isinstance(steps, int):
            num_steps = steps
            step_indices = torch.arange(num_steps, device=self.device)
        elif isinstance(steps, (list, torch.Tensor)):
            num_steps = len(steps)
            if isinstance(steps, list):
                step_indices = torch.tensor(steps, device=self.device)
            else:
                step_indices = steps.to(self.device)
        else:
            raise ValueError(f"Invalid steps type: {type(steps)}")

        # 完全优化的批量处理：按 unique motion 分组处理
        # 转换到 CPU（一次性）
        time_steps_cpu = time_steps.cpu().numpy()
        motion_ids_cpu = motion_ids.cpu().numpy()

        if not isinstance(steps, int):
            step_indices_cpu = step_indices.cpu().numpy()

        # 预分配输出数组（直接使用 NumPy 预分配，避免 list append）
        # 获取数据维度（从第一个 motion 获取）
        first_motion_data = next(iter(self.current_batch_data.values()))
        num_bodies = first_motion_data['body_pos_w'].shape[1]
        num_joints = first_motion_data['joint_pos'].shape[1]

        body_pos_w_np = np.zeros((N, num_steps, num_bodies, 3), dtype=np.float32)
        body_quat_w_np = np.zeros((N, num_steps, num_bodies, 4), dtype=np.float32)
        body_lin_vel_w_np = np.zeros((N, num_steps, num_bodies, 3), dtype=np.float32)
        body_ang_vel_w_np = np.zeros((N, num_steps, num_bodies, 3), dtype=np.float32)
        joint_pos_np = np.zeros((N, num_steps, num_joints), dtype=np.float32)
        joint_vel_np = np.zeros((N, num_steps, num_joints), dtype=np.float32)

        # 按 unique motion ID 分组批量处理
        unique_motion_ids = np.unique(motion_ids_cpu)

        for unique_mid in unique_motion_ids:
            # 找到所有使用这个 motion 的环境索引
            env_mask = (motion_ids_cpu == unique_mid)
            env_indices = np.where(env_mask)[0]

            motion_data = self.current_batch_data[int(unique_mid)]
            motion_length = motion_data['body_pos_w'].shape[0]

            # 获取这些环境的 time_steps
            batch_time_steps = time_steps_cpu[env_mask]

            # 批量处理这些环境
            if isinstance(steps, int):
                # 简单情况：连续切片
                for i, env_idx in enumerate(env_indices):
                    t_val = batch_time_steps[i]
                    end_idx = min(t_val + steps, motion_length)
                    actual_steps = end_idx - t_val

                    body_pos_w_np[env_idx, :actual_steps] = motion_data['body_pos_w'][t_val:end_idx]
                    body_quat_w_np[env_idx, :actual_steps] = motion_data['body_quat_w'][t_val:end_idx]
                    body_lin_vel_w_np[env_idx, :actual_steps] = motion_data['body_lin_vel_w'][t_val:end_idx]
                    body_ang_vel_w_np[env_idx, :actual_steps] = motion_data['body_ang_vel_w'][t_val:end_idx]
                    joint_pos_np[env_idx, :actual_steps] = motion_data['joint_pos'][t_val:end_idx]
                    joint_vel_np[env_idx, :actual_steps] = motion_data['joint_vel'][t_val:end_idx]
            else:
                # 复杂情况：指定索引
                for i, env_idx in enumerate(env_indices):
                    t_val = batch_time_steps[i]
                    indices = np.clip(t_val + step_indices_cpu, 0, motion_length - 1)

                    body_pos_w_np[env_idx] = motion_data['body_pos_w'][indices]
                    body_quat_w_np[env_idx] = motion_data['body_quat_w'][indices]
                    body_lin_vel_w_np[env_idx] = motion_data['body_lin_vel_w'][indices]
                    body_ang_vel_w_np[env_idx] = motion_data['body_ang_vel_w'][indices]
                    joint_pos_np[env_idx] = motion_data['joint_pos'][indices]
                    joint_vel_np[env_idx] = motion_data['joint_vel'][indices]

        # 一次性转换到 GPU
        body_pos_w = torch.from_numpy(body_pos_w_np).to(self.device)
        body_quat_w = torch.from_numpy(body_quat_w_np).to(self.device)
        body_lin_vel_w = torch.from_numpy(body_lin_vel_w_np).to(self.device)
        body_ang_vel_w = torch.from_numpy(body_ang_vel_w_np).to(self.device)
        joint_pos = torch.from_numpy(joint_pos_np).to(self.device)
        joint_vel = torch.from_numpy(joint_vel_np).to(self.device)

        # 创建 motion_id 和 step 字段（与原始实现一致）
        # motion_id: [N, num_steps] 每帧对应的 motion ID
        # 使用 expand 创建的是视图，需要 clone 确保是连续内存且设备一致
        motion_id_tensor = motion_ids.unsqueeze(1).expand(-1, num_steps).clone().to(self.device, dtype=torch.long)

        # step: [N, num_steps] 每帧对应的时间步
        if isinstance(steps, int):
            step_tensor = torch.arange(num_steps, device=self.device, dtype=torch.long).unsqueeze(0).expand(N, -1).clone()
        else:
            step_tensor = step_indices.unsqueeze(0).expand(N, -1).clone().to(self.device, dtype=torch.long)

        # Debug: 打印所有张量的设备和形状（仅在第一次调用时）
        if not hasattr(self, '_debug_device_printed'):
            print(f"\n[OnDemandTwistMotionDataset.get_slice] Debug Info:")
            print(f"  Input: motion_ids.shape={motion_ids.shape}, time_steps.shape={time_steps.shape}")
            print(f"  Steps: {steps if isinstance(steps, int) else f'tensor/list with {num_steps} elements'}")
            print(f"  N={N}, num_steps={num_steps}")
            print(f"\n  Output tensor shapes:")
            print(f"    motion_id: {motion_id_tensor.shape} (device: {motion_id_tensor.device}, dtype: {motion_id_tensor.dtype})")
            print(f"    step: {step_tensor.shape} (device: {step_tensor.device}, dtype: {step_tensor.dtype})")
            print(f"    body_pos_w: {body_pos_w.shape} (device: {body_pos_w.device})")
            print(f"    body_quat_w: {body_quat_w.shape} (device: {body_quat_w.device})")
            print(f"    joint_pos: {joint_pos.shape} (device: {joint_pos.device})")
            print(f"  self.device: {self.device}")
            self._debug_device_printed = True

        # 创建 TwistMotionData（确保所有张量都在同一设备上，并指定 batch_size）
        result = TwistMotionData(
            motion_id=motion_id_tensor,
            step=step_tensor,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            batch_size=[N, num_steps],  # 明确指定 batch_size 为 [num_envs, num_steps]
        )

        # Debug: 打印 TensorClass 的 batch_size
        if not hasattr(self, '_debug_batch_printed'):
            print(f"\n  TwistMotionData batch_size: {result.batch_size}")
            print(f"  TwistMotionData shape: {result.shape}")
            self._debug_batch_printed = True

        return result

    def update_hnm(
        self,
        motion_ids: torch.Tensor,
        success_flags: torch.Tensor
    ):
        """更新 Hard Negative Mining 权重

        Args:
            motion_ids: [num_envs] 刚刚使用的 motion IDs
            success_flags: [num_envs] 是否成功完成（True/False）
        """
        if not self.enable_hnm:
            return

        # 统计每个 motion 的成功次数
        for motion_id, success in zip(motion_ids, success_flags):
            mid = motion_id.item()
            self.attempt_count[mid] += 1
            if success.item() if isinstance(success, torch.Tensor) else success:
                self.success_count[mid] += 1

        # 更新成功率
        self.success_rate = self.success_count / (self.attempt_count + 1e-8)

        # 更新权重
        for mid in range(self.num_motions):
            if self.attempt_count[mid] < 10:  # 样本数太少，不调整
                # 未采样的 motion，略微提升权重
                if self.attempt_count[mid] == 0:
                    self.motion_weights[mid] *= self.hnm_boost_unsampled
                continue

            sr = self.success_rate[mid]
            if sr > 0.95:  # 太简单
                self.motion_weights[mid] *= self.hnm_beta
            elif sr < 0.5:  # 太难
                self.motion_weights[mid] *= self.hnm_alpha

        # 应用最小权重
        self.motion_weights = torch.clamp(self.motion_weights, min=self.hnm_min_weight)

        # 归一化
        self.motion_weights = self.motion_weights / self.motion_weights.sum()

    def filter_impossible_motions(self):
        """过滤掉持续失败的 motion"""
        if not self.enable_hnm or not self.hnm_filter_enabled:
            return

        # 找出尝试次数足够且失败率过高的 motion
        enough_attempts = self.attempt_count >= self.hnm_min_attempts
        too_hard = self.success_rate < (1 - self.hnm_max_failure_rate)

        filtered = enough_attempts & too_hard
        num_filtered = filtered.sum().item()

        if num_filtered > 0:
            print(f"🚫 Filtering {num_filtered} impossible motions "
                  f"(success rate < {1-self.hnm_max_failure_rate:.1%})")
            self.filtered_mask[filtered] = False

            # 重新归一化权重
            valid_weights = self.motion_weights * self.filtered_mask.float()
            self.motion_weights = valid_weights / valid_weights.sum()

    def get_coverage_stats(self) -> Dict:
        """获取覆盖率统计

        Returns:
            Dict containing:
                - coverage: 被采样过的 motion 数量
                - coverage_rate: 覆盖率 (采样过的motion数 / 总motion数)
                - num_sampled: 被采样过的 motion 数量 (与 coverage 相同)
                - mean_attempts: 平均采样次数
                - max_attempts: 最大采样次数
                - min_attempts: 最小采样次数
                - num_filtered: 被过滤的 motion 数量
                - mean_weight: 平均采样权重
                - mean_success_rate: 平均成功率
        """
        if not self.enable_hnm:
            # 如果未启用 HNM，返回默认值
            return {
                'coverage': 0,
                'coverage_rate': 0.0,
                'num_sampled': 0,
                'mean_attempts': 0.0,
                'max_attempts': 0,
                'min_attempts': 0,
                'num_filtered': 0,
                'mean_weight': 1.0 / self.num_motions if self.num_motions > 0 else 0.0,
                'mean_success_rate': 0.0,
            }

        # 计算覆盖率统计
        sampled_mask = self.attempt_count > 0
        coverage = sampled_mask.sum().item()

        if coverage > 0:
            mean_attempts = self.attempt_count[sampled_mask].mean().item()
            max_attempts = self.attempt_count.max().item()
            min_attempts = self.attempt_count[sampled_mask].min().item()
        else:
            mean_attempts = max_attempts = min_attempts = 0.0

        # 计算平均权重（只考虑未过滤的 motion）
        valid_weights = self.motion_weights[self.filtered_mask]
        mean_weight = valid_weights.mean().item() if len(valid_weights) > 0 else 0.0

        # 计算平均成功率（只考虑被采样过的 motion）
        if coverage > 0:
            mean_success_rate = self.success_rate[sampled_mask].mean().item()
        else:
            mean_success_rate = 0.0

        return {
            'coverage': coverage,
            'coverage_rate': coverage / self.num_motions,
            'num_sampled': coverage,  # num_sampled 与 coverage 相同
            'mean_attempts': mean_attempts,
            'max_attempts': max_attempts,
            'min_attempts': min_attempts,
            'num_filtered': (~self.filtered_mask).sum().item(),
            'mean_weight': mean_weight,
            'mean_success_rate': mean_success_rate,
        }

    @classmethod
    def create_from_yaml(
        cls,
        yaml_path: str,
        device: str = "cuda",
        **kwargs
    ):
        """从 YAML 配置文件创建数据集

        Args:
            yaml_path: YAML 配置文件路径
            device: 设备
            **kwargs: 其他参数传递给 __init__

        Returns:
            OnDemandTwistMotionDataset 实例
        """
        print(f"📄 Loading dataset from YAML: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # 获取 root_path 和 motions 列表
        yaml_root = Path(config.get('root_path', Path(yaml_path).parent))
        motions_config = config.get('motions', [])

        print(f"   Dataset root: {yaml_root}")
        print(f"   Total motions in YAML: {len(motions_config)}")

        # 扫描文件
        motion_paths = []
        for motion_entry in tqdm(motions_config, desc="Scanning files"):
            file_path = motion_entry.get('file')
            if not file_path:
                continue

            full_path = yaml_root / file_path
            if full_path.exists():
                motion_paths.append(full_path)

        print(f"✅ Found {len(motion_paths)} valid motion files")

        if not motion_paths:
            raise RuntimeError(f"No valid motion files found in {yaml_path}")
        import sys
        if not hasattr(np, '_core'):
            import numpy.core as _core_module
            np._core = _core_module
            sys.modules['numpy._core'] = _core_module
            sys.modules['numpy._core.multiarray'] = sys.modules['numpy.core.multiarray']
        # 从第一个文件获取 body_names 和 joint_names
        with open(motion_paths[0], 'rb') as f:
            sample_data = pickle.load(f)
            if 'body_names' in sample_data:
                body_names = sample_data['body_names']
                joint_names = sample_data['joint_names']
            else:
                # 使用默认名称
                from .twist_motion import unitree_body_names, unitree_joint_names
                body_names = unitree_body_names
                joint_names = unitree_joint_names

        return cls(
            motion_paths=motion_paths,
            body_names=body_names,
            joint_names=joint_names,
            device=device,
            **kwargs
        )

    def to(self, device: str):
        """移动到指定设备"""
        self.device = device
        # 注意：当前 batch 数据在 load_batch 时会加载到新设备
        return self
