"""
TwistMotionTracking with Hard Negative Mining

Based on the HNM strategy from motion imitation research:
"When training on large datasets, the motion imitation policy may converge to an average point,
thereby hindering full coverage of the whole dataset. To address this issue, we employ the strategy
of hard negative mining by periodically evaluating our policy over the entire dataset and
dynamically adjusting the sampling probability for each motion sample."

Key difference from Curriculum Learning:
- Curriculum: Easy → Hard (淘汰困难样本)
- HNM: Focus on Hard samples (专注攻克困难样本)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Union

from tensordict import TensorDict
from torchrl.envs.transforms import CatTensors

from active_adaptation.envs.mdp.commands.twist.command import TwistMotionTracking
from active_adaptation.envs.mdp.commands.twist.hnm_strategy import HardNegativeMining

# Import common components
from active_adaptation.envs.mdp.commands.twist.command import (
    OBS_KEY, OBS_PRIV_KEY, ACTION_KEY, REWARD_KEY, TERM_KEY, DONE_KEY,
    make_mlp, GAE, Actor, normalize, make_batch
)


class TwistMotionTrackingHNM(TwistMotionTracking):
    """TWIST Motion Tracking with Hard Negative Mining"""

    def __init__(
        self,
        env,
        action_scale: Union[float, List[float]],
        reset_range: Dict[str, List[float]],
        tracking_joint_names: List[str] = None,
        lookat: List[float] = None,
        eye: List[float] = None,
        seed: int = 0,
        init_joint_pos_noise: float = 0.0,
        init_joint_vel_noise: float = 0.0,
        # observation parameters
        future_steps: List[int] = [1, 2, 8, 16],
        # motion curriculum parameters (DISABLED for HNM)
        motion_curriculum: bool = False,  # Disabled, conflicts with HNM
        motion_curriculum_gamma: float = 0.01,
        # on-demand loading parameters
        lazy_loading: bool = False,
        motion_pool_size: int = 0,
        motion_pool_resample_interval: int = 1500,
        # Hard Negative Mining parameters (MAIN FEATURE)
        enable_hnm: bool = True,
        hnm_strategy: str = "success_rate",
        hnm_alpha: float = 1.5,          # 失败motion的采样概率乘数 (>1)
        hnm_beta: float = 0.7,            # 成功motion的采样概率乘数 (<1)
        hnm_min_weight: float = 1e-6,     # 最小采样权重
        hnm_eval_interval: int = 1000,   # 评估间隔
        hnm_gamma: float = 0.01,         # HNM权重调整速率
        hnm_filter_enabled: bool = True,
        hnm_min_attempts: int = 20,
        hnm_max_failure_rate: float = 0.1,
        hnm_boost_unsampled: float = 1.05,
    ):

        # Disable motion_curriculum when HNM is enabled (they conflict)
        if enable_hnm and motion_curriculum:
            print("⚠️  Warning: HNM and motion_curriculum conflict. Disabling motion_curriculum.")
            motion_curriculum = False

        # Call parent constructor (will create motion dataset)
        super().__init__(
            env=env,
            action_scale=action_scale,
            reset_range=reset_range,
            tracking_joint_names=tracking_joint_names,
            lookat=lookat,
            eye=eye,
            seed=seed,
            init_joint_pos_noise=init_joint_pos_noise,
            init_joint_vel_noise=init_joint_vel_noise,
            future_steps=future_steps,
            motion_curriculum=motion_curriculum,  # Disabled
            motion_curriculum_gamma=motion_curriculum_gamma,
            lazy_loading=lazy_loading,
            motion_pool_size=motion_pool_size,
            motion_pool_resample_interval=motion_pool_resample_interval,
            enable_hnm=False,  # We'll create our own HNM
            hnm_alpha=hnm_alpha,
            hnm_beta=hnm_beta,
            hnm_min_weight=hnm_min_weight,
            hnm_boost_unsampled=hnm_boost_unsampled,
            hnm_filter_enabled=hnm_filter_enabled,
        )

        # HNM Configuration
        self.enable_hnm = enable_hnm
        self.hnm_strategy = hnm_strategy
        self.hnm_eval_interval = hnm_eval_interval
        self.step_counter = 0

        # Initialize HNM
        if self.enable_hnm:
            print(f"🎯 Initializing Hard Negative Mining")
            print(f"   Strategy: {hnm_strategy}")
            print(f"   Alpha (failure boost): {hnm_alpha}")
            print(f"   Beta (success reduce): {hnm_beta}")
            print(f"   Min weight: {hnm_min_weight}")
            print(f"   Eval interval: {hnm_eval_interval} steps")
            print(f"   Filter enabled: {hnm_filter_enabled}")

            self.hnm = HardNegativeMining(
                num_motions=self.dataset.num_motions,
                device=self.device,
                alpha=hnm_alpha,
                beta=hnm_beta,
                min_weight=hnm_min_weight,
                hnm_enabled=True,
                hnm_gamma=hnm_gamma,
                eval_interval=hnm_eval_interval,
                filter_threshold=hnm_max_failure_rate,
                filter_enabled=hnm_filter_enabled,
            )

            # Episode buffer for tracking success/failure
            self.episode_length_buf = torch.zeros(self.num_envs, device=self.device)
            self.episode_success_buf = torch.zeros(self.num_envs, device=self.device)
        else:
            self.hnm = None

    def _sample_motions(self, env_ids: torch.Tensor):
        """
        采样motion IDs，使用HNM策略

        HNM逻辑：
        - 失败的motion → 增加采样概率 (alpha > 1)
        - 成功的motion → 降低采样概率 (beta < 1)
        - 专注攻克困难样本，避免收敛到平均点
        """
        if self.sample_motion or self.first_sample_motion:

            if self.enable_hnm and self.hnm is not None:
                # ==================== Hard Negative Mining Sampling ====================
                print(f"🎯 HNM Sampling: {len(env_ids)} motions")

                # 使用HNM权重采样
                motion_ids = self.hnm.sample_motions(len(env_ids))

                # 更新采样统计
                hnm_stats = self.hnm.get_hnm_stats()
                if hnm_stats:
                    print(f"   Mean success rate: {hnm_stats.get('hnm/mean_success_rate', 0):.3f}")
                    print(f"   Attempted motions: {hnm_stats.get('hnm/attempted_motions', 0)}/{self.dataset.num_motions}")
                    print(f"   Filtered motions: {hnm_stats.get('hnm/filtered_motions', 0)}")

            elif self.motion_curriculum:
                # ==================== TWIST-Aligned Curriculum Sampling ====================
                # Note: This should not happen when HNM is enabled
                print(f"⚠️  Using Curriculum Learning (HNM disabled)")

                # 计算当前的最大难度（基于训练进度）
                mean_difficulty = self.motion_difficulty.mean().item()
                self.mean_motion_difficulty = mean_difficulty

                # 过滤出难度 <= mean_difficulty 的运动
                valid_mask = (self.motion_difficulty <= mean_difficulty).float()

                if valid_mask.sum() > 0:
                    weights = valid_mask / valid_mask.sum()
                    motion_ids = torch.multinomial(
                        weights,
                        num_samples=len(env_ids),
                        replacement=True
                    )
                else:
                    print(f"[WARNING] No motions with difficulty <= {mean_difficulty:.2f}, falling back to random sampling")
                    motion_ids = torch.randint(0, self.dataset.num_motions, size=(len(env_ids),), device=self.device)
            else:
                # 原始随机采样（无课程学习，无HNM）
                motion_ids = torch.randint(0, self.dataset.num_motions, size=(len(env_ids),), device=self.device)

            self.motion_ids[env_ids] = motion_ids

            # 获取 motion lengths 和 starts/ends
            if self.lazy_loading:
                self.motion_len[env_ids] = motion_len = self.dataset.motion_lengths[motion_ids]
                self.motion_starts[env_ids] = self.dataset.starts[motion_ids]
                self.motion_ends[env_ids] = self.dataset.ends[motion_ids]
            else:
                self.motion_len[env_ids] = motion_len = self.dataset.lengths[motion_ids]
                self.motion_starts[env_ids] = self.dataset.starts[motion_ids]
                self.motion_ends[env_ids] = self.dataset.ends[motion_ids]

            # 采样开始时间步
            self.t[env_ids] = torch.randint(
                self.motion_starts[env_ids],
                self.motion_ends[env_ids],
                (len(env_ids),),
                device=self.device
            ).float()

    def sample_init(self, env_ids):
        """
        初始化环境状态，支持HNM统计更新
        """
        # ==================== Hard Negative Mining ====================
        # 在初始化新episode之前，更新HNM统计
        if self.enable_hnm and self.hnm is not None and len(env_ids) > 0:
            self._update_hnm_stats(env_ids)

            # 重置episode buffer
            self.episode_length_buf[env_ids] = 0.0
            self.episode_success_buf[env_ids] = 0.0
        # ===============================================================

        # 采样新motion
        self._sample_motions(env_ids)

        # 从运动数据中获取重置状态
        self._motion_reset: TensorDict = self.dataset.get_slice(
            self.motion_ids[env_ids],
            self.t[env_ids].long(),
            1
        ).squeeze(1)

        # 应用reset_range偏移
        base_state = self._motion_reset["state"]
        reset_noise = {}
        for key, (min_val, max_val) in self.reset_range.items():
            noise = (torch.rand(len(env_ids), 3, device=self.device) - 0.5) * (max_val - min_val)
            reset_noise[key] = noise

        # 应用噪声到base state
        for key, noise in reset_noise.items():
            if key in base_state:
                if key == "root_pos_w":
                    base_state[key] += noise
                elif key == "root_quat_w":
                    # 对于四元数，使用更小的噪声并重新归一化
                    quat_noise = torch.randn(len(env_ids), 4, device=self.device) * 0.01
                    base_state[key] = self._add_quat_noise(base_state[key], quat_noise)
                elif key == "joint_pos":
                    base_state[key] += noise[:, :base_state[key].shape[-1]] * self.init_joint_pos_noise
                elif key == "joint_vel":
                    base_state[key] += noise[:, :base_state[key].shape[-1]] * self.init_joint_vel_noise

        # 设置时间步
        self.t[env_ids] = self.t[env_ids] + 1

        # 获取初始奖励和done
        with torch.no_grad():
            initial_state = base_state
            initial_reward = torch.zeros(len(env_ids), 1, device=self.device)
            initial_done = torch.zeros(len(env_ids), 1, dtype=torch.bool, device=self.device)

        # 创建重置tensordict
        reset_tensordict = TensorDict({
            "policy": initial_state["dof_pos"],
            "priv": torch.cat([
                initial_state["root_pos_w"],
                initial_state["root_quat_w"],
                initial_state["dof_pos"],
                initial_state["dof_vel"],
                initial_state["root_vel_w"],
                initial_state["ang_vel_w"],
            ], dim=-1),
            "next_policy": initial_state["dof_pos"],
            "next_priv": torch.cat([
                initial_state["root_pos_w"],
                initial_state["root_quat_w"],
                initial_state["dof_pos"],
                initial_state["dof_vel"],
                initial_state["root_vel_w"],
                initial_state["ang_vel_w"],
            ], dim=-1),
            "action": torch.zeros(len(env_ids), self.action_dim, device=self.device),
            "sample_log_prob": torch.zeros(len(env_ids), self.action_dim, device=self.device),
            "next": torch.ones(len(env_ids), dtype=torch.bool, device=self.device),
            "step": torch.zeros(len(env_ids), 1, device=self.device),
            "is_init": torch.ones(len(env_ids), 1, dtype=torch.bool, device=self.device),
            "episode_length": self.episode_length_buf[env_ids].unsqueeze(1),
            "success": self.episode_success_buf[env_ids].unsqueeze(1),
        }, batch_size=[len(env_ids)])

        reset_tensordict.set(
            REWARD_KEY,
            initial_reward,
        )
        reset_tensordict.set(
            TERM_KEY,
            initial_done,
        )
        reset_tensordict.set(
            DONE_KEY,
            initial_done,
        )
        reset_tensordict.set(
            "discount",
            torch.ones_like(initial_reward),
        )

        # 记录当前motion ID和时间步
        reset_tensordict.set("motion_id", self.motion_ids[env_ids].unsqueeze(1))
        reset_tensordict.set("t", self.t[env_ids].unsqueeze(1))

        return reset_tensordict

    def _update_hnm_stats(self, env_ids: torch.Tensor):
        """
        更新HNM统计信息

        Args:
            env_ids: 需要更新的环境ID
        """
        if not self.enable_hnm or self.hnm is None:
            return

        # 计算成功标志：完成率 >= 95% 为成功
        success_flags = (self.episode_length_buf[env_ids] >= 9.0)  # 10秒episode，9.5秒算成功

        # 获取对应的motion IDs
        motion_ids = self.motion_ids[env_ids]

        # 更新HNM统计
        hnm_stats = self.hnm.update_hnm_stats(motion_ids, success_flags)

        # 记录到环境extra中
        if hasattr(self.env, 'extra') and hnm_stats:
            for key, value in hnm_stats.items():
                self.env.extra[f"hnm/{key.split('/')[-1]}"] = value

    def update(self):
        """
        更新运动跟踪状态
        """
        # 推进时间步
        self.t += 1

        # 累计episode长度
        if self.enable_hnm and hasattr(self, 'episode_length_buf'):
            self.episode_length_buf += self.env.step_dt

        # HNM step counter
        if self.enable_hnm:
            self.step_counter += 1

        # 记录统计信息
        if hasattr(self.env, 'extra'):
            if self.enable_hnm:
                hnm_stats = self.hnm.get_hnm_stats()
                if hnm_stats:
                    for key, value in hnm_stats.items():
                        self.env.extra[f"hnm/{key.split('/')[-1]}"] = value

            # 记录motion pool状态（如果使用lazy_loading）
            if self.lazy_loading and hasattr(self, 'motion_pool_step_counter'):
                self.env.extra['motion_pool/step_counter'] = self.motion_pool_step_counter
                self.env.extra['motion_pool/cache_size'] = len(getattr(self.dataset, 'current_batch_data', {}))

    def get_coverage_stats(self):
        """获取覆盖率统计（兼容OnDemandTwistMotionDataset）"""
        if hasattr(self.dataset, 'get_coverage_stats'):
            return self.dataset.get_coverage_stats()
        else:
            # 原版TwistMotionDataset的简单统计
            return {
                'coverage': self.dataset.num_motions,
                'coverage_rate': 1.0,
                'num_sampled': self.dataset.num_motions,
                'mean_attempts': 1.0,
                'max_attempts': 1.0,
                'min_attempts': 1.0,
                'num_filtered': 0,
                'mean_weight': 1.0 / self.dataset.num_motions,
                'mean_success_rate': 0.5,
            }

    def _add_quat_noise(self, quat, noise):
        """为四元数添加噪声并重新归一化"""
        # 将噪声转换为四元数旋转
        noise_quat = torch.cat([
            torch.zeros_like(noise[:, :1]),  # w分量无噪声
            noise[:, 1:]  # xyz分量有噪声
        ], dim=-1)

        # 四元数乘法（应用噪声旋转）
        result_quat = self._quat_multiply(noise_quat, quat)

        # 重新归一化
        result_quat = result_quat / torch.norm(result_quat, dim=-1, keepdim=True)

        return result_quat

    def _quat_multiply(self, q1, q2):
        """四元数乘法"""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)