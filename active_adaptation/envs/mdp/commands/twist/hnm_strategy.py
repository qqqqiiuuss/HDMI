"""
Hard Negative Mining Strategy for Motion Imitation

Reference: "When training on large datasets, the motion imitation policy may converge to an average point,
thereby hindering full coverage of the whole dataset. To address this issue, we employ the strategy of
hard negative mining by periodically evaluating our policy over the entire dataset and dynamically
adjusting the sampling probability for each motion sample. If the policy fails to track a particular
sample, its sampling probability is increased by a predefined factor, whereas successful tracking
leads to a corresponding decrease."

Implementation based on the paper's HNM approach.
"""

import torch
from typing import Dict, Tuple


class HardNegativeMining:
    """Hard Negative Mining for motion imitation learning"""

    def __init__(
        self,
        num_motions: int,
        device: torch.device,
        alpha: float = 1.5,           # 增加失败motion的采样概率
        beta: float = 0.7,            # 降低成功motion的采样概率
        min_weight: float = 1e-6,     # 最小采样权重
        hnm_enabled: bool = True,
        hnm_gamma: float = 0.01,      # HNM调整速率
        eval_interval: int = 1000,    # 评估间隔（步数）
        filter_threshold: float = 0.2, # 过滤阈值（持续失败率）
        filter_enabled: bool = True,
    ):
        """
        Args:
            num_motions: 总motion数量
            device: 计算设备
            alpha: 失败motion的采样概率乘数 (>1)
            beta: 成功motion的采样概率乘数 (<1)
            min_weight: 最小采样权重，防止某些motion永远不被采样
            hnm_enabled: 是否启用HNM
            hnm_gamma: HNM权重调整速率
            eval_interval: 多少步评估一次policy表现
            filter_threshold: 过滤持续失败的motion阈值
            filter_enabled: 是否启用过滤机制
        """
        self.num_motions = num_motions
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.min_weight = min_weight
        self.hnm_enabled = hnm_enabled
        self.hnm_gamma = hnm_gamma
        self.eval_interval = eval_interval
        self.filter_threshold = filter_threshold
        self.filter_enabled = filter_enabled

        # HNM统计
        self.success_count = torch.zeros(num_motions, device=device)
        self.attempt_count = torch.zeros(num_motions, device=device)
        self.success_rate = torch.zeros(num_motions, device=device)
        self.filtered_mask = torch.ones(num_motions, dtype=torch.bool, device=device)

        # 采样权重（初始化为均匀分布）
        self.sampling_weights = torch.ones(num_motions, device=device) / num_motions

        # 评估计数器
        self.step_counter = 0
        self.last_eval_step = 0

    def update_hnm_stats(
        self,
        motion_ids: torch.Tensor,
        success_flags: torch.Tensor
    ) -> Dict:
        """
        更新HNM统计信息

        Args:
            motion_ids: [batch_size] 当前使用的motion IDs
            success_flags: [batch_size] 是否成功完成 (True/False)

        Returns:
            Dict: HNM统计信息
        """
        if not self.hnm_enabled:
            return {}

        # 统计每个motion的成功次数和尝试次数
        for motion_id, success in zip(motion_ids, success_flags):
            mid = motion_id.item()
            self.attempt_count[mid] += 1
            if success.item() if isinstance(success, torch.Tensor) else success:
                self.success_count[mid] += 1

        # 更新成功率
        self.success_rate = self.success_count / (self.attempt_count + 1e-8)

        # 每 eval_interval 步更新采样权重
        self.step_counter += 1
        if self.step_counter - self.last_eval_step >= self.eval_interval:
            self._update_sampling_weights()
            self.last_eval_step = self.step_counter

        return self.get_hnm_stats()

    def _update_sampling_weights(self):
        """更新采样权重（HNM核心逻辑）"""
        if not self.hnm_enabled:
            return

        # 根据成功率调整权重
        for mid in range(self.num_motions):
            if self.attempt_count[mid] < 10:  # 样本数太少，不调整
                # 未采样的motion，略微提升权重
                if self.attempt_count[mid] == 0:
                    self.sampling_weights[mid] *= 1.1  # 稍微增加权重
                continue

            sr = self.success_rate[mid]

            # HNM逻辑：失败增加权重，成功降低权重
            if sr < 0.3:  # 严重失败
                self.sampling_weights[mid] *= self.alpha
            elif sr < 0.5:  # 一般失败
                self.sampling_weights[mid] *= (self.alpha + 1) / 2
            elif sr > 0.95:  # 非常成功
                self.sampling_weights[mid] *= self.beta
            elif sr > 0.8:  # 一般成功
                self.sampling_weights[mid] *= (self.beta + 1) / 2

        # 应用最小权重约束
        self.sampling_weights = torch.clamp(self.sampling_weights, min=self.min_weight)

        # 归一化权重
        if self.filtered_mask.sum() > 0:
            valid_weights = self.sampling_weights[self.filtered_mask]
            self.sampling_weights[self.filtered_mask] = valid_weights / valid_weights.sum()

        # 过滤持续失败的motion
        if self.filter_enabled:
            self._filter_impossible_motions()

    def _filter_impossible_motions(self):
        """过滤掉持续失败的motion"""
        # 尝试次数 >= 20 且 成功率 < 10% 的motion
        persistent_fail_mask = (self.attempt_count >= 20) & (self.success_rate < 0.1)

        if persistent_fail_mask.sum() > 0:
            print(f"🔍 HNM Filter: {persistent_fail_mask.sum()} motions marked as impossible")
            print(f"   Attempts: {self.attempt_count[persistent_fail_mask].mean().item():.1f}")
            print(f"   Success rate: {self.success_rate[persistent_fail_mask].mean().item():.3f}")

            # 设置为不可采样
            self.filtered_mask[persistent_fail_mask] = False
            self.sampling_weights[persistent_fail_mask] = 0.0

    def sample_motions(self, num_samples: int) -> torch.Tensor:
        """
        根据HNM权重采样motion IDs

        Args:
            num_samples: 采样数量

        Returns:
            torch.Tensor: [num_samples] 采样的motion IDs
        """
        if not self.hnm_enabled:
            # 随机采样
            return torch.randint(0, self.num_motions, (num_samples,), device=self.device)

        # 只从未过滤的motions中采样
        valid_weights = self.sampling_weights[self.filtered_mask]
        valid_motion_ids = torch.where(self.filtered_mask)[0]

        if len(valid_motion_ids) == 0:
            print("⚠️  Warning: No valid motions available for HNM sampling")
            return torch.randint(0, self.num_motions, (num_samples,), device=self.device)

        # 根据权重采样
        sampled_indices = torch.multinomial(valid_weights, num_samples, replacement=True)
        motion_ids = valid_motion_ids[sampled_indices]

        return motion_ids

    def get_hnm_stats(self) -> Dict:
        """获取HNM统计信息"""
        if not self.hnm_enabled:
            return {}

        attempted_motions = self.attempt_count > 0
        if attempted_motions.sum() > 0:
            avg_attempts = self.attempt_count[attempted_motions].mean().item()
            max_attempts = self.attempt_count.max().item()
            min_attempts = self.attempt_count[attempted_motions].min().item()
        else:
            avg_attempts = max_attempts = min_attempts = 0.0

        return {
            'hnm/attempted_motions': attempted_motions.sum().item(),
            'hnm/avg_attempts': avg_attempts,
            'hnm/max_attempts': max_attempts,
            'hnm/min_attempts': min_attempts,
            'hnm/filtered_motions': (~self.filtered_mask).sum().item(),
            'hnm/mean_weight': self.sampling_weights[self.filtered_mask].mean().item() if self.filtered_mask.sum() > 0 else 0.0,
            'hnm/mean_success_rate': self.success_rate[attempted_motions].mean().item() if attempted_motions.sum() > 0 else 0.0,
            'hnm/eval_interval': self.eval_interval,
            'hnm/step_counter': self.step_counter,
        }

    def should_eval(self) -> bool:
        """检查是否到了评估时间"""
        return self.hnm_enabled and (self.step_counter - self.last_eval_step >= self.eval_interval)

    def reset_stats(self):
        """重置统计信息（用于新的训练session）"""
        self.success_count.zero_()
        self.attempt_count.zero_()
        self.success_rate.zero_()
        self.filtered_mask.fill_(True)
        self.sampling_weights.fill_(1.0 / self.num_motions)
        self.step_counter = 0
        self.last_eval_step = 0