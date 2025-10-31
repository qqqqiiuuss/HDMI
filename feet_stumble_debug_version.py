"""
调试版本的 feet_stumble_twist reward

添加详细的打印信息来诊断为什么 reward 一直是 0
"""

import torch
from typing import List
from active_adaptation.envs.mdp.commands.twist.rewards_new import TrackReward

class feet_stumble_twist_debug(TrackReward):
    """
    调试版本：带详细日志的 feet_stumble_twist reward
    """
    def __init__(self, threshold: float = 1.0, body_names: List[str] | str = ".*ankle_roll_link", **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.debug_counter = 0
        self.debug_interval = 100  # 每100步打印一次

        # Get contact sensor
        from isaaclab.sensors import ContactSensor
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        # 调试：打印 sensor 信息
        print("\n" + "=" * 80)
        print("feet_stumble_twist_debug 初始化")
        print("=" * 80)
        print(f"Contact Sensor 类型: {type(self.contact_sensor)}")
        print(f"可用的 body names: {self.contact_sensor.body_names}")
        print(f"总共 {len(self.contact_sensor.body_names)} 个 bodies")

        self.contact_body_indices, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.contact_body_indices = torch.tensor(self.contact_body_indices, device=self.device, dtype=torch.long)

        print(f"\n搜索模式: '{body_names}'")
        print(f"找到的 indices: {self.contact_body_indices.tolist()}")
        print(f"匹配的 body names: {self.body_names}")

        if len(self.contact_body_indices) == 0:
            print("\n❌ 警告: 没有找到匹配的 bodies!")
        else:
            print(f"\n✓ 成功找到 {len(self.contact_body_indices)} 个匹配的 bodies")
        print("=" * 80 + "\n")

    def compute(self):
        # Get contact forces from contact sensor
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.contact_body_indices]

        # TWIST implementation: check if XY force magnitude > 4 * |Z force|
        xy_force_norm = contact_forces[..., :2].norm(dim=-1)
        z_force_abs = contact_forces[..., 2].abs()

        # stumble = any(||F_xy|| > 4 * |F_z|)
        stumble_mask = xy_force_norm > 4.0 * z_force_abs
        stumble = stumble_mask.any(dim=-1).float()

        # 调试输出（每N步一次）
        self.debug_counter += 1
        if self.debug_counter % self.debug_interval == 0:
            print(f"\n--- feet_stumble_twist Debug (Step {self.debug_counter}) ---")
            print(f"Contact forces shape: {contact_forces.shape}")
            print(f"Contact forces 统计:")
            print(f"  - 非零元素数量: {(contact_forces.abs() > 0.01).sum().item()} / {contact_forces.numel()}")
            print(f"  - 最小值: {contact_forces.min().item():.4f}")
            print(f"  - 最大值: {contact_forces.max().item():.4f}")
            print(f"  - 平均绝对值: {contact_forces.abs().mean().item():.4f}")

            print(f"\nXY force norm:")
            print(f"  - 平均: {xy_force_norm.mean().item():.4f}")
            print(f"  - 最大: {xy_force_norm.max().item():.4f}")

            print(f"\nZ force abs:")
            print(f"  - 平均: {z_force_abs.mean().item():.4f}")
            print(f"  - 最大: {z_force_abs.max().item():.4f}")

            # 计算力比率
            valid_z = z_force_abs > 0.1
            if valid_z.any():
                ratios = torch.where(valid_z, xy_force_norm / (z_force_abs + 1e-8), torch.zeros_like(xy_force_norm))
                print(f"\n力比率 (XY/Z) [只统计 Z > 0.1 的情况]:")
                print(f"  - 平均: {ratios[valid_z].mean().item():.4f}")
                print(f"  - 最大: {ratios.max().item():.4f}")
                print(f"  - 阈值: 4.0")
            else:
                print(f"\n⚠️  所有 Z 力都 < 0.1 (可能机器人在空中或没有接触)")

            print(f"\nStumble 检测:")
            print(f"  - 触发的 envs: {stumble.sum().item()} / {stumble.shape[0]}")
            print(f"  - Stumble mask shape: {stumble_mask.shape}")
            print(f"  - 触发的 (env, foot) 对数: {stumble_mask.sum().item()}")

            if stumble.sum().item() > 0:
                print(f"\n✓ 检测到 stumble!")
            else:
                print(f"\n  (未检测到 stumble)")
            print("-" * 60)

        return stumble.unsqueeze(1)


# 使用说明
USAGE = """
使用调试版本的 feet_stumble_twist:

1. 在配置文件中修改 reward 类型:

reward:
  regularization:
    feet_stumble_twist:
      _target_: feet_stumble_debug_version.feet_stumble_twist_debug  # 使用调试版本
      weight: -1.25
      enabled: true
      threshold: 1.0
      body_names: ".*ankle_roll_link"

2. 运行训练并查看输出

3. 检查输出中的：
   - Contact forces 是否全是 0
   - XY/Z 力比率是否总是小于 4.0
   - Body names 是否正确匹配

4. 根据输出结果确定问题所在
"""

if __name__ == "__main__":
    print(USAGE)
