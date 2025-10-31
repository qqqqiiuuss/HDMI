"""
诊断脚本：检查 feet_stumble_twist reward 为什么一直是 0

这个脚本会：
1. 检查 contact sensor 是否正确初始化
2. 检查 body indices 是否正确
3. 打印 contact forces 的实际值
4. 计算并显示 stumble 条件
"""

import torch
import re

def diagnose_feet_stumble_reward(env, step_count=100):
    """
    诊断 feet_stumble_twist reward

    Args:
        env: 环境实例
        step_count: 要检查的步数
    """
    print("=" * 80)
    print("开始诊断 feet_stumble_twist reward")
    print("=" * 80)

    # 1. 检查 contact sensor
    print("\n[1] 检查 Contact Sensor")
    print("-" * 80)

    if "contact_forces" not in env.scene:
        print("❌ 错误: scene 中没有 'contact_forces' sensor!")
        return

    contact_sensor = env.scene["contact_forces"]
    print(f"✓ Contact sensor 找到")
    print(f"  - Sensor 类型: {type(contact_sensor)}")
    print(f"  - Body names: {contact_sensor.body_names}")

    # 2. 检查 body indices 匹配
    print("\n[2] 检查 Body Indices 匹配")
    print("-" * 80)

    body_pattern = ".*ankle_roll_link"
    body_indices, matched_names = contact_sensor.find_bodies(body_pattern)

    print(f"  - 搜索模式: '{body_pattern}'")
    print(f"  - 找到的 indices: {body_indices}")
    print(f"  - 匹配的 body names: {matched_names}")

    if len(body_indices) == 0:
        print("❌ 错误: 没有找到匹配的 body!")
        print(f"  可用的 body names: {contact_sensor.body_names}")
        return

    contact_body_indices = torch.tensor(body_indices, device=env.device, dtype=torch.long)

    # 3. 运行环境并收集数据
    print("\n[3] 收集 Contact Forces 数据")
    print("-" * 80)

    stumble_stats = {
        'total_steps': 0,
        'stumble_detected': 0,
        'max_xy_force': 0.0,
        'max_z_force': 0.0,
        'max_ratio': 0.0,
        'zero_force_steps': 0
    }

    print(f"运行 {step_count} 步并监控 contact forces...")

    for step in range(step_count):
        # 执行一步
        actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        env.step(actions)

        # 获取 contact forces
        contact_forces = contact_sensor.data.net_forces_w[:, contact_body_indices]

        # 计算 XY 和 Z 方向的力
        xy_force_norm = contact_forces[..., :2].norm(dim=-1)
        z_force_abs = contact_forces[..., 2].abs()

        # 检查是否满足 stumble 条件
        stumble_mask = xy_force_norm > 4.0 * z_force_abs
        stumble = stumble_mask.any(dim=-1)

        # 统计
        stumble_stats['total_steps'] += 1
        stumble_stats['stumble_detected'] += stumble.sum().item()
        stumble_stats['max_xy_force'] = max(stumble_stats['max_xy_force'], xy_force_norm.max().item())
        stumble_stats['max_z_force'] = max(stumble_stats['max_z_force'], z_force_abs.max().item())

        # 计算力比率（避免除零）
        valid_z = z_force_abs > 0.1
        if valid_z.any():
            ratios = torch.where(valid_z, xy_force_norm / (z_force_abs + 1e-8), torch.zeros_like(xy_force_norm))
            stumble_stats['max_ratio'] = max(stumble_stats['max_ratio'], ratios.max().item())

        # 检查是否所有力都是零
        if contact_forces.abs().sum() < 1e-6:
            stumble_stats['zero_force_steps'] += 1

        # 每20步打印一次详细信息
        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{step_count}:")
            print(f"    - 平均 XY force: {xy_force_norm.mean():.4f}")
            print(f"    - 平均 Z force: {z_force_abs.mean():.4f}")
            print(f"    - Stumble 检测: {stumble.sum().item()} / {env.num_envs} envs")

    # 4. 打印统计结果
    print("\n[4] 统计结果")
    print("-" * 80)
    print(f"  - 总步数: {stumble_stats['total_steps']}")
    print(f"  - 检测到 stumble 的次数: {stumble_stats['stumble_detected']}")
    print(f"  - 最大 XY force: {stumble_stats['max_xy_force']:.4f}")
    print(f"  - 最大 Z force: {stumble_stats['max_z_force']:.4f}")
    print(f"  - 最大力比率 (XY/Z): {stumble_stats['max_ratio']:.4f}")
    print(f"  - 零力步数: {stumble_stats['zero_force_steps']} / {stumble_stats['total_steps']}")

    # 5. 诊断结论
    print("\n[5] 诊断结论")
    print("=" * 80)

    if stumble_stats['zero_force_steps'] == stumble_stats['total_steps']:
        print("❌ 问题: Contact forces 一直是 0!")
        print("   可能原因:")
        print("   1. Contact sensor 没有正确更新")
        print("   2. Sensor 配置的 prim_path 不正确")
        print("   3. 机器人没有与地面接触")

    elif stumble_stats['stumble_detected'] == 0:
        if stumble_stats['max_ratio'] < 2.0:
            print("✓ 正常: 机器人步态非常稳定，XY/Z 力比率 < 2.0")
            print(f"   最大比率: {stumble_stats['max_ratio']:.4f} (阈值: 4.0)")
        else:
            print("⚠️  警告: 有较大的力比率但未达到阈值")
            print(f"   最大比率: {stumble_stats['max_ratio']:.4f} (阈值: 4.0)")
            print("   建议: 检查阈值是否过高")
    else:
        print(f"✓ 正常: Stumble 检测正常工作，检测到 {stumble_stats['stumble_detected']} 次")

    print("=" * 80)


if __name__ == "__main__":
    print("此脚本需要在训练环境中运行")
    print("使用方法:")
    print("  1. 在训练脚本中导入: from debug_feet_stumble import diagnose_feet_stumble_reward")
    print("  2. 创建环境后调用: diagnose_feet_stumble_reward(env, step_count=100)")
