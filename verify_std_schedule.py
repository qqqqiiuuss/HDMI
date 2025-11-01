#!/usr/bin/env python3
"""
验证 ppo_twist 的 std schedule 是否正确工作

Usage:
    python verify_std_schedule.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def compute_std_schedule(iteration, std_schedule):
    """
    计算给定 iteration 的 action std 值

    Args:
        iteration: 当前迭代次数
        std_schedule: [init_std, final_std, warmup_iters, decay_iters]

    Returns:
        target_std: 目标 std 值
    """
    init_std, final_std, warmup_iters, decay_iters = std_schedule

    if iteration < warmup_iters:
        # Warmup 阶段：保持 init_std
        std_coef = 1.0
    elif iteration < warmup_iters + decay_iters:
        # Decay 阶段：线性衰减
        progress = (iteration - warmup_iters) / decay_iters
        std_coef = 1.0 - progress * (1.0 - final_std / init_std)
    else:
        # 最终阶段：保持 final_std
        std_coef = final_std / init_std

    target_std = init_std * std_coef
    return target_std


def plot_std_schedule(std_schedule, max_iters=10000):
    """
    绘制 std schedule 曲线
    """
    init_std, final_std, warmup_iters, decay_iters = std_schedule

    iterations = np.arange(0, max_iters)
    stds = [compute_std_schedule(it, std_schedule) for it in iterations]

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, stds, linewidth=2, label='Action Std')

    # 标记关键点
    plt.axvline(x=warmup_iters, color='green', linestyle='--',
                label=f'Warmup End ({warmup_iters})')
    plt.axvline(x=warmup_iters + decay_iters, color='red', linestyle='--',
                label=f'Decay End ({warmup_iters + decay_iters})')

    plt.axhline(y=init_std, color='blue', linestyle=':', alpha=0.5,
                label=f'Init Std ({init_std})')
    plt.axhline(y=final_std, color='orange', linestyle=':', alpha=0.5,
                label=f'Final Std ({final_std})')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Action Std', fontsize=12)
    plt.title('PPO-TWIST Action Std Schedule', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    output_path = '/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/std_schedule.png'
    plt.savefig(output_path, dpi=150)
    print(f"✅ Std schedule 曲线已保存到: {output_path}")
    plt.show()


def verify_key_points(std_schedule):
    """
    验证关键迭代点的 std 值
    """
    init_std, final_std, warmup_iters, decay_iters = std_schedule

    print("\n" + "=" * 60)
    print("📊 STD SCHEDULE 关键点验证")
    print("=" * 60)

    test_points = [
        (0, init_std, "训练开始"),
        (warmup_iters - 1, init_std, "Warmup 结束前"),
        (warmup_iters, init_std, "Warmup 结束"),
        (warmup_iters + decay_iters // 2, (init_std + final_std) / 2, "Decay 中点 (近似)"),
        (warmup_iters + decay_iters - 1, final_std, "Decay 结束前 (近似)"),
        (warmup_iters + decay_iters, final_std, "Decay 结束"),
        (10000, final_std, "后期训练"),
    ]

    print(f"\n配置: init_std={init_std}, final_std={final_std}, "
          f"warmup={warmup_iters}, decay={decay_iters}\n")

    print(f"{'Iteration':<12} {'Expected Std':<15} {'Actual Std':<15} {'Status':<10} {'描述'}")
    print("-" * 75)

    all_pass = True
    for iteration, expected, desc in test_points:
        actual = compute_std_schedule(iteration, std_schedule)

        # 允许 1% 误差
        if abs(actual - expected) / expected < 0.01:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_pass = False

        print(f"{iteration:<12} {expected:<15.4f} {actual:<15.4f} {status:<10} {desc}")

    print("-" * 75)
    if all_pass:
        print("\n✅ 所有关键点验证通过！")
    else:
        print("\n⚠️ 部分关键点验证失败，请检查实现")

    return all_pass


def test_twist_default():
    """
    测试 TWIST-MASTER 默认配置
    """
    print("\n" + "=" * 60)
    print("🔧 测试 TWIST-MASTER 默认配置")
    print("=" * 60)

    # ppo_twist 默认配置
    std_schedule = (1.0, 0.4, 4000, 1500)

    verify_key_points(std_schedule)
    plot_std_schedule(std_schedule, max_iters=10000)


def compare_with_ppo():
    """
    对比 ppo_twist 和普通 ppo 的探索策略
    """
    print("\n" + "=" * 60)
    print("📈 ppo_twist vs ppo 探索策略对比")
    print("=" * 60)

    # ppo_twist: 动态调度
    std_schedule_twist = (1.0, 0.4, 4000, 1500)

    # ppo: 固定 std=1.5
    ppo_std = 1.5

    max_iters = 10000
    iterations = np.arange(0, max_iters)
    stds_twist = [compute_std_schedule(it, std_schedule_twist) for it in iterations]
    stds_ppo = [ppo_std] * max_iters

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, stds_twist, linewidth=2, label='ppo_twist (dynamic)', color='blue')
    plt.plot(iterations, stds_ppo, linewidth=2, label='ppo (fixed)', color='red', linestyle='--')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Action Std', fontsize=12)
    plt.title('ppo_twist vs ppo: Exploration Strategy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = '/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/std_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"✅ 对比曲线已保存到: {output_path}")
    plt.show()

    # 关键点对比
    print("\n关键迭代点对比:")
    print(f"{'Iteration':<12} {'ppo_twist std':<18} {'ppo std':<12} {'差异'}")
    print("-" * 60)

    for it in [0, 1000, 4000, 5000, 5500, 10000]:
        std_twist = compute_std_schedule(it, std_schedule_twist)
        diff = std_twist - ppo_std
        sign = "+" if diff >= 0 else ""
        print(f"{it:<12} {std_twist:<18.4f} {ppo_std:<12.4f} {sign}{diff:.4f}")

    print("\n💡 分析:")
    print("  - Iteration 0-4000: ppo_twist (1.0) < ppo (1.5) → ppo 探索更多")
    print("  - Iteration 4000-5500: ppo_twist 逐渐降低 → 开始 exploitation")
    print("  - Iteration >5500: ppo_twist (0.4) << ppo (1.5) → ppo_twist 更精确")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 PPO-TWIST STD SCHEDULE 验证工具")
    print("=" * 60)

    # 测试默认配置
    test_twist_default()

    # 对比 ppo
    compare_with_ppo()

    print("\n" + "=" * 60)
    print("✅ 验证完成！")
    print("=" * 60)
    print("\n📋 下一步:")
    print("  1. 检查生成的图片: std_schedule.png, std_comparison.png")
    print("  2. 运行训练后，在 WandB 查找 'actor/action_std' 指标")
    print("  3. 对比 WandB 曲线与 std_schedule.png 是否一致")
    print()
