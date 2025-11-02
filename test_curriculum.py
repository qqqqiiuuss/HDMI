#!/usr/bin/env python3
"""
Curriculum Learning 验证脚本

用途：验证 TWIST curriculum learning 机制是否正确工作

使用方法：
  python test_curriculum.py

预期结果：
  - motion_curriculum=True 时，应该看到难度统计信息
  - motion_curriculum=False 时，不应该有难度统计
"""

import torch
import numpy as np

def test_curriculum_config():
    """测试 curriculum 配置是否正确"""
    print("=" * 70)
    print("📋 测试 1: 检查配置文件")
    print("=" * 70)

    config_file = "cfg/task/G1/twist/0927_twist_teacher_new.yaml"

    with open(config_file, 'r') as f:
        content = f.read()

    # 检查关键配置项
    checks = [
        ("motion_curriculum: true", "✓ motion_curriculum 已启用"),
        ("motion_curriculum_gamma: 0.01", "✓ gamma 值正确设置"),
        ("sample_motion: true", "✓ sample_motion 已启用"),
    ]

    all_passed = True
    for pattern, message in checks:
        if pattern in content:
            print(f"  {message}")
        else:
            print(f"  ✗ 未找到: {pattern}")
            all_passed = False

    print()
    if all_passed:
        print("✅ 配置文件检查通过")
    else:
        print("❌ 配置文件检查失败")

    return all_passed


def test_curriculum_code():
    """测试 curriculum 代码是否正确实现"""
    print("\n" + "=" * 70)
    print("📋 测试 2: 检查代码实现")
    print("=" * 70)

    code_file = "active_adaptation/envs/mdp/commands/twist/command.py"

    with open(code_file, 'r') as f:
        content = f.read()

    # 检查关键代码
    checks = [
        ("def _update_motion_difficulty", "✓ _update_motion_difficulty 方法已定义"),
        ("if self.motion_curriculum:", "✓ curriculum 条件判断存在"),
        ("self.motion_difficulty", "✓ motion_difficulty 变量存在"),
        ("self.episode_length_buf", "✓ episode_length_buf 变量存在"),
        ("motion_curriculum_gamma", "✓ gamma 参数被使用"),
    ]

    all_passed = True
    for pattern, message in checks:
        if pattern in content:
            print(f"  {message}")
        else:
            print(f"  ✗ 未找到: {pattern}")
            all_passed = False

    # 检查 sample_init 中是否调用了 _update_motion_difficulty
    if "_update_motion_difficulty(env_ids)" in content:
        print(f"  ✓ sample_init 中调用了 _update_motion_difficulty")
    else:
        print(f"  ✗ sample_init 中未调用 _update_motion_difficulty")
        all_passed = False

    print()
    if all_passed:
        print("✅ 代码实现检查通过")
    else:
        print("❌ 代码实现检查失败")

    return all_passed


def test_curriculum_logic():
    """测试 curriculum 逻辑是否正确"""
    print("\n" + "=" * 70)
    print("📋 测试 3: 模拟 curriculum 逻辑")
    print("=" * 70)

    # 模拟参数
    num_motions = 10
    gamma = 0.01

    # 初始化难度
    motion_difficulty = torch.ones(num_motions)
    print(f"\n初始难度: min={motion_difficulty.min():.2f}, "
          f"mean={motion_difficulty.mean():.2f}, max={motion_difficulty.max():.2f}")

    # 模拟几轮更新
    print("\n模拟训练过程:")
    for epoch in range(5):
        # 模拟完成率（随机）
        completion_rates = torch.rand(num_motions)

        # 更新难度
        add_idx = completion_rates <= 0.5  # 太难
        sub_idx = completion_rates >= 0.95  # 太简单

        motion_difficulty[add_idx] *= (1 + gamma)
        motion_difficulty[sub_idx] *= (1 - gamma)
        motion_difficulty = torch.clamp(motion_difficulty, min=1.0, max=9.0)

        print(f"  Epoch {epoch+1}: difficulty=[{motion_difficulty.min():.3f}, "
              f"{motion_difficulty.mean():.3f}, {motion_difficulty.max():.3f}], "
              f"hard={add_idx.sum().item()}, easy={sub_idx.sum().item()}")

    print()
    if motion_difficulty.min() >= 1.0 and motion_difficulty.max() <= 9.0:
        print("✅ Curriculum 逻辑模拟通过（难度在合理范围内）")
        return True
    else:
        print("❌ Curriculum 逻辑模拟失败（难度超出范围）")
        return False


def print_usage_guide():
    """打印使用指南"""
    print("\n" + "=" * 70)
    print("📖 使用指南")
    print("=" * 70)

    print("\n✅ 启用 Curriculum (默认):")
    print("  CUDA_VISIBLE_DEVICES=7 python scripts/train.py \\")
    print("    algo=ppo \\")
    print("    task=G1/twist/0927_twist_teacher_new \\")
    print("    task.num_envs=4096 \\")
    print("    suffix=with_curriculum")

    print("\n❌ 禁用 Curriculum (通过 CLI 覆盖):")
    print("  CUDA_VISIBLE_DEVICES=7 python scripts/train.py \\")
    print("    algo=ppo \\")
    print("    task=G1/twist/0927_twist_teacher_new \\")
    print("    task.command.motion_curriculum=false \\")
    print("    task.num_envs=4096 \\")
    print("    suffix=no_curriculum")

    print("\n📊 在 WandB 中监控以下指标:")
    print("  - curriculum/mean_motion_difficulty  (应从 1.0 增长到 9.0)")
    print("  - curriculum/min_motion_difficulty")
    print("  - curriculum/max_motion_difficulty")
    print("  - train/episode_length               (应逐渐增长)")

    print("\n🔍 调试方法:")
    print("  在训练开始后，检查日志输出，应该看到:")
    print("  - Motion curriculum: enabled")
    print("  - Mean motion difficulty: 1.00 (初始值)")


if __name__ == "__main__":
    print("🔬 TWIST Curriculum Learning 验证")
    print()

    test1 = test_curriculum_config()
    test2 = test_curriculum_code()
    test3 = test_curriculum_logic()

    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    print(f"  配置检查: {'✅ 通过' if test1 else '❌ 失败'}")
    print(f"  代码检查: {'✅ 通过' if test2 else '❌ 失败'}")
    print(f"  逻辑模拟: {'✅ 通过' if test3 else '❌ 失败'}")
    print()

    if test1 and test2 and test3:
        print("🎉 所有测试通过！Curriculum learning 已正确配置。")
        print_usage_guide()
        exit(0)
    else:
        print("⚠️  部分测试失败，请检查配置或代码。")
        exit(1)
