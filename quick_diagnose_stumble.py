"""
快速诊断 feet_stumble_twist 为什么一直是 0

使用方法：
python quick_diagnose_stumble.py
"""

import sys
sys.path.insert(0, "/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk")

import torch
import hydra
from omegaconf import DictConfig
from scripts.helpers import create_env

@hydra.main(version_base=None, config_path="cfg", config_name="train")
def main(cfg: DictConfig):
    print("\n" + "=" * 80)
    print("开始诊断 feet_stumble_twist reward")
    print("=" * 80)

    # 强制使用你的配置
    cfg.task = "G1/twist/0927_twist_teacher_new"
    cfg.num_envs = 16  # 用少量环境加快测试

    print(f"\n[1] 创建环境...")
    print(f"    Task: {cfg.task}")

    env = create_env(cfg)

    # 检查 contact sensor
    print(f"\n[2] 检查 Contact Sensor")
    print("-" * 80)

    if "contact_forces" not in env.scene:
        print("❌ 错误: scene 中没有 'contact_forces'!")
        return

    contact_sensor = env.scene["contact_forces"]
    print(f"✓ Contact sensor 存在")
    print(f"  类型: {type(contact_sensor)}")
    print(f"  监控的 bodies ({len(contact_sensor.body_names)}): {contact_sensor.body_names}")

    # 检查 feet_stumble_twist reward
    print(f"\n[3] 检查 feet_stumble_twist Reward 配置")
    print("-" * 80)

    if hasattr(env, 'reward_manager'):
        reward_manager = env.reward_manager

        # 查找 feet_stumble_twist
        stumble_reward = None
        for name, reward_term in reward_manager.active_terms.items():
            if 'feet_stumble' in name:
                print(f"✓ 找到 reward: {name}")
                print(f"  类型: {type(reward_term)}")
                print(f"  权重: {reward_term.weight}")
                stumble_reward = reward_term

                # 检查 body indices
                if hasattr(reward_term, 'contact_body_indices'):
                    print(f"  Body indices: {reward_term.contact_body_indices}")
                if hasattr(reward_term, 'body_names'):
                    print(f"  Body names: {reward_term.body_names}")
                break

        if stumble_reward is None:
            print("❌ 错误: 没有找到 feet_stumble 相关的 reward!")
            print(f"   可用的 rewards: {list(reward_manager.active_terms.keys())}")
            return
    else:
        print("❌ 环境没有 reward_manager")
        return

    # 运行环境并收集数据
    print(f"\n[4] 运行环境并监控 Contact Forces")
    print("-" * 80)

    stats = {
        'contact_forces_zero_steps': 0,
        'max_xy_force': 0.0,
        'max_z_force': 0.0,
        'max_ratio': 0.0,
        'stumble_count': 0,
        'total_steps': 0
    }

    print("运行 200 步...")

    # 重置环境
    env.reset()

    for step in range(200):
        # 随机动作
        actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)

        # 执行
        obs, rewards, dones, info = env.step(actions)

        # 获取 contact forces
        contact_forces = contact_sensor.data.net_forces_w

        # 检查是否全零
        if contact_forces.abs().sum() < 1e-6:
            stats['contact_forces_zero_steps'] += 1

        # 统计最大值
        stats['max_xy_force'] = max(stats['max_xy_force'],
                                    contact_forces[..., :2].norm(dim=-1).max().item())
        stats['max_z_force'] = max(stats['max_z_force'],
                                   contact_forces[..., 2].abs().max().item())

        # 检查 stumble reward
        if 'feet_stumble_twist' in rewards:
            stumble_val = rewards['feet_stumble_twist'].sum().item()
            if abs(stumble_val) > 1e-6:
                stats['stumble_count'] += 1

        stats['total_steps'] += 1

        # 每 50 步打印一次
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/200:")
            print(f"    Contact forces 范围: [{contact_forces.min():.2f}, {contact_forces.max():.2f}]")
            if 'feet_stumble_twist' in rewards:
                print(f"    feet_stumble_twist: {rewards['feet_stumble_twist'].mean():.6f}")

    # 打印诊断结果
    print(f"\n[5] 诊断结果")
    print("=" * 80)

    print(f"\nContact Forces 统计:")
    print(f"  - 全零的步数: {stats['contact_forces_zero_steps']} / {stats['total_steps']}")
    print(f"  - 最大 XY force: {stats['max_xy_force']:.4f}")
    print(f"  - 最大 Z force: {stats['max_z_force']:.4f}")

    print(f"\nfeet_stumble_twist 统计:")
    print(f"  - 非零次数: {stats['stumble_count']} / {stats['total_steps']}")

    # 诊断结论
    print(f"\n" + "=" * 80)
    print("结论:")
    print("=" * 80)

    if stats['contact_forces_zero_steps'] == stats['total_steps']:
        print("❌ 问题确认: Contact forces 完全是 0!")
        print("\n可能原因:")
        print("  1. ContactSensor 的 prim_path 配置错误，没有监控到正确的 bodies")
        print("  2. 机器人模型问题，没有与地面接触")
        print("  3. IsaacLab 的 ContactSensor 需要显式刷新")
        print("\n建议:")
        print("  1. 检查 locomotion.py 中 ContactSensorCfg 的 prim_path")
        print("  2. 在可视化模式下查看机器人是否正常站立")
        print("  3. 检查其他 feet 相关的 reward (feet_slip, feet_contact_forces)")

    elif stats['stumble_count'] == 0:
        print("⚠️  Contact forces 有值，但 stumble 从未触发")
        print(f"\n力的统计:")
        print(f"  - 最大 XY/Z 比率需要 > 4.0 才能触发")
        print(f"  - 当前最大 XY force: {stats['max_xy_force']:.4f}")
        print(f"  - 当前最大 Z force: {stats['max_z_force']:.4f}")

        if stats['max_z_force'] > 0:
            estimated_ratio = stats['max_xy_force'] / stats['max_z_force']
            print(f"  - 估计的最大比率: {estimated_ratio:.4f}")

            if estimated_ratio < 2.0:
                print("\n✓ 这是正常的！机器人步态很稳定，XY 力远小于 Z 力")
                print("  TWIST-master 中出现波动可能是因为:")
                print("  - 训练早期步态不稳定")
                print("  - 不同的物理参数设置")
                print("  - 随机化导致的偶然碰撞")
            else:
                print("\n建议: 考虑降低阈值到 2.0 或 3.0 来测试")
    else:
        print(f"✓ stumble 检测正常工作! 触发了 {stats['stumble_count']} 次")

    print("=" * 80)

    # 额外检查：其他 feet rewards
    print(f"\n[6] 检查其他 Feet Rewards")
    print("-" * 80)

    feet_rewards = [name for name in reward_manager.active_terms.keys() if 'feet' in name.lower()]
    print(f"找到 {len(feet_rewards)} 个 feet 相关的 rewards:")
    for name in feet_rewards:
        print(f"  - {name}")

    print("\n提示: 如果其他 feet rewards 也都是 0，说明是 ContactSensor 的问题")

if __name__ == "__main__":
    main()
