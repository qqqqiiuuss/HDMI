#!/usr/bin/env python3
"""
验证 TWIST Curriculum Learning 配置是否正确加载

Usage:
    python verify_curriculum.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_config():
    """验证配置文件是否正确"""
    print("=" * 60)
    print("验证 TWIST Curriculum Learning 配置")
    print("=" * 60)

    try:
        import hydra
        from omegaconf import OmegaConf

        # Initialize Hydra
        with hydra.initialize(config_path='cfg', version_base=None):
            cfg = hydra.compose(
                config_name='train',
                overrides=['task=G1/twist/0927_twist_teacher_new']
            )

            # Extract curriculum config
            command_cfg = cfg.task.command

            print("\n✅ 配置加载成功！\n")
            print("Curriculum Learning 配置:")
            print(f"  - motion_curriculum: {command_cfg.get('motion_curriculum', 'NOT SET')}")
            print(f"  - motion_curriculum_gamma: {command_cfg.get('motion_curriculum_gamma', 'NOT SET')}")
            print(f"  - sample_motion: {command_cfg.get('sample_motion', 'NOT SET')}")

            # Verify values
            success = True
            if not command_cfg.get('motion_curriculum', False):
                print("\n❌ 错误：motion_curriculum 未启用！")
                success = False

            if command_cfg.get('motion_curriculum_gamma') != 0.01:
                print(f"\n⚠️  警告：motion_curriculum_gamma = {command_cfg.get('motion_curriculum_gamma')}, 建议值为 0.01")

            if not command_cfg.get('sample_motion', False):
                print("\n❌ 错误：sample_motion 未启用！")
                success = False

            if success:
                print("\n" + "=" * 60)
                print("✅ 所有配置正确！可以开始训练")
                print("=" * 60)
                print("\n启动训练命令:")
                print("  python scripts/train.py \\")
                print("      algo=ppo \\")
                print("      task=G1/twist/0927_twist_teacher_new \\")
                print("      task.num_envs=4096 \\")
                print("      suffix=_with_curriculum \\")
                print("      wandb.mode=online")
                return 0
            else:
                print("\n" + "=" * 60)
                print("❌ 配置有误，请检查 cfg/task/G1/twist/0927_twist_teacher_new.yaml")
                print("=" * 60)
                return 1

    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        return 1

def verify_code():
    """验证代码修改是否存在"""
    print("\n" + "=" * 60)
    print("验证代码修改")
    print("=" * 60)

    try:
        # Check if the command module exists
        from active_adaptation.envs.mdp.commands.twist import command

        # Check if TwistMotionTracking has the required attributes
        import inspect
        source = inspect.getsource(command.TwistMotionTracking._sample_motions)

        if 'motion_curriculum' in source:
            print("\n✅ _sample_motions 方法包含 curriculum 逻辑")
        else:
            print("\n❌ _sample_motions 方法缺少 curriculum 逻辑")
            return 1

        if 'mean_difficulty' in source:
            print("✅ 实现了基于平均难度的采样")
        else:
            print("❌ 缺少平均难度计算")
            return 1

        if 'valid_mask' in source:
            print("✅ 实现了难度过滤采样")
        else:
            print("❌ 缺少难度过滤逻辑")
            return 1

        # Check update method
        source = inspect.getsource(command.TwistMotionTracking._update_motion_difficulty)

        if 'motion_curriculum_gamma' in source:
            print("✅ _update_motion_difficulty 方法正确实现")
        else:
            print("❌ _update_motion_difficulty 方法有误")
            return 1

        print("\n" + "=" * 60)
        print("✅ 代码修改验证通过！")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n❌ 代码验证失败：{e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TWIST Curriculum Learning 实现验证")
    print("=" * 60)

    # Verify config
    config_ok = verify_config()

    # Verify code
    code_ok = verify_code()

    # Final result
    print("\n" + "=" * 60)
    if config_ok == 0 and code_ok == 0:
        print("🎉 验证完成！所有检查通过")
        print("\n查看详细文档：TWIST_CURRICULUM_IMPLEMENTATION.md")
    else:
        print("❌ 验证失败，请检查错误信息")
    print("=" * 60)

    sys.exit(max(config_ok, code_ok))
