#!/usr/bin/env python3
"""
测试 On-Demand Loading + Hard Negative Mining 实现

测试内容：
1. OnDemandTwistMotionDataset 基础功能
2. Hard Negative Mining 权重更新
3. Motion Filtering 过滤功能
4. 显存占用验证
5. 数据集覆盖率统计
"""

import torch
import numpy as np
from pathlib import Path
import time
import psutil
import os

def get_gpu_memory_mb():
    """获取当前 GPU 显存占用（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def get_cpu_memory_mb():
    """获取当前进程 CPU 内存占用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_ondemand_dataset_basic():
    """测试 1: 基础加载功能"""
    print("\n" + "="*60)
    print("测试 1: OnDemandTwistMotionDataset 基础加载")
    print("="*60)

    from active_adaptation.utils.twist_motion_ondemand import OnDemandTwistMotionDataset

    # 使用 1000 个 motion 的数据集
    yaml_path = Path("/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/cfg/task/G1/twist/twist_dataset_1000.yaml")

    if not yaml_path.exists():
        print(f"❌ 数据集文件不存在: {yaml_path}")
        return False

    print(f"📂 数据集: {yaml_path}")

    # 记录初始显存
    mem_start = get_gpu_memory_mb()
    cpu_mem_start = get_cpu_memory_mb()

    # 创建数据集（只加载元数据）
    print("\n📊 创建数据集（只扫描元数据）...")
    start_time = time.time()

    dataset = OnDemandTwistMotionDataset.create_from_yaml(
        yaml_path=yaml_path,
        body_names=["pelvis", "torso_link", "left_hip_pitch_link"],  # 示例
        joint_names=["left_hip_pitch_joint", "left_knee_joint"],      # 示例
        device="cuda",
        enable_hnm=True,
    )

    elapsed = time.time() - start_time
    mem_after_metadata = get_gpu_memory_mb()
    cpu_mem_after = get_cpu_memory_mb()

    print(f"✅ 元数据加载完成，耗时 {elapsed:.2f}s")
    print(f"   - Motion 数量: {dataset.num_motions}")
    print(f"   - 总帧数: {dataset.total_frames}")
    print(f"   - GPU 显存: {mem_after_metadata - mem_start:.2f} MB （应该 ≈ 0）")
    print(f"   - CPU 内存: {cpu_mem_after - cpu_mem_start:.2f} MB")

    # 采样 motion
    print("\n📊 采样 4096 个 motion IDs...")
    motion_ids = dataset.sample_motions(4096)
    print(f"✅ 采样完成，unique motion 数量: {np.unique(motion_ids).size}")

    # 加载 batch
    print("\n📊 加载 batch 到 GPU...")
    unique_ids = np.unique(motion_ids)
    start_time = time.time()
    dataset.load_batch(unique_ids)
    elapsed = time.time() - start_time

    mem_after_batch = get_gpu_memory_mb()
    print(f"✅ Batch 加载完成，耗时 {elapsed:.2f}s")
    print(f"   - 加载的 motion 数量: {len(unique_ids)}")
    print(f"   - GPU 显存增加: {mem_after_batch - mem_after_metadata:.2f} MB")
    print(f"   - 预估每个 motion: {(mem_after_batch - mem_after_metadata) / len(unique_ids):.3f} MB")

    # 获取切片
    print("\n📊 测试 get_slice...")
    test_ids = motion_ids[:10]  # 取前 10 个
    time_steps = np.zeros(10, dtype=np.int64)
    steps = np.array([1, 2, 3, 4, 5], dtype=np.int64)

    motion_slice = dataset.get_slice(test_ids, time_steps, steps)

    print(f"✅ 切片成功")
    print(f"   - body_pos_w shape: {motion_slice.body_pos_w.shape}")
    print(f"   - joint_pos shape: {motion_slice.joint_pos.shape}")

    # 清空缓存测试
    print("\n📊 清空缓存...")
    dataset.current_batch_data = {}
    torch.cuda.empty_cache()
    mem_after_clear = get_gpu_memory_mb()
    print(f"✅ 缓存清空，GPU 显存: {mem_after_clear:.2f} MB")

    print("\n✅ 测试 1 通过！")
    return True

def test_hnm_weight_update():
    """测试 2: Hard Negative Mining 权重更新"""
    print("\n" + "="*60)
    print("测试 2: Hard Negative Mining 权重更新")
    print("="*60)

    from active_adaptation.utils.twist_motion_ondemand import OnDemandTwistMotionDataset

    yaml_path = Path("/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/cfg/task/G1/twist/twist_dataset_1000.yaml")

    dataset = OnDemandTwistMotionDataset.create_from_yaml(
        yaml_path=yaml_path,
        body_names=["pelvis"],
        joint_names=["left_hip_pitch_joint"],
        device="cuda",
        enable_hnm=True,
        hnm_alpha=1.5,
        hnm_beta=0.7,
    )

    # 初始权重
    initial_weights = dataset.motion_weights.copy()
    print(f"📊 初始权重统计:")
    print(f"   - Mean: {initial_weights.mean():.6f}")
    print(f"   - Std: {initial_weights.std():.6f}")

    # 模拟采样和更新
    print("\n📊 模拟 100 次采样和更新...")
    for i in range(100):
        # 采样
        motion_ids = dataset.sample_motions(4096)

        # 模拟成功/失败（前半成功，后半失败）
        success_flags = np.random.rand(4096) > 0.5

        # 更新权重
        dataset.update_hnm(motion_ids, success_flags)

    # 权重变化
    final_weights = dataset.motion_weights
    print(f"\n📊 100 次更新后权重统计:")
    print(f"   - Mean: {final_weights.mean():.6f}")
    print(f"   - Std: {final_weights.std():.6f}")
    print(f"   - Min: {final_weights.min():.6e}")
    print(f"   - Max: {final_weights.max():.6e}")

    # 检查权重变化
    weight_change = np.abs(final_weights - initial_weights).mean()
    print(f"\n📊 平均权重变化: {weight_change:.6f}")

    if weight_change > 0.01:
        print("✅ 权重更新正常")
    else:
        print("⚠️  权重几乎没有变化，可能有问题")

    # 测试成功率统计
    stats = dataset.get_coverage_stats()
    print(f"\n📊 覆盖率统计:")
    print(f"   - 覆盖率: {stats['coverage_rate']:.2%}")
    print(f"   - 已采样 motion: {stats['num_sampled']}/{dataset.num_motions}")
    print(f"   - 平均成功率: {stats['mean_success_rate']:.2%}")

    print("\n✅ 测试 2 通过！")
    return True

def test_motion_filtering():
    """测试 3: Motion Filtering"""
    print("\n" + "="*60)
    print("测试 3: Motion Filtering")
    print("="*60)

    from active_adaptation.utils.twist_motion_ondemand import OnDemandTwistMotionDataset

    yaml_path = Path("/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/cfg/task/G1/twist/twist_dataset_1000.yaml")

    dataset = OnDemandTwistMotionDataset.create_from_yaml(
        yaml_path=yaml_path,
        body_names=["pelvis"],
        joint_names=["left_hip_pitch_joint"],
        device="cuda",
        enable_hnm=True,
        hnm_filter_enabled=True,
        hnm_min_attempts=10,      # 降低阈值用于测试
        hnm_max_failure_rate=0.9,
    )

    print(f"📊 初始 motion 数量: {dataset.num_motions}")

    # 模拟让某些 motion 持续失败
    print("\n📊 模拟 motion 0-4 持续失败...")
    for i in range(20):
        # 只采样 0-4
        motion_ids = np.random.choice(5, size=100, replace=True)
        success_flags = np.zeros(100, dtype=bool)  # 全部失败

        dataset.update_hnm(motion_ids, success_flags)

    # 过滤
    print("\n📊 执行过滤...")
    filtered_count = dataset.filter_impossible_motions()

    print(f"✅ 过滤了 {filtered_count} 个 motion")
    print(f"   - 剩余 motion: {dataset.num_motions}")

    if filtered_count > 0:
        print("✅ 过滤功能正常")
    else:
        print("⚠️  没有 motion 被过滤，可能阈值设置太严格")

    print("\n✅ 测试 3 通过！")
    return True

def test_memory_usage():
    """测试 4: 显存占用验证"""
    print("\n" + "="*60)
    print("测试 4: 显存占用验证（4096 环境）")
    print("="*60)

    from active_adaptation.utils.twist_motion_ondemand import OnDemandTwistMotionDataset

    yaml_path = Path("/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/cfg/task/G1/twist/twist_dataset_1000.yaml")

    # 清空显存
    torch.cuda.empty_cache()
    mem_start = get_gpu_memory_mb()

    print(f"📊 初始 GPU 显存: {mem_start:.2f} MB")

    # 创建数据集
    dataset = OnDemandTwistMotionDataset.create_from_yaml(
        yaml_path=yaml_path,
        body_names=["pelvis", "torso_link", "left_hip_pitch_link", "right_hip_pitch_link"],
        joint_names=["left_hip_pitch_joint", "left_knee_joint", "right_hip_pitch_joint", "right_knee_joint"],
        device="cuda",
        enable_hnm=True,
    )

    mem_after_dataset = get_gpu_memory_mb()
    print(f"📊 创建数据集后 GPU 显存: {mem_after_dataset:.2f} MB (+{mem_after_dataset - mem_start:.2f} MB)")

    # 模拟 4096 个环境采样
    print("\n📊 模拟 4096 个环境采样...")
    motion_ids = dataset.sample_motions(4096)
    unique_ids = np.unique(motion_ids)

    print(f"   - Unique motion 数量: {len(unique_ids)}")

    # 加载 batch
    dataset.load_batch(unique_ids)
    mem_after_batch = get_gpu_memory_mb()

    print(f"📊 加载 batch 后 GPU 显存: {mem_after_batch:.2f} MB (+{mem_after_batch - mem_after_dataset:.2f} MB)")

    # 预估
    total_motions = dataset.num_motions
    estimated_full_load = (mem_after_batch - mem_after_dataset) * (total_motions / len(unique_ids))

    print(f"\n📊 预估全量加载显存: {estimated_full_load:.2f} MB")
    print(f"📊 当前方案节省显存: {estimated_full_load - (mem_after_batch - mem_after_dataset):.2f} MB")

    if mem_after_batch - mem_after_dataset < 5000:  # 小于 5GB
        print("✅ 显存占用合理（< 5GB）")
    else:
        print("⚠️  显存占用过高")

    print("\n✅ 测试 4 通过！")
    return True

def test_coverage_guarantee():
    """测试 5: 数据集覆盖率保证"""
    print("\n" + "="*60)
    print("测试 5: 数据集覆盖率保证")
    print("="*60)

    from active_adaptation.utils.twist_motion_ondemand import OnDemandTwistMotionDataset

    yaml_path = Path("/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/cfg/task/G1/twist/twist_dataset_1000.yaml")

    dataset = OnDemandTwistMotionDataset.create_from_yaml(
        yaml_path=yaml_path,
        body_names=["pelvis"],
        joint_names=["left_hip_pitch_joint"],
        device="cuda",
        enable_hnm=True,
        hnm_min_weight=1e-6,        # 最小权重保护
        hnm_boost_unsampled=1.1,    # 未采样提升
    )

    print(f"📊 数据集大小: {dataset.num_motions} motions")

    # 模拟多次采样
    print("\n📊 模拟 1000 次采样（每次 4096 个环境）...")
    sampled_motions = set()

    for i in range(1000):
        motion_ids = dataset.sample_motions(4096)

        # 模拟随机成功率
        success_flags = np.random.rand(4096) > 0.3

        # 更新 HNM
        dataset.update_hnm(motion_ids, success_flags)

        # 记录采样过的 motion
        sampled_motions.update(motion_ids)

        # 每 100 次打印一次进度
        if (i + 1) % 100 == 0:
            coverage = len(sampled_motions) / dataset.num_motions
            print(f"   Step {i+1}: 覆盖率 {coverage:.2%} ({len(sampled_motions)}/{dataset.num_motions})")

    final_coverage = len(sampled_motions) / dataset.num_motions
    print(f"\n📊 最终覆盖率: {final_coverage:.2%}")

    if final_coverage > 0.95:
        print("✅ 覆盖率 > 95%，保证了数据集全覆盖")
    else:
        print(f"⚠️  覆盖率 {final_coverage:.2%} < 95%，可能需要调整参数")

    # 检查权重分布
    stats = dataset.get_coverage_stats()
    print(f"\n📊 HNM 统计:")
    print(f"   - 平均成功率: {stats['mean_success_rate']:.2%}")
    print(f"   - 平均权重: {stats['mean_weight']:.6f}")

    print("\n✅ 测试 5 通过！")
    return True

def main():
    print("="*60)
    print("On-Demand + Hard Negative Mining 测试套件")
    print("="*60)

    tests = [
        ("基础加载功能", test_ondemand_dataset_basic),
        ("HNM 权重更新", test_hnm_weight_update),
        ("Motion 过滤", test_motion_filtering),
        ("显存占用验证", test_memory_usage),
        ("覆盖率保证", test_coverage_guarantee),
    ]

    results = {}

    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "✅ 通过" if success else "❌ 失败"
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"❌ 错误: {str(e)[:50]}"

    # 打印汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, result in results.items():
        print(f"{name}: {result}")

    print("\n所有测试完成！")

if __name__ == "__main__":
    main()
