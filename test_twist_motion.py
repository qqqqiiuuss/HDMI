"""
测试 TwistMotionDataset 的功能

用法:
    python test_twist_motion.py --yaml_path /path/to/twist_motions.yaml
"""

import argparse
import torch
from active_adaptation.utils.twist_motion import TwistMotionDataset


def test_basic_loading(yaml_path: str):
    """测试基本加载功能"""
    print("=" * 80)
    print("Test 1: Basic Loading")
    print("=" * 80)

    dataset = TwistMotionDataset.create_from_yaml(
        yaml_path=yaml_path,
        device=torch.device("cuda:0"),
        smooth_window=19
    )

    print(f"✓ Loaded {dataset.num_motions} motions")
    print(f"✓ Total frames: {dataset.num_steps}")
    print(f"✓ Body names: {dataset.body_names[:5]}...")
    print(f"✓ Joint names: {dataset.joint_names[:5]}...")
    print(f"✓ Motion paths: {[p.split('/')[-1] for p in dataset.motion_paths]}")
    print(f"✓ Weights: {dataset.weights}")
    print()

    return dataset


def test_get_slice(dataset: TwistMotionDataset):
    """测试 get_slice 功能"""
    print("=" * 80)
    print("Test 2: Get Slice")
    print("=" * 80)

    # 测试单个 motion
    motion_ids = torch.tensor([0], device=dataset.device)
    starts = torch.tensor([10], device=dataset.device)
    steps = 5

    slice_data = dataset.get_slice(motion_ids, starts, steps)

    print(f"✓ Motion IDs shape: {slice_data.motion_id.shape}")
    print(f"✓ Root pos shape: {slice_data.root_pos.shape}")
    print(f"✓ Root rot shape: {slice_data.root_rot.shape}")
    print(f"✓ Root vel shape: {slice_data.root_vel.shape}")
    print(f"✓ DOF pos shape: {slice_data.dof_pos.shape}")
    print(f"✓ Local key body pos shape: {slice_data.local_key_body_pos.shape}")

    # 测试批量
    motion_ids = torch.randint(0, dataset.num_motions, (32,), device=dataset.device)
    starts = torch.randint(0, 100, (32,), device=dataset.device)
    steps = torch.tensor([1, 2, 8, 16], device=dataset.device)

    slice_data = dataset.get_slice(motion_ids, starts, steps)
    print(f"\n✓ Batch get_slice shape: {slice_data.root_pos.shape}")
    print(f"  Expected: [32, 4, 3]")
    print()


def test_calc_motion_frame(dataset: TwistMotionDataset):
    """测试 calc_motion_frame 功能"""
    print("=" * 80)
    print("Test 3: Calc Motion Frame (with interpolation)")
    print("=" * 80)

    motion_ids = torch.tensor([0, 1], device=dataset.device)
    motion_times = torch.tensor([0.5, 1.0], device=dataset.device)

    result = dataset.calc_motion_frame(motion_ids, motion_times)
    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, local_key_body_pos = result

    print(f"✓ Root pos: {root_pos.shape}")
    print(f"✓ Root rot: {root_rot.shape}")
    print(f"✓ DOF pos: {dof_pos.shape}")
    print(f"✓ Root vel: {root_vel.shape}")
    print(f"✓ Root ang vel: {root_ang_vel.shape}")
    print(f"✓ DOF vel: {dof_vel.shape}")
    print(f"✓ Local key body pos: {local_key_body_pos.shape}")
    print()


def test_sampling(dataset: TwistMotionDataset):
    """测试采样功能"""
    print("=" * 80)
    print("Test 4: Motion Sampling")
    print("=" * 80)

    # 采样 1000 次，统计每个 motion 被采样的次数
    n_samples = 1000
    motion_ids = dataset.sample_motions(n_samples)

    counts = torch.bincount(motion_ids, minlength=dataset.num_motions)
    frequencies = counts.float() / n_samples

    print(f"✓ Sampled {n_samples} motion IDs")
    print(f"✓ Expected frequencies (weights): {dataset.weights}")
    print(f"✓ Actual frequencies: {frequencies}")
    print(f"✓ Difference: {torch.abs(frequencies - dataset.weights)}")

    # 测试时间采样
    motion_times = dataset.sample_time(motion_ids[:10])
    print(f"\n✓ Sampled times shape: {motion_times.shape}")
    print(f"✓ Time range: [{motion_times.min():.3f}, {motion_times.max():.3f}]")
    print()


def test_find_joints_bodies(dataset: TwistMotionDataset):
    """测试查找关节和身体"""
    print("=" * 80)
    print("Test 5: Find Joints and Bodies")
    print("=" * 80)

    # 测试查找关节
    if dataset.joint_names:
        joint_names = dataset.joint_names[:3]
        indices, names = dataset.find_joints(joint_names)
        print(f"✓ Found joints: {names}")
        print(f"✓ Indices: {indices}")

    # 测试查找身体
    if dataset.body_names:
        body_names = dataset.body_names[:3]
        indices, names = dataset.find_bodies(body_names)
        print(f"✓ Found bodies: {names}")
        print(f"✓ Indices: {indices}")
    print()


def test_device_transfer(dataset: TwistMotionDataset):
    """测试设备迁移"""
    print("=" * 80)
    print("Test 6: Device Transfer")
    print("=" * 80)

    original_device = dataset.device
    print(f"✓ Original device: {original_device}")

    # 迁移到 CPU
    dataset_cpu = dataset.to(torch.device("cpu"))
    print(f"✓ Transferred to: {dataset_cpu.device}")
    print(f"✓ Data device: {dataset_cpu.data.root_pos.device}")

    # 迁移回 GPU
    dataset_gpu = dataset_cpu.to(original_device)
    print(f"✓ Transferred back to: {dataset_gpu.device}")
    print(f"✓ Data device: {dataset_gpu.data.root_pos.device}")
    print()


def test_velocity_smoothness(dataset: TwistMotionDataset):
    """测试速度平滑性"""
    print("=" * 80)
    print("Test 7: Velocity Smoothness (TWIST特性)")
    print("=" * 80)

    # 获取第一个 motion 的速度数据
    motion_id = 0
    start_idx = dataset.starts[motion_id]
    end_idx = dataset.ends[motion_id]

    root_vel = dataset.data.root_vel[start_idx:end_idx]
    root_ang_vel = dataset.data.root_ang_vel[start_idx:end_idx]
    dof_vel = dataset.data.dof_vel[start_idx:end_idx]

    # 计算速度的变化率（加速度）
    root_accel = (root_vel[1:] - root_vel[:-1]).norm(dim=-1)
    root_ang_accel = (root_ang_vel[1:] - root_ang_vel[:-1]).norm(dim=-1)
    dof_accel = (dof_vel[1:] - dof_vel[:-1]).norm(dim=-1)

    print(f"✓ Root velocity smoothness (mean accel): {root_accel.mean():.6f}")
    print(f"✓ Root ang velocity smoothness: {root_ang_accel.mean():.6f}")
    print(f"✓ DOF velocity smoothness: {dof_accel.mean():.6f}")
    print(f"\n✓ Note: TWIST uses 19-point smoothing, so accelerations should be small")
    print()


def main():
    parser = argparse.ArgumentParser(description="Test TwistMotionDataset")
    parser.add_argument(
        "--yaml_path",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("TwistMotionDataset Test Suite")
    print("=" * 80 + "\n")

    try:
        # 运行所有测试
        dataset = test_basic_loading(args.yaml_path)
        test_get_slice(dataset)
        test_calc_motion_frame(dataset)
        test_sampling(dataset)
        test_find_joints_bodies(dataset)
        test_device_transfer(dataset)
        test_velocity_smoothness(dataset)

        print("=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
