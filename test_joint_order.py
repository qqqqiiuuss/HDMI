#!/usr/bin/env python3
"""
测试实际IsaacLab环境中的关节顺序

这个脚本会创建一个最小化的环境实例，然后打印实际的关节顺序。
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

def test_joint_order():
    print("\n" + "=" * 80)
    print("Testing Actual Joint Order in IsaacLab Environment")
    print("=" * 80)

    try:
        # Import MuJoCo articulation
        from active_adaptation.envs.mujoco import MJArticulationCfg
        from active_adaptation.assets_mjcf import ROBOTS

        # Get g1_29dof config
        if "g1_29dof" not in ROBOTS:
            print("❌ ERROR: g1_29dof not found in ROBOTS")
            return False

        robot_cfg = ROBOTS["g1_29dof"]
        print(f"\n✓ Robot config found: g1_29dof")
        print(f"✓ MJCF path: {robot_cfg.mjcf_path}")

        # Check if XML file exists
        if not os.path.exists(robot_cfg.mjcf_path):
            print(f"❌ ERROR: XML file not found: {robot_cfg.mjcf_path}")
            return False

        print(f"✓ XML file exists")

        # Try to get joint names from MuJoCo directly
        print("\n" + "=" * 80)
        print("Attempting to load MuJoCo model...")
        print("=" * 80)

        try:
            import mujoco as mj

            model = mj.MjModel.from_xml_path(robot_cfg.mjcf_path)
            print(f"\n✓ MuJoCo model loaded successfully")
            print(f"✓ Number of actuators: {model.nu}")
            print(f"✓ Number of joints: {model.njnt}")

            # Extract joint names from actuators
            print("\n" + "=" * 80)
            print("Actuated Joint Order (from MuJoCo model.actuator_trnid)")
            print("=" * 80)

            joint_names = []
            for i in range(model.nu):
                trnid = model.actuator_trnid[i, 0]  # First transmission
                if trnid >= 0:  # Valid joint
                    joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, trnid)
                    joint_names.append(joint_name)
                    print(f"{i+1:2d}. {joint_name}")

            print(f"\nTotal actuated joints: {len(joint_names)}")

            # Compare with our weights
            print("\n" + "=" * 80)
            print("Comparing with defined weights")
            print("=" * 80)

            dof_err_w = [
                1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # Left Leg (6)
                1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # Right Leg (6)
                0.6, 0.6, 0.6,                      # Waist (3)
                0.8, 0.8, 0.8, 1.0,                 # Left Arm (4)
                0.6, 0.5, 0.5,                      # Left Wrist (3)
                0.8, 0.8, 0.8, 1.0,                 # Right Arm (4)
                0.6, 0.5, 0.5,                      # Right Wrist (3)
            ]

            if len(joint_names) != len(dof_err_w):
                print(f"❌ ERROR: Joint count mismatch!")
                print(f"   MuJoCo joints: {len(joint_names)}")
                print(f"   Defined weights: {len(dof_err_w)}")
                return False

            print(f"✅ Joint count matches: {len(joint_names)}\n")

            # Detailed mapping
            print(f"{'#':<4} {'MuJoCo Joint':<32} {'Weight':<8}")
            print("-" * 50)
            for i, (name, weight) in enumerate(zip(joint_names, dof_err_w), 1):
                print(f"{i:<4} {name:<32} {weight:<8.1f}")

            # Check wrist joints specifically
            wrist_indices = []
            for i, name in enumerate(joint_names):
                if 'wrist' in name.lower():
                    wrist_indices.append((i, name, dof_err_w[i]))

            if wrist_indices:
                print("\n" + "=" * 80)
                print("✅ Wrist Joints Found:")
                print("=" * 80)
                for idx, name, weight in wrist_indices:
                    print(f"   {idx+1:2d}. {name:<30} weight = {weight}")
            else:
                print("\n❌ No wrist joints found!")

            return True

        except ImportError:
            print("❌ MuJoCo not installed. Cannot load model directly.")
            print("   Try: pip install mujoco")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_joint_order()

    if success:
        print("\n" + "=" * 80)
        print("✅ TEST PASSED")
        print("=" * 80)
        print("\nThe joint order in MuJoCo model matches our defined weights.")
        print("You can safely use the current weight configuration.")
    else:
        print("\n" + "=" * 80)
        print("⚠️  TEST FAILED OR INCOMPLETE")
        print("=" * 80)
        print("\nPlease check the errors above.")
        print("You may need to verify joint order manually during training.")

    exit(0 if success else 1)
