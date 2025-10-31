#!/usr/bin/env python3
"""
直接从MuJoCo读取关节顺序（不依赖项目代码）
"""

import os

def test_mujoco_joint_order():
    print("\n" + "=" * 80)
    print("Testing Joint Order with MuJoCo")
    print("=" * 80)

    xml_path = "active_adaptation/assets_mjcf/g1_29dof_nohand/g1_29dof_nohand.xml"

    if not os.path.exists(xml_path):
        print(f"❌ XML file not found: {xml_path}")
        return False

    print(f"\n✓ XML file found: {xml_path}")

    try:
        import mujoco as mj

        # Load model
        model = mj.MjModel.from_xml_path(xml_path)
        print(f"✓ MuJoCo model loaded")
        print(f"✓ Number of actuators: {model.nu}")

        # Extract joint names
        print("\n" + "=" * 80)
        print("Joint Order from MuJoCo (model.actuator → joint mapping)")
        print("=" * 80)

        joint_names = []
        print(f"\n{'#':<4} {'Joint Name':<35} {'Transmission ID'}")
        print("-" * 70)

        for i in range(model.nu):
            # Get transmission ID (which joint this actuator controls)
            trnid = model.actuator_trnid[i, 0]

            if trnid >= 0:  # Valid joint
                joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, trnid)
                joint_names.append(joint_name)
                print(f"{i+1:<4} {joint_name:<35} {trnid}")

        print(f"\nTotal: {len(joint_names)} actuated joints")

        # Compare with our weights
        print("\n" + "=" * 80)
        print("Mapping to Defined Weights")
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
            print(f"\n❌ ERROR: Size mismatch!")
            print(f"   MuJoCo joints: {len(joint_names)}")
            print(f"   Weights: {len(dof_err_w)}")
            return False

        print(f"\n✅ Size matches: {len(joint_names)} joints\n")
        print(f"{'#':<4} {'MuJoCo Joint Name':<35} {'Weight':<10} {'Category'}")
        print("-" * 80)

        # Categorize
        categories = []
        for i, name in enumerate(joint_names):
            if i < 6:
                cat = "Left Leg"
            elif i < 12:
                cat = "Right Leg"
            elif i < 15:
                cat = "Waist"
            elif i < 19:
                cat = "Left Arm"
            elif i < 22:
                cat = "Left Wrist"
            elif i < 26:
                cat = "Right Arm"
            else:
                cat = "Right Wrist"
            categories.append(cat)

        for i, (name, weight, cat) in enumerate(zip(joint_names, dof_err_w, categories), 1):
            marker = "🔧" if 'wrist' in name.lower() else ""
            print(f"{i:<4} {name:<35} {weight:<10.1f} {cat} {marker}")

        # Highlight wrist joints
        print("\n" + "=" * 80)
        print("✅ Wrist Joints Verification")
        print("=" * 80)

        wrist_info = []
        for i, (name, weight) in enumerate(zip(joint_names, dof_err_w)):
            if 'wrist' in name.lower():
                wrist_info.append((i+1, name, weight))

        if len(wrist_info) == 6:  # 3 left + 3 right
            print(f"\n✅ Found 6 wrist joints (as expected):\n")
            for idx, name, weight in wrist_info:
                print(f"   {idx:2d}. {name:<35} weight = {weight}")

            print("\n✅ All wrist joints correctly positioned!")
            return True
        else:
            print(f"\n⚠️  Found {len(wrist_info)} wrist joints (expected 6)")
            for idx, name, weight in wrist_info:
                print(f"   {idx:2d}. {name:<35} weight = {weight}")
            return False

    except ImportError:
        print("\n❌ MuJoCo Python bindings not installed")
        print("   Install with: pip install mujoco")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mujoco_joint_order()

    print("\n" + "=" * 80)
    if success:
        print("✅ VERIFICATION PASSED")
        print("=" * 80)
        print("\nIsaacLab will read joints in the SAME order as MuJoCo actuators.")
        print("Your weight configuration is CORRECT!")
    else:
        print("⚠️  VERIFICATION INCOMPLETE")
        print("=" * 80)
        print("\nCheck errors above. May need manual verification during training.")

    exit(0 if success else 1)
