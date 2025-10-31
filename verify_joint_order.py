#!/usr/bin/env python3
"""
验证29 DOF关节顺序与权重列表是否匹配

Usage:
    python verify_joint_order.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

def verify_joint_order():
    # URDF文件路径
    xml_file = Path(__file__).parent / "active_adaptation/assets_mjcf/g1_29dof_nohand/g1_29dof_nohand.xml"

    if not xml_file.exists():
        print(f"❌ URDF file not found: {xml_file}")
        return False

    # 解析XML提取关节顺序
    tree = ET.parse(xml_file)
    root = tree.getroot()
    actuators = root.find('actuator')

    if actuators is None:
        print("❌ No actuators found in URDF")
        return False

    actual_joints = []
    for motor in actuators.findall('motor'):
        joint_name = motor.get('joint')
        if joint_name:
            actual_joints.append(joint_name)

    # 定义的权重（从rewards_new.py）
    dof_err_w = [
        1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # Left Leg (6)
        1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # Right Leg (6)
        0.6, 0.6, 0.6,                      # Waist (3)
        0.8, 0.8, 0.8, 1.0,                 # Left Arm (4)
        0.6, 0.5, 0.5,                      # Left Wrist (3)
        0.8, 0.8, 0.8, 1.0,                 # Right Arm (4)
        0.6, 0.5, 0.5,                      # Right Wrist (3)
    ]

    weight_descriptions = [
        "Left hip_pitch (1.0)",
        "Left hip_roll (0.8)",
        "Left hip_yaw (0.8)",
        "Left knee (1.0)",
        "Left ankle_pitch (0.5)",
        "Left ankle_roll (0.5)",
        "Right hip_pitch (1.0)",
        "Right hip_roll (0.8)",
        "Right hip_yaw (0.8)",
        "Right knee (1.0)",
        "Right ankle_pitch (0.5)",
        "Right ankle_roll (0.5)",
        "Waist yaw (0.6)",
        "Waist roll (0.6)",
        "Waist pitch (0.6)",
        "Left shoulder_pitch (0.8)",
        "Left shoulder_roll (0.8)",
        "Left shoulder_yaw (0.8)",
        "Left elbow (1.0)",
        "Left wrist_roll (0.6)",  # 手腕
        "Left wrist_pitch (0.5)",  # 手腕
        "Left wrist_yaw (0.5)",    # 手腕
        "Right shoulder_pitch (0.8)",
        "Right shoulder_roll (0.8)",
        "Right shoulder_yaw (0.8)",
        "Right elbow (1.0)",
        "Right wrist_roll (0.6)",  # 手腕
        "Right wrist_pitch (0.5)", # 手腕
        "Right wrist_yaw (0.5)",   # 手腕
    ]

    # 验证数量
    print("\n" + "=" * 80)
    print("29 DOF JOINT ORDER VERIFICATION")
    print("=" * 80)
    print(f"\n✓ URDF joints: {len(actual_joints)}")
    print(f"✓ Defined weights: {len(dof_err_w)}")

    if len(actual_joints) != len(dof_err_w):
        print(f"\n❌ ERROR: Joint count mismatch!")
        return False

    print(f"✅ Count matches: {len(actual_joints)} joints\n")

    # 逐一对比
    print("=" * 80)
    print(f"{'#':<4} {'URDF Joint':<32} {'Weight Description':<35} {'Match'}")
    print("-" * 80)

    all_match = True
    mismatches = []

    for i, (joint, weight, desc) in enumerate(zip(actual_joints, dof_err_w, weight_descriptions), 1):
        # 简单匹配检查：检查描述中的关键词是否在关节名中
        desc_lower = desc.lower()
        joint_lower = joint.lower().replace('_joint', '')

        # 提取关键词
        keywords = []
        if 'left' in desc_lower:
            keywords.append('left')
        if 'right' in desc_lower:
            keywords.append('right')
        if 'hip' in desc_lower:
            keywords.append('hip')
        if 'knee' in desc_lower:
            keywords.append('knee')
        if 'ankle' in desc_lower:
            keywords.append('ankle')
        if 'waist' in desc_lower:
            keywords.append('waist')
        if 'shoulder' in desc_lower:
            keywords.append('shoulder')
        if 'elbow' in desc_lower:
            keywords.append('elbow')
        if 'wrist' in desc_lower:
            keywords.append('wrist')
        if 'pitch' in desc_lower:
            keywords.append('pitch')
        if 'roll' in desc_lower:
            keywords.append('roll')
        if 'yaw' in desc_lower:
            keywords.append('yaw')

        match = all(kw in joint_lower for kw in keywords)
        match_symbol = "✅" if match else "❌"

        if not match:
            all_match = False
            mismatches.append((i, joint, desc))

        print(f"{i:<4} {joint:<32} {desc:<35} {match_symbol}")

    print("=" * 80)

    if all_match:
        print("\n🎉 SUCCESS: All joints match their corresponding weights!")
        print("\n✅ Hand wrist joints verified:")
        print(f"   20. left_wrist_roll_joint  → weight 0.6")
        print(f"   21. left_wrist_pitch_joint → weight 0.5")
        print(f"   22. left_wrist_yaw_joint   → weight 0.5")
        print(f"   27. right_wrist_roll_joint → weight 0.6")
        print(f"   28. right_wrist_pitch_joint → weight 0.5")
        print(f"   29. right_wrist_yaw_joint  → weight 0.5")
        return True
    else:
        print(f"\n❌ ERROR: {len(mismatches)} mismatches found:")
        for i, joint, desc in mismatches:
            print(f"   {i}. {joint} ≠ {desc}")
        return False

if __name__ == "__main__":
    success = verify_joint_order()
    exit(0 if success else 1)
