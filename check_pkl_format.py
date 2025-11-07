#!/usr/bin/env python3
"""
Check PKL format from TWIST training data
"""
import pickle
import sys

# Load a PKL file from TWIST training data
pkl_path = '/home/ubuntu/DATA2/Dataset-G1/AMASS_G1/accad/A7___crouch.pkl'

try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print('='*80)
    print('PKL FILE FORMAT CHECK (TWIST Training Data)')
    print('='*80)
    print(f'\nFile: {pkl_path}')
    print(f'\nKeys in PKL file: {list(data.keys())}')
    print()

    # Check joint names
    if 'joint_names' in data:
        print(f'Number of joints: {len(data["joint_names"])}')
        print('\nJoint names order:')
        for i, name in enumerate(data['joint_names']):
            print(f'  {i:2d}: {name}')
        print()

    # Check quaternion format
    if 'body_quat_w' in data:
        import numpy as np
        quat_shape = data['body_quat_w'].shape
        print(f'body_quat_w shape: {quat_shape}')
        print(f'  Format: [T={quat_shape[0]}, N_bodies={quat_shape[1]}, quat_dim={quat_shape[2]}]')

        # Check first frame root quaternion
        root_quat = data["body_quat_w"][0, 0]
        print(f'\nFirst frame root quaternion: {root_quat}')
        print(f'  Norm: {np.linalg.norm(root_quat):.6f} (should be ~1.0)')

        # Infer quaternion format
        if abs(root_quat[0]) < 0.1 and abs(np.linalg.norm(root_quat[1:]) - 1.0) < 0.1:
            print('  ** Likely xyzw format (w is small, xyz has norm ~1)')
        elif abs(root_quat[3]) < 0.1 and abs(np.linalg.norm(root_quat[:3]) - 1.0) < 0.1:
            print('  ** Likely wxyz format (w is small, xyz has norm ~1)')
        else:
            print(f'  ** Cannot determine format clearly')
        print()

    # Check root position
    if 'body_pos_w' in data:
        pos_shape = data['body_pos_w'].shape
        print(f'body_pos_w shape: {pos_shape}')
        print(f'  Format: [T={pos_shape[0]}, N_bodies={pos_shape[1]}, pos_dim={pos_shape[2]}]')
        print(f'First frame root position: {data["body_pos_w"][0, 0]}')
        print()

    # Check joint positions
    if 'joint_pos' in data:
        joint_pos_shape = data['joint_pos'].shape
        print(f'joint_pos shape: {joint_pos_shape}')
        print(f'  Format: [T={joint_pos_shape[0]}, N_joints={joint_pos_shape[1]}]')
        print(f'First frame joint positions (first 10):')
        print(f'  {data["joint_pos"][0, :10]}')
        print()

    print('='*80)
    print('COMPARISON WITH NPZ FORMAT')
    print('='*80)
    print('\nNPZ format (HDMI task):')
    print('  - 29 joints')
    print('  - body_quat_w: wxyz format (confirmed in meta.json and code)')
    print('  - Joint order: left_leg(6) + right_leg(6) + waist(3) + left_arm(3) + left_wrist(3) + right_arm(3) + right_wrist(3)')
    print()

    if 'joint_names' in data:
        if len(data['joint_names']) == 29:
            print('✓ PKL has same number of joints (29)')
        else:
            print(f'✗ PKL has DIFFERENT number of joints ({len(data["joint_names"])} vs 29)')

    print()

except Exception as e:
    print(f'Error loading PKL file: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
