"""TWIST command managers."""

# Import command manager
from .command import *

# Import TWIST-specific observations
from .observations import *

# Import TWIST-specific terminations
from .terminations import *

# Import only the _twist suffix reward classes from rewards_new to avoid conflicts
# The tracking rewards (keypoint_pos_tracking_local_product, etc.) are already
# defined in rewards.py and will be used from there
from .rewards_new import (
    # Regularization rewards (modified)
    feet_slip_twist,
    feet_contact_forces_twist,
    feet_stumble_twist,
    joint_pos_limits_twist,
    joint_torque_limits_twist,
    dof_vel_twist,
    dof_acc_twist,
    action_rate_l2_twist,
    feet_air_time_twist,
    ang_vel_xy_twist,
    ankle_dof_acc_twist,
    ankle_dof_vel_twist,

    # TWIST-aligned tracking rewards (new)
    tracking_root_pose_twist_aligned,
    tracking_root_vel_twist_aligned,
    tracking_joint_dof_twist_aligned,
    tracking_joint_vel_twist_aligned,
    tracking_keybody_pos_twist_aligned,
)
