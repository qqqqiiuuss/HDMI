import os

from .objects import *
from .g1 import *


ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "g1": G1_CYLINDER_CFG,
}

OBJECTS = {
    "door": DOOR_CFG,
    "box": BOX_CFG,
    "box_small": BOX_SMALL_CFG,
    "suitcase": SUITCASE_CFG,
    "stool": STOOL_CFG,
    "stool_low": STOOL_LOW_CFG,
    "ball": BALL_CFG,
    "foldchair": FOLDCHAIR_CFG,
    "stool_support": STOOL_SUPPORT_CFG,
    "foam": FOAM_CFG,
    "support0": SUPPORT0_CFG,
    "support1": SUPPORT1_CFG,
    # "trash_bin": TRASH_BIN_CFG,
    "stair": STAIR_CFG,
    "wood_board": WOOD_BOARD_CFG,
    "bread_box": BREAD_BOX_CFG,
    "wall0": WALL0_CFG,
    "platform0": PLATFORM0_CFG,
    "platform1": PLATFORM1_CFG,
}


def get_asset_meta(asset: Articulation):
    if not asset.is_initialized:
        raise RuntimeError("Articulation is not initialized. Please wait until `sim.reset` is called.")
    meta = {
        "init_state": asset.cfg.init_state.to_dict(),
        "body_names_isaac": asset.body_names,
        "joint_names_isaac": asset.joint_names,
        "actuators": {},
    }
    if asset.is_initialized: # parsed values
        meta["default_joint_pos"] = asset.data.default_joint_pos[0].tolist()
        meta["stiffness"] = asset.data.joint_stiffness[0].tolist()
        meta["damping"] = asset.data.joint_damping[0].tolist()

    for actuator_name, actuator in asset.actuators.items():
        meta["actuators"][actuator_name] = actuator.cfg.to_dict()
    return meta

