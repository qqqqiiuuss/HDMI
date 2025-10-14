import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg, Articulation
from isaaclab.assets.rigid_object import RigidObjectCfg
import torch

ASSET_PATH = os.path.dirname(__file__)

class CustomArticulation(Articulation):
    def _create_buffers(self):
        super()._create_buffers()
        self._custom_friction = torch.zeros((self.num_instances,), device=self.device)
        self._custom_damping = torch.zeros((self.num_instances,), device=self.device)
        assert len(self.joint_names) == 1, "DoorArticulation should have exactly one joint."
        self.custom_joint_id = 0
        self.custom_torques = torch.zeros((self.num_instances,), device=self.device)
    
    def _initialize_impl(self):
        super()._initialize_impl()

        # set joint stiffness and damping to 0
        joint_attrs_zero = torch.zeros((self.num_instances, self.num_joints), device=self.device)
        self.write_joint_stiffness_to_sim(joint_attrs_zero)
        self.write_joint_damping_to_sim(joint_attrs_zero)
        self.write_joint_friction_coefficient_to_sim(joint_attrs_zero)

        # set actuator stiffness and damping to 0
        for actuator in self.actuators.values():
            actuator.stiffness.fill_(0.0)
            actuator.damping.fill_(0.0)
    
    def write_data_to_sim(self):
        j_vel = self.data.joint_vel[:, self.custom_joint_id]
        j_friction = -torch.sign(j_vel) * (j_vel.abs() > 0.01) * self._custom_friction
        j_damping = -j_vel * self._custom_damping
        self.custom_torques[:] = j_friction + j_damping
        
        self.set_joint_effort_target(self.custom_torques.unsqueeze(-1), joint_ids=[self.custom_joint_id])
        super().write_data_to_sim()

DOOR_CFG = ArticulationCfg(
    class_type=CustomArticulation,
    prim_path="{ENV_REGEX_NS}/Door",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/door/door.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            enabled_self_collisions=False
        )
    ),
    actuators={
        "door_joint": IdealPDActuatorCfg(
            joint_names_expr="door_joint",
            # will be randomized
            stiffness=0.0, 
            damping=0.0,
            friction=0.0,
            effort_limit=100.0,
            velocity_limit=20.0,
        )
    },
)

BOX_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Box",
    # spawn=sim_utils.UrdfFileCfg(
    #     asset_path=f"{ASSET_PATH}/box/box.urdf",
    spawn=sim_utils.UsdFileCfg(
        scale=(1.0, 1.0, 1.0),
        usd_path=f"{ASSET_PATH}/objects/box/box.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=5.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
)
BOX_SMALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/box_small",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/box_small/box_small.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,
        ),
    ),

)

# from active_adaptation.assets.spawn import clone
# spawn_func = BOX_CFG.spawn.func.__wrapped__
# BOX_CFG.spawn.func = clone(spawn_func)
# BOX_CFG.spawn.scale_range = (0.9, 1.1)

# size: [1.0 0.8 0.8] z: 0.6 - 0.8
# friction 0.5-1.0
# mass 4-8

SUITCASE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/suitcase",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/suitcase/suitcase.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,
        ),
    ),
)
FOAM_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/foam",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}" + "/objects/foam/foam.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,
        ),
    ),
)
STOOL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/stool",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}" + "/objects/stool/stool.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,
        ),
    ),
)
STOOL_LOW_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/stool",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}" + "/objects/stool/stool_low.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,
        ),
    ),
)
STOOL_LOW2_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/stool",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/stool/stool-low2.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,
        ),
    ),
)
STAIR_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/stair",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/stair/stair.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=20.0,
        ),
    ),
)
SUPPORT0_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/support0",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/support/support0.usd",
        activate_contact_sensors=False,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    ),
)
SUPPORT1_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/support1",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/support/support1.usd",
        activate_contact_sensors=False,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    ),
)
STOOL_SUPPORT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/stool_support",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/stool/stool_support.usd",
        activate_contact_sensors=False,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    ),
)

WALL0_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/wall0",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/platform/wall0.usd",
        activate_contact_sensors=False,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    ),
)
PLATFORM0_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/platform0",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/platform/platform0.usd",
        activate_contact_sensors=False,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    ),
)
PLATFORM1_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/platform1",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/platform/platform1.usd",
        activate_contact_sensors=False,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.5,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    ),
)
BREAD_BOX_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/bread_box",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/bread_box/bread_box.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=3.0,
        ),
    ),
)
WOOD_BOARD_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/wood_board",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/wood_board/wood_board.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=3.0,
        ),
    ),
)
BALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ball",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/ball/ball.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=3.0,
        ),
    ),
)

FOLDCHAIR_CFG = ArticulationCfg(
    class_type=CustomArticulation,
    prim_path="{ENV_REGEX_NS}/foldchair",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/foldchair/foldchair.usd",
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            enabled_self_collisions=False,
            fix_root_link=True
        ),
    ),
    actuators={
        "foldchair_joint": IdealPDActuatorCfg(
            joint_names_expr="foldchair_joint",
            # will be randomized
            stiffness=0.0, 
            damping=1.0,
            friction=0.0,
            effort_limit=100.0,
            velocity_limit=20.0,
        )
    },
)