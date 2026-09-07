import os
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

DOG_LOC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",

    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.expanduser("~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping/source/Leg_Jumping/Leg_Jumping/data/full_dog.usd"),

        # May or may not be useful
        activate_contact_sensors=True,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=3.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # NOTE: self collisions disabled for stability
            enabled_self_collisions=False,
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=2
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # TODO: need to set these right
            # "fl_lat_joint": -0.1, # Rad # Negative for lat is outwards
            # "fl_hip_joint": 0.4, # Rad
            # "fl_knee_joint": 1.1, # rad
            # "fr_lat_joint": -0.1, # Rad
            # "fr_hip_joint": 0.4, # Rad
            # "fr_knee_joint": 1.1, # rad
            # "br_lat_joint": -0.1, # Rad
            # "br_hip_joint": 0.4, # Rad
            # "br_knee_joint": 1.1, # rad
            # "bl_lat_joint": -0.1, # Rad
            # "bl_hip_joint": 0.4, # Rad
            # "bl_knee_joint": 1.1, # rad
            "fl_lat_joint": 0.0, # Rad
            "fl_hip_joint": 0.4, # Rad
            "fl_knee_joint": 1.1, # rad
            "fr_lat_joint": 0.0, # Rad
            "fr_hip_joint": 0.4, # Rad
            "fr_knee_joint": 1.1, # rad
            "br_lat_joint": 0.0, # Rad
            "br_hip_joint": 0.4, # Rad
            "br_knee_joint": 1.1, # rad
            "bl_lat_joint": 0.0, # Rad
            "bl_hip_joint": 0.4, # Rad
            "bl_knee_joint": 1.1, # rad
            # "fl_lat_joint": 0.0, # Rad
            # "fl_hip_joint": 0.4, # Rad
            # "fl_knee_joint": 0.6, # rad
            # "fr_lat_joint": 0.0, # Rad
            # "fr_hip_joint": 0.4, # Rad
            # "fr_knee_joint": 0.6, # rad
            # "br_lat_joint": 0.0, # Rad
            # "br_hip_joint": 0.4, # Rad
            # "br_knee_joint": 0.6, # rad
            # "bl_lat_joint": 0.0, # Rad
            # "bl_hip_joint": 0.4, # Rad
            # "bl_knee_joint": 0.6, # rad
        },
        pos=(0.0, 0.0, 0.0), # Start the robot at ground level and at (0,0) with respect to its scene
        joint_vel={".*": 0.0}, # Start everything at velocity 0
    ),

    soft_joint_pos_limit_factor=0.8, # Reduces physical limits by a factor for safety

    actuators = {
        "leg_actuators": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=70.0, # Nm
            # TODO: Make these match real pd gains
            velocity_limit_sim=30.0, # rad/s
            stiffness=30,
            damping=2,
        )
    },
)



