import os
from isaaclab.actuators import IdealPDActuatorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

SLIDING_LEG_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",

    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.expanduser("~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping/source/Leg_Jumping/Leg_Jumping/data/sliding_leg.usd"),

        # May or may not be useful
        activate_contact_sensors=True,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # NOTE: self collisions disabled for stability
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # TODO: need to set these right
            "slider_to_base": -0.4, # m
            "hip_joint": 0.4, # Rad
            "knee_joint": 1.1, # rad
        },
        pos=(0.0, 0.0, 0.0), # Start the robot at ground level z=0, and at (0,0) with respect to its scene
        joint_vel={".*": 0.0}, # Start everything at velocity 0
    ),

    soft_joint_pos_limit_factor=0.8, # Reduces physical limits by a factor for safety

    actuators = {
        # "leg_actuators": ImplicitActuatorCfg(
        #     joint_names_expr=["hip_joint", "knee_joint"],
        #     effort_limit_sim={
        #         "hip_joint": 15.0, # Nm
        #         "knee_joint": 15.0, # Nm
        #     },
        #     # TODO: Make these match real pd gains
        #     velocity_limit_sim=100.0, # rad/s
        #     stiffness={
        #         "hip_joint": 30.0,  # Nm/rad
        #         "knee_joint": 30.0, # Nm/rad
        #     },
        #     damping={
        #         "hip_joint": 2.0, # Nm/rad
        #         "knee_joint": 2.0, # Nm/rad
        #     },
        # ),

        "leg_actuators": DelayedPDActuatorCfg(
            joint_names_expr=["hip_joint", "knee_joint"],
            # Note: For explicit actuators in Isaac Lab, use 'effort_limit' and 'velocity_limit'
            effort_limit=15.0,        # Nm
            velocity_limit=100.0,     # rad/s
            stiffness={
                "hip_joint": 30.0,    # Nm/rad
                "knee_joint": 30.0,   # Nm/rad
            },
            damping={
                "hip_joint": 2.0,     # Nm/(rad/s)
                "knee_joint": 2.0,    # Nm/(rad/s)
            },
            # Delays are specified in physics steps (sim.dt = 1/240 s ≈ 4.16 ms)
            # Decimation = 4 (1 policy step = 4 physics steps = 16.67 ms)
            min_delay=0,  # 0 physics steps (0 ms)
            max_delay=8,  # 8 physics steps (~33.3 ms = 2 policy steps)
        ),


        "slider_passive": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_base"],
            effort_limit_sim=0.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        )
    },
)



