# Quadrupeds Reinforcement Learning!

## This repo contains training code for running and testing reinforcement learning.
## This README is written and verified by humans. So please read this carefully!

# Setup

We'll put our code in ~/Documents since that's what I did

Run every command here sequentially. Some installations may take up to a few minutes:

```
mkdir -p ~/Documents/laika_rl
cd ~/Documents/laika_rl
git clone -b main https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

uv venv env_isaaclab --python 3.11 --seed
source env_isaaclab/bin/activate
export UV_HTTP_TIMEOUT=600
uv pip install "isaacsim[all,extscache]==5.1.0" \
       --extra-index-url https://pypi.nvidia.com \
       --index-strategy unsafe-best-match
uv pip install "setuptools<82" wheel
uv pip install --no-build-isolation "flatdict==4.0.1"
uv pip install "numpy==1.26.4"
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --timeout 1000
uv pip install --no-build-isolation \
    -e source/isaaclab \
    -e source/isaaclab_assets \
    -e source/isaaclab_tasks \
    -e source/isaaclab_rl \
    -e source/isaaclab_mimic \
    rsl-rl-lib

cd ~/Documents/laika_rl
git clone https://github.com/CMU-Robotics-Club/quads_rl leg_jumping_rl/Leg_Jumping
cd leg_jumping_rl/Leg_Jumping
uv pip install -e source/*

mkdir -p "$TMPDIR"
```

Your environment is now set up! Now let's make some aliases to make our life easier. Add these to your `~/.bashrc` or `~/.zshrc`:

```
export TMPDIR="/tmp/$USER"
alias ish='~/Documents/laika_rl/IsaacLab/isaaclab.sh'
alias isact='. ~/Documents/laika_rl/IsaacLab/env_isaaclab/bin/activate'
alias iltrain='cd ~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping && ish -p scripts/rsl_rl/train.py --task Leg-Jumping-v0 --headless && cd -'
alias ilplay='ish -p ~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping/scripts/rsl_rl/play.py --task Leg-Jumping-v0 --num_envs 1'
alias urdf2usd='cd ~/Documents/laika_rl/ && ish -p IsaacLab/scripts/tools/convert_urdf.py leg_jumping_rl/Leg_Jumping/source/Leg_Jumping/Leg_Jumping/data/sliding_leg.urdf leg_jumping_rl/Leg_Jumping/source/Leg_Jumping/Leg_Jumping/data/sliding_leg.usd --fix-base --joint-stiffness 0.0 --joint-damping 0.0 && cd -'
```

IMPORTANT: run `isact` before running anything to source your environment,

Now run `iltrain` to train leg jumping, and `ilplay` to play the most recent trained policy

# Understanding the repo

Prereq: Do the above setup.

The repo has a ton of subdirectories, most of which are boilerplate fluff. Navigate to this repo's root will be at `cd ~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping/`. You'll never have to go to any other directory.

Navigate to `source/Leg_Jumping/Leg_Jumping/data`. Here, you can find our urdf files and usd files. 

URDF files are imported from the ROS project, and is easy to interchange. However, isaac lab doesn't use URDF, but rather USD to describe the robot. But good news: isaac lab has a built-in way of converting directly from urdf to usd! just run `urdf2usd`, which is an alias we created above

Now navigate to `source/Leg_Jumping/Leg_Jumping/tasks/manager_based`. Here, you can see `dog_walking` and `leg_jumping` (self-explanatory). We'll dissect `leg_jumping`, but `dog_walking` is just the same thing with different configs. 
Go into `leg_jumping`. We'll explore some files: 
- `leg_asset.py` contains the path to the USD file for the sliding leg, global simulator configurations, default positions for the leg, and actuator PD gains. Notice how slider_passive is an actuator with 0 PD gains. This lets the joint slide around without any external behavior. 
- `leg_jumping_env_cfg.py`: This is the MEAT of the RL. We have a bunch of classes, each handling different parts of the RL. Here are the classes and what they do:
    - `LegJumpingSceneCfg`: Just sets up the environment (ground, lighting, etc)
    - `ActionsCfg`: Sets up what the RL outputs. In this case, we output positions for ["hip_joint", "knee_joint"]. Notice how we define it via mdp.JointPositionActionCfg.
    - `ObservationsCfg`: Sets up what the RL takes in/observes. We have `joint_pos_rel` (position of the joints relative to the default position from `leg_asset.py`), `joint_vel_rel` (joint velocities), and `last_action` (the action the RL outputted in the previous timestep).
    - `EventCfg`: This is where we define domain randomization. Each environment can get random actuator gains, friction, mass perturbations of links, etc.
    - `RewardsCfg`: This is a pretty chunky one. We have a bunch of rewards defined here. Each has a `weight` attribute that is carefully tuned. Notice how we have functions like `mdp.target_above_threshold`. Some are in the default mdp library, but some (actually, many!) are custom defined in `mdp/rewards.py`. 
    - `TerminationsCfg`: Defines when an environment terminates and the next episode begins. We always have `time_out` (we don't want an environment to keep running forever). 
    - `LegJumpingEnvCfg`: We bring all the above classes together into one single config class.
        - We define the simulation timestep interval AND the policy frequency. 
- `agents/rsl_rl_ppo_cfg.py`: Here, we define the neural network for the policy, num iterations, hyperparameters for PPO, checkpoint policy save interval, etc.

Phew, that was a lot. When you run `iltrain`, you can see some statistics for how the rewards change and which ones contribute a lot. Try running it and understanding each part of what it prints out.
