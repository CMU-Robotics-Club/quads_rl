# Quadrupeds Reinforcement Learning!

This repo contains training code for running and testing reinforcement learning.
This README is written and verified by humans. So please read this carefully!

# Setup

We'll put our code in /Documents since that's what I did

Run every command here sequentially. Some installations may take up to a few minutes:

```
mkdir -p ~/Documents/laika_rl
cd ~/Documents/laika_rl
git clone https://github.com/isaac-sim/IsaacLab.git
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
```

Your environment is now set up! Now let's make some aliases to make our life easier:

```
alias ish='~/Documents/laika_FL/IsaacLab/isaaclab.sh'
alias isact='. ~/Documents/laika_rl/IsaacLab/env_isaaclab/bin/activate'
alias iltrain='cd ~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping && ish -p scripts/rsl_rl/train.py --task Leg-Jumping-v0 --headless && cd -'
alias ilplay='ish -p ~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping/scripts/rsl_rl/play.py --task Leg-Jumping-v0 --num_envs 1'
```

Now run iltrain to train, and ilplay to play the most recent trained policy
