# Quadrupeds Reinforcement Learning!

## This repo contains training code for running and testing reinforcement learning.
## This README is written and verified by humans. So please read this carefully!

# Setup

Before we start, type `bash` in your terminal and press enter just in case you're not in bash.

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
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
uv pip install "setuptools<82" wheel
uv pip install --no-build-isolation "flatdict==4.0.1"
uv pip install "numpy==1.26.4"
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --timeout 1000
uv pip install --no-build-isolation -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks -e source/isaaclab_rl -e source/isaaclab_mimic rsl-rl-lib

cd ~/Documents/laika_rl
git clone https://github.com/CMU-Robotics-Club/quads_rl leg_jumping_rl/Leg_Jumping
cd leg_jumping_rl/Leg_Jumping
uv pip install -e source/*
```

Until this hopper is on `main`, clone or checkout the **PR branch** for this repo (plain `main` will not have the shipped policy).

Your environment is now set up! Now let's make some aliases to make our life easier. Add these to the end of your `~/.bashrc` or `~/.zshrc`:

```
export TMPDIR="/tmp/$USER"
mkdir -p "$TMPDIR"
alias ish='~/Documents/laika_rl/IsaacLab/isaaclab.sh'
alias isact='. ~/Documents/laika_rl/IsaacLab/env_isaaclab/bin/activate'
alias iltrain='cd ~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping && LEG_PRESET=A_UNDER ish -p scripts/rsl_rl/train.py --task Leg-Jumping-v0 --headless && cd -'
alias ilplay='ish -p ~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping/scripts/rsl_rl/play.py --task Leg-Jumping-v0 --num_envs 1'
alias urdf2usd='cd ~/Documents/laika_rl/ && ish -p IsaacLab/scripts/tools/convert_urdf.py leg_jumping_rl/Leg_Jumping/source/Leg_Jumping/Leg_Jumping/data/sliding_leg.urdf leg_jumping_rl/Leg_Jumping/source/Leg_Jumping/Leg_Jumping/data/sliding_leg.usd --fix-base --joint-stiffness 0.0 --joint-damping 0.0 && cd -'
```

Now, in your terminal, run this:

```
source ~/.bashrc
```

We're done!

IMPORTANT: run `isact` before running anything to source your environment.

Play needs a display (see **Display options**). Hopper commands are under **Leg jumping**. Bare `ilplay` (no `--checkpoint`) is not the demo — that would pick a local `logs/` dump.

## Display options

<details>
<summary>For Macs</summary>

- **In person.** At the machine's monitors: `isact`, `cd ~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping`, then the play command in **Leg jumping**.
- **X11 forwarding.** Install [XQuartz](https://www.xquartz.org/), then log out of the Mac and back in. In a new Terminal window, `echo $DISPLAY` should contain `org.xquartz`. Then `ssh -Y <user>@<host>`, check `DISPLAY` is set, `isact`, `cd` to this repo, and the same play command. The window will be choppy. X11 copies each frame to the Mac, and the sim waits on it.
- **WebRTC.** This stays smooth and still takes mouse and keyboard. Download **Isaac Sim WebRTC Streaming Client 1.1.5** from the [Isaac Sim 5.1.0 downloads](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html) (`macOS (arm64)` on Apple Silicon). Newer clients belong to newer Isaac Sim versions and will not connect. On the machine, add this function below your aliases, then `source ~/.bashrc`:

  ```
  ilstream() {
    local cudart
    cudart="$(python -c 'import pathlib, nvidia.cuda_runtime as c; print(pathlib.Path(c.__file__).resolve().parent / "lib" / "libcudart.so.12")')"
    if [ $# -eq 0 ]; then
      set -- ~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping/scripts/rsl_rl/play.py --task Leg-Jumping-v0 --num_envs 1
    fi
    LD_PRELOAD="$cudart" PUBLIC_IP="$(hostname -I | awk '{print $1}')" ish -p "$@" --livestream 1
  }
  ```

  `ilstream` is the function above. It is not one of the shared aliases. On the machine: `isact`, `cd` to this repo, `hostname -I` (first address is what the stream advertises), then the **stream** command in **Leg jumping**. Wait until the terminal prints `Loading model checkpoint`. In the Mac client, replace `127.0.0.1` in the Server field with that address and click Connect. Quit with **File → Exit** in the streamed window, then close the client. There is no password on the stream, so do not leave `ilstream` running. The Mac must be able to reach TCP port 49100 and UDP port 47998 (campus network or VPN).

</details>

# Leg jumping

One leg on a vertical rail. **Action 0 is the stand.** Play stands until you ask for a hop; train auto-requests so it can learn. Policy loop is 60 Hz (`dt` 1/240, decimation 4). Actor sees hip/knee (knee may be a slack encoder), velocities, last action, jump latch + time, last **10** steps. No contact, IMU, or rail on the actor.

In every hopper shell (`TERM=xterm-256color` so Isaac Lab’s launcher does not die on a dumb terminal):

```
export TERM=xterm-256color
export LEG_PRESET=A_UNDER
isact
cd ~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping
```

If you omit `LEG_PRESET`, you get `LEGACY` (old always-jump pose), not this hopper. `iltrain` in the aliases above already sets `A_UNDER`.

## Hardware numbers (sim → real)

**Edit one file when CAD / motors / the real stand arrive:**  
`source/Leg_Jumping/Leg_Jumping/tasks/manager_based/leg_jumping/leg_params.py`

Torque (N·m at the joint), kp/kd, command delay, stand pose, and dummy-link mass live **there**. Those kp/kd are the **center**; the env still randomizes them ±20% at startup. Pick a sheet with `LEG_PRESET` (`A_UNDER` = 15/15 N·m, foot under the hip, delay 0; `B_UNDER` = 10/6 N·m; `LEGACY` = old always-jump pose). Do not hunt through `leg_jumping_env_cfg.py` for those numbers. `RANGES` in the same file is a planned sweep; the env does not use it yet.

## Shipped policy

`source/Leg_Jumping/Leg_Jumping/data/policies/a_under_hop.pt` is **in git**. It is one network: idle stand **and** one hop (~4800 PPO steps, delay 0, 10-step history). Train dumps go in `logs/` (gitignored). After clone you will not have `logs/`. Do not play the newest file in `logs/` — extra training on this env often hops but fidgets after landing. Do not overwrite `a_under_hop.pt` unless you mean to change the demo.

## 1. Idle stand (done)

Same file as (2). Play **watch** mode (play.py default), **do not press J**. Under-hip stand (`A_UNDER`: hip 0.55, knee 1.13).

```
ilplay --checkpoint source/Leg_Jumping/Leg_Jumping/data/policies/a_under_hop.pt
```

Stream (needs the `ilstream` function in **Display options**):

```
ilstream scripts/rsl_rl/play.py --task Leg-Jumping-v0 --num_envs 1 \
  --checkpoint source/Leg_Jumping/Leg_Jumping/data/policies/a_under_hop.pt
```

## 2. One hop on command (done)

Same checkpoint as (1). After the picture is up: **J** in the stream, or **Enter** in the play terminal. One press → one hop (2 s lockout). **Space** pauses Isaac; it does not jump.

https://github.com/user-attachments/assets/0e86c63d-2381-4593-b6f0-af6f252ad0ea

The hop is trained on a **jump request**, not on a keyboard. Each policy step the actor sees two command values: `latch` (1 for 2 s after an accepted request, else 0) and time since that request. In train, a random idle timer sets request true. In play, J or Enter sets request true. On the real leg, a button or ROS message would set request true the same way. PPO is paid to hop once while `latch` is 1 and to sit while `latch` is 0. The net never observes J, Enter, or ROS; only `latch` and time. If `latch` stays 0, it only sits. Encoders replace sim joint angles; the net still outputs hip and knee position targets for the motor PD. No contact, IMU, or rail on the actor, so the bench does not need those sensors.

Train is headless and auto-hops (not watch). It writes a new `logs/` folder and does **not** replace `a_under_hop.pt`. Scratch `iltrain` is 2000 iters in the PPO cfg, not the shipped ~4800-step file. `--resume` only loads from `logs/…/model_*.pt`, not from `data/policies/`.

When you play the shipped file, the env already randomizes friction, kp/kd ±20%, mass ×0.95–1.05 on **hip / thigh / shin** only (not the 100 kg rail), knee backlash 0–3°, and knee encoder offset ±2°.

## 3. Continuous hopping (not done)

No checkpoint. `HeldRequests` (`mode=hold`) is in the code for later. Play does not bind a hold key, and nothing was trained for a hop burst. (1) and (2) will not machine-gun hops.

# Understanding the repo

Prereq: Do the above setup.

The repo has a ton of subdirectories, most of which are boilerplate fluff. Navigate to this repo's root will be at `cd ~/Documents/laika_rl/leg_jumping_rl/Leg_Jumping/`. You'll never have to go to any other directory.

Navigate to `source/Leg_Jumping/Leg_Jumping/data`. Here, you can find our urdf files and usd files, and `policies/a_under_hop.pt` (the shipped stand + one-hop weights; see **Leg jumping**).

URDF files are imported from the ROS project, and is easy to interchange. However, isaac lab doesn't use URDF, but rather USD to describe the robot. But good news: isaac lab has a built-in way of converting directly from urdf to usd! just run `urdf2usd`, which is an alias we created above

Now navigate to `source/Leg_Jumping/Leg_Jumping/tasks/manager_based`. Here, you can see `dog_walking` and `leg_jumping` (self-explanatory). We'll dissect `leg_jumping`. `dog_walking` is the same Isaac Lab layout with different configs; it is not this hopper.
Go into `leg_jumping`. We'll explore some files:
- `leg_params.py`: **sim-to-real sheet** (see **Hardware numbers** above). Torque, gains, delay, stand. Not the env cfg.
- `leg_asset.py` contains the path to the USD file for the sliding leg, global simulator configurations, default positions for the leg, and actuator PD gains. Notice how slider_passive is an actuator with 0 PD gains. This lets the joint slide around without any external behavior.
- `mdp/jump_command.py` and `mdp/commands.py`: jump timer. Modes `train` | `eval` | `watch` | `hold`. Play defaults to watch (one hop per J/Enter).
- `mdp/events.py`: knee backlash, encoder offset, phantom masses, stand reset.
- `leg_jumping_env_cfg.py`: This is the MEAT of the RL. We have a bunch of classes, each handling different parts of the RL. Here are the classes and what they do:
    - `LegJumpingSceneCfg`: Just sets up the environment (ground, lighting, etc)
    - `ActionsCfg`: Sets up what the RL outputs. In this case, we output positions for ["hip_joint", "knee_joint"]. Notice how we define it via mdp.JointPositionActionCfg. Outputs are relative to the stand (`use_default_offset=True`).
    - `ObservationsCfg`: Actor: hip/knee vs default (knee can be hysteresis + offset), joint velocities, last action, jump command, last 10 steps. Critic also gets slider pos/vel and airborne. No contact/IMU/rail on the actor.
    - `EventCfg`: Domain randomization: friction, kp/kd, mass on hip/thigh/shin only, knee backlash 0–3°, encoder offset ±2°.
    - `RewardsCfg`: Weights in this file, functions in `mdp/rewards.py`. `foot_air` and `torque_spike` are listed at weight 0 (leftover hops; they do not train).
    - `TerminationsCfg`: Defines when an environment terminates and the next episode begins. We always have `time_out` (we don't want an environment to keep running forever).
    - `LegJumpingEnvCfg`: We bring all the above classes together into one single config class.
        - We define the simulation timestep interval AND the policy frequency (60 Hz).
- `agents/rsl_rl_ppo_cfg.py`: Here, we define the neural network for the policy, num iterations, hyperparameters for PPO, checkpoint policy save interval, etc.

Phew, that was a lot. When you run `iltrain`, you can see some statistics for how the rewards change and which ones contribute a lot. Try running it and understanding each part of what it prints out.
