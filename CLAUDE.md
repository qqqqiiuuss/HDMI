# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HDMI (Humanoid Demonstration and Motion Imitation)** is a framework for training humanoid robots to perform whole-body interaction tasks by learning from human demonstration videos. The project enables G1 humanoid robots to acquire diverse manipulation skills like moving suitcases, pushing boxes, opening doors, and carrying objects.

This codebase integrates:
- **IsaacLab** (formerly Isaac Gym) for GPU-accelerated physics simulation
- **PPO-ROA** (PPO with Rapid Online Adaptation) for reinforcement learning
- **Teacher-Student architecture** with privilege distillation for sim-to-real transfer
- **Motion retargeting** from human SMPL-X/BVH data to robot joint trajectories

## Installation

### Environment Setup
```bash
# Create conda environment
conda create -n hdmi python=3.11
conda activate hdmi

# Install IsaacSim
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
isaacsim  # Test installation

# Install IsaacLab
cd ..
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh -i none

# Install HDMI
cd ../HDMI-main
pip install -e .
```

### Dependencies
Key packages from `setup.py`:
- `torch==2.7.0`, `torchrl==0.7.0`, `tensordict==0.7.0`
- `hydra-core`, `omegaconf` (configuration)
- `wandb` (experiment tracking)
- `mujoco` (physics engine)
- `moviepy`, `imageio` (video recording)

## Common Commands

### Training

#### Teacher Policy (Stage 1)
Train a teacher policy with privileged information:
```bash
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase
```

#### Student Policy (Stage 2)
Fine-tune student policy for deployment:
```bash
python scripts/train.py \
    algo=ppo_roa_finetune \
    task=G1/hdmi/move_suitcase \
    checkpoint_path=run:<teacher_wandb_run_path>
```

### Evaluation

Evaluate a trained policy:
```bash
python scripts/play.py \
    algo=ppo_roa_finetune \
    task=G1/hdmi/move_suitcase \
    checkpoint_path=run:<wandb_run_path>
```

### Motion Retargeting

Convert human motion data to robot format:
```bash
# Using GMR submodule
cd GMR
python scripts/smplx_to_robot.py --smplx_file <path> --robot g1 --save_path <output.pkl>

# Convert PKL to HDMI NPZ format (in project root)
cd ..
python convert_pkl_to_npz.py \
    --pkl_path GMR/output/robot_motion.pkl \
    --output_dir data/motion/g1/converted_motion \
    --robot_name unitree_g1
```

### Configuration

Override config parameters:
```bash
python scripts/train.py \
    task=G1/hdmi/push_box \
    algo.lr=3e-4 \
    total_frames=200_000_000 \
    wandb.mode=disabled
```

## Code Architecture

### High-Level Structure

```
active_adaptation/          # Core RL framework
├── envs/                   # Environment definitions
│   ├── base.py            # Base environment class
│   ├── humanoid.py        # Humanoid-specific environment
│   ├── locomotion.py      # Locomotion tasks
│   └── mdp/               # MDP components (observations, rewards, commands)
├── learning/              # RL algorithms
│   ├── ppo/
│   │   ├── ppo_roa.py    # Main PPO-ROA implementation (Teacher-Student)
│   │   ├── ppo.py        # Standard PPO
│   │   └── ppo_amp.py    # Adversarial Motion Priors
│   └── modules/           # Neural network modules
└── utils/
    └── motion.py          # Motion data loading and processing

cfg/                       # Hydra configuration files
├── algo/                  # Algorithm configs (registered in code, not files)
├── task/                  # Task-specific configs
│   └── G1/hdmi/          # G1 humanoid HDMI tasks
└── train.yaml            # Main training config

scripts/
├── train.py              # Training entry point
├── play.py               # Evaluation/visualization
└── helpers.py            # Environment and policy creation utilities

data/motion/              # Motion datasets
├── g1/                   # G1-specific motion data (NPZ format)
├── lafan/                # LAFAN1 BVH dataset
└── data_for_sim/         # Simulation-ready datasets

GMR/                      # General Motion Retargeting submodule
```

### PPO-ROA Architecture (Teacher-Student)

The core learning algorithm is in `active_adaptation/learning/ppo/ppo_roa.py`. See `PPO_ROA_Architecture_Analysis.md` for detailed Chinese documentation.

**Key concepts:**

1. **Teacher (Training Phase)**:
   - Uses `encoder_priv` to encode privileged information (terrain, object properties) → `PRIV_FEATURE_KEY`
   - Policy `actor` takes `[CMD, OBS, PRIV_FEATURE]` → action
   - Trained with PPO

2. **Student (Deployment Phase)**:
   - Uses `adapt_module` (MLP or GRU) to **infer** privileged features from observation history → `PRIV_PRED_KEY`
   - Policy `actor_adapt` takes `[CMD, OBS, PRIV_PRED]` → action
   - Trained first by **distillation** (imitating teacher), then **PPO fine-tuning**

3. **Shared Components**:
   - `critic`: Shared value network (uses privileged info during training)
   - `adapt_ema`: EMA version of `adapt_module` for stable deployment

**Training Stages:**
- `ppo_roa_train` (phase="train"): Train teacher + student via distillation
- `ppo_roa_finetune` (phase="finetune"): Fine-tune student with PPO (no teacher)

**Algorithm configs are registered in code** (`ppo_roa.py` lines 88-93), not as YAML files in `cfg/algo/`.

### Environment System

Environments extend `active_adaptation.envs.base._Env`:
- **Observation**: Built from MDP observation functions in `envs/mdp/observations.py`
- **Reward**: Composed from reward terms in `envs/mdp/rewards.py`
- **Commands**: Generated by command managers in `envs/mdp/commands.py`

Key environment classes:
- `HumanoidEnv` (`envs/humanoid.py`): Base humanoid tracking environment
- `LocomotionEnv` (`envs/locomotion.py`): Locomotion-specific tasks

### Motion Data Format

**HDMI NPZ format** (used by `MotionDataset` in `utils/motion.py`):
```python
motion.npz:
  - body_pos_w: [T, N_bodies, 3]     # World-frame body positions
  - body_quat_w: [T, N_bodies, 4]    # World-frame quaternions (wxyz)
  - joint_pos: [T, N_joints]         # Joint positions
  - body_lin_vel_w: [T, N_bodies, 3] # Linear velocities
  - body_ang_vel_w: [T, N_bodies, 3] # Angular velocities
  - joint_vel: [T, N_joints]         # Joint velocities

meta.json:
  - body_names: [...]  # Body link names
  - joint_names: [...] # Joint names
  - fps: 50.0
```

**GMR PKL format** (from motion retargeting):
- Convert using `convert_pkl_to_npz.py` (see `README_conversion.md`)

### Configuration System

Uses **Hydra** with structured configs:
- Main entry: `cfg/train.yaml`
- Task configs: `cfg/task/G1/hdmi/*.yaml` (inherit from `base/hdmi-base.yaml`)
- Algorithm configs: **Registered in `ppo_roa.py`**, not as files

Config resolution order: `defaults` → task config → CLI overrides

## Key Technical Details

### Coordinate Systems

See `coordinate_system_explanation.md` and `joint_rotation_explanation.md` for details on:
- Quaternion conventions (wxyz vs xyzw)
- World vs local coordinate frames
- Joint angle definitions

### WandB Integration

Checkpoints are loaded via WandB:
```bash
checkpoint_path=run:<project>/<run_id>  # e.g., run:hdmi/abc123xyz
```

Helper function in `utils/wandb.py` parses paths and downloads checkpoints.

### Training Hyperparameters

Default settings (from `ppo_roa.py`):
- `train_every=32`: Steps between policy updates
- `lr=3e-4`: Learning rate
- `gamma=0.99`: Discount factor
- `num_epochs=5`: PPO epochs per update
- `minibatch_size=4096`: Minibatch size

### Available Tasks

Example tasks in `cfg/task/G1/hdmi/`:
- `move_suitcase`: Drag a suitcase
- `push_box`, `move_foam`: Object manipulation
- `open_door-feet`, `push_door-hand`: Door interaction
- `carry_box_over_shoulder`: Lifting and carrying
- `roll_ball-hand`: Ball rolling

Each task config specifies:
- Robot type (e.g., `g1_29dof_rubberhand-feet_sphere-eef_box`)
- Motion dataset path
- Observation/reward function weights
- Episode length

## Development Workflow

1. **Prepare motion data**: Retarget human motion using GMR → Convert to NPZ
2. **Configure task**: Create/modify YAML in `cfg/task/G1/hdmi/`
3. **Train teacher**: `python scripts/train.py algo=ppo_roa_train task=...`
4. **Monitor on WandB**: Check `project=hdmi`
5. **Fine-tune student**: `python scripts/train.py algo=ppo_roa_finetune checkpoint_path=run:...`
6. **Evaluate**: `python scripts/play.py ...`

## Related Subprojects

- **GMR/** (`general_motion_retargeting`): Standalone motion retargeting toolkit (see `GMR/CLAUDE.md`)
- **basketball-main/**: Basketball-specific task implementation
- **loco-mujoco/**: MuJoCo locomotion environments
- **sim2real/**: Sim-to-real deployment code (TODO)

## Important Notes

- **IsaacLab dependency**: Requires IsaacSim 5.0.0 and IsaacLab installed separately
- **GPU required**: Training uses GPU-accelerated simulation (CUDA)
- **WandB account**: Required for checkpoint management and logging
- **Motion data**: Not included in repo; must be generated or obtained separately
