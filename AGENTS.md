# Repository Guidelines

## Project Structure & Module Organization
- **Core Package (`active_adaptation/`)**: Houses humanoid control logic, including environments under `envs/`, learning algorithms in `learning/`, and shared utilities in `utils/`. Extend sensors, policies, or task logic here.
- **Configuration (`cfg/`)**: Hydra YAMLs define algorithms, tasks, and experiment overrides. Duplicate an existing config and adjust only the deltas you need.
- **Execution Scripts (`scripts/`)**: Entry points for training (`train.py`, `train_sequential.py`), evaluation (`eval.py`, `eval_multiple.py`), and playback (`play.py`, `render.py`). Keep new orchestration scripts here so they can reuse Hydra flags.
- **Supporting Assets**: `assets/` and `assets_mjcf/` store URDF/MJCF resources, `data/` holds sample motion datasets, and `outputs/` is the default wandb/log export target. Treat `loco-mujoco/` as an embedded dependency—change it only when upstream fixes are required.

## Build, Test, and Development Commands
- **Environment Setup**: `conda create -n hdmi python=3.11` followed by `pip install -e .` installs the package with the dependencies declared in `setup.py`.
- **Training**: `python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase` launches a teacher policy run; swap `algo` and `task` to target other configs.
- **Evaluation / Playback**: `python scripts/play.py algo=<config> task=<config> checkpoint_path=run:<wandb-run>` replays saved checkpoints in Isaac Sim.
- **Testing**: `pytest loco-mujoco/tests -k <keyword>` runs the embedded locomotion test suite; ensure Mujoco, JAX, and related dependencies are available before executing.

## Coding Style & Naming Conventions
- **Python**: Follow Black-compatible formatting with 4-space indentation and descriptive module-level names (`humanoid_controller.py`, `imu_encoder.py`).
- **Configs**: Use lowercase, hyphen-free Hydra node names; prefer additive overrides (e.g., `python scripts/train.py task=G1/hdmi/push_box seed=42`).
- **Logging**: Reuse existing WandB group naming (`${algo}_${task}`) to keep dashboards consistent.

## Testing Guidelines
- **Framework**: PyTest powers integration checks in `loco-mujoco/tests`. Mirror its fixture patterns when adding coverage for new environments or algorithms.
- **Expectations**: Provide lightweight determinism guards (seeded RNGs, bounded tolerances) so tests remain reproducible across CPU/GPU backends.
- **Artifacts**: Avoid writing to `data/` during tests; use temporary directories under `outputs/` or fixtures that clean up after execution.

## Commit & Pull Request Guidelines
- **Commit Messages**: Use an imperative one-line summary (e.g., `Add contact reward shaping`) followed by optional wrapped body context. Group related changes rather than bundling unrelated fixes.
- **Pull Requests**: Link to the motivating issue, describe experiment commands that validate the change, and attach logs or short clips when behavior changes. Request reviews for modifications to `active_adaptation/` or `cfg/` because they directly affect training reproducibility.

## Configuration Tips
- **Hydra Overrides**: Keep default configs untouched; add new YAMLs under `cfg/task/` or `cfg/algo/` and reference them via command-line overrides.
- **Secrets & Paths**: Store machine-specific dataset or Isaac Sim paths in a personal `.env` (excluded via `.gitignore`) and reference them via environment variables, not hard-coded strings.
