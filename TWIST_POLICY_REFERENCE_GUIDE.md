# TWIST Policy Reference Integration Guide

本文档介绍如何使用冻结的 TWIST teacher policy 作为 HDMI 训练的参考动作提供者。

## 概述

### 功能简介

传统的 HDMI 训练使用 motion library 中的 `ref_joint_pos` 作为参考动作，通过残差学习训练策略：

```python
dfinal_action = ref_joint_pos_from_motionlib + residual_action
```

现在我们支持使用**冻结的 TWIST teacher policy** 作为参考动作提供者：

```python
ref_action = frozen_twist_policy(twist_observations)  # 冻结不训练
residual_action = hdmi_student_policy(hdmi_observations)  # 训练
final_action = ref_action + residual_action
```

### 核心特性

1. **双 Command Manager 系统**

   - HDMI command manager: 为 student policy 提供 HDMI 格式的观察
   - TWIST command manager: 为 frozen teacher policy 提供 TWIST 格式的观察
   - 两者共享相同的 motion 数据文件
2. **冻结的 TWIST Policy**

   - 从 checkpoint 加载预训练的 TWIST teacher
   - 所有参数冻结（`requires_grad=False`）
   - 仅用于推理，不参与训练
3. **双观察系统**

   - TWIST observations: `proprio_history_combined` + `ref_motion_windowed`
   - HDMI observations: 标准 HDMI 观察（`ref_joint_pos_action_policy`, `object_pos_b`, 等）
4. **无缝集成**

   - 通过 YAML 配置即可启用/禁用
   - 向后兼容原有的 motion library reference

---

## 快速开始

### 步骤 1: 准备 TWIST Teacher Checkpoint

首先，你需要训练一个 TWIST teacher policy（或使用已有的）：

```bash
# 训练 TWIST teacher（如果还没有）
python scripts/train.py \
    algo=ppo \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096
```

记录下训练完成后的 wandb run ID，例如: `hdmi/abc123xyz`

### 步骤 2: 配置 HDMI 任务使用 TWIST Reference

创建或修改任务配置文件（例如 `cfg/task/G1/hdmi/move_suitcase_twist_ref.yaml`）：

```yaml
defaults:
  - base/hdmi-base-twist-ref  # 使用 TWIST reference 基础配置
  - _self_

name: G1TrackSuitcase_TwistRef

# ... 其他任务配置 ...

# TWIST Policy Reference 配置
reference:
  source: twist_policy  # 使用 TWIST policy 作为 reference

  twist_policy:
    # 指定 TWIST teacher checkpoint（必须！）
    checkpoint_path: "run:hdmi/abc123xyz"  # 替换为你的 TWIST run ID

    policy_type: ppo

    # TWIST command manager 配置（使用相同的 motion 数据）
    command:
      data_path: ${task.command.data_path}
      # ... TWIST 特定配置 ...
```

### 步骤 3: 启动训练

```bash
python scripts/train.py \
    algo=ppo_roa_train_twist_ref \
    task=G1/hdmi/move_suitcase_twist_ref \
    task.reference.twist_policy.checkpoint_path="/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/outputs/2025-11-03/21-56-06-G1TwistTeacherAligned-ppotest_1014_twist/wandb/run-20251103_215613-mmk3woo1/files/checkpoint_9000.pt"
```

**重要参数：**

- `algo=ppo_roa_train_twist_ref`: 使用支持冻结策略的算法配置
- `task.reference.twist_policy.checkpoint_path`: TWIST teacher checkpoint 路径

---

## 配置详解

### Algorithm 配置

在 `active_adaptation/learning/ppo/ppo_roa.py` 中注册了新的算法配置：

```python
cs.store("ppo_roa_train_twist_ref", node=PPOConfig(
    phase="train",
    vecnorm="train",
    entropy_coef_start=0.001,
    entropy_coef_end=0.001,
    use_frozen_policy_ref=True,  # 启用冻结策略 reference
    enable_residual_distillation=True  # 启用残差学习
), group="algo")
```

**关键参数：**

- `use_frozen_policy_ref=True`: 启用冻结策略 reference
- `frozen_policy_checkpoint`: checkpoint 路径（可在此或 task config 中设置）
- `frozen_policy_type`: 策略类型（"ppo" 或 "ppo_roa"）

### Task 配置

#### 基础配置：`cfg/task/base/hdmi-base-twist-ref.yaml`

```yaml
defaults:
  - hdmi-base  # 继承 HDMI 基础配置
  - _self_

reference:
  source: twist_policy  # "motionlib" 或 "twist_policy"

  twist_policy:
    checkpoint_path: ???  # 必须在具体任务中指定

    # TWIST command manager
    command:
      _target_: active_adaptation.envs.mdp.commands.twist.command.TwistMotionTracking
      data_path: ${task.command.data_path}  # 与 HDMI 共享
      tracking_keypoint_names: [...]
      future_steps: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # TWIST observations
    observation:
      policy:
        proprio_history_combined:
          _target_: ...
          history_length: 11
          # 推理时不加噪声
          root_ori_noise: 0.0
          root_ang_vel_noise: 0.0
          joint_pos_noise: 0.0

        ref_motion_windowed:
          _target_: ...
          past_frames: 10
          future_frames: 10
          coordinate_frame: world
```

#### 具体任务配置示例

参见 `cfg/task/G1/hdmi/move_suitcase_twist_ref.yaml`

---

## 架构说明

### 系统架构图

```
┌──────────────────────────────────────────────────────────┐
│                  HDMI Environment                         │
│                                                           │
│  Motion Data: move_suitcase.npz (共享)                   │
│                                                           │
│  ┌────────────────────┐    ┌──────────────────────────┐ │
│  │ TWIST Command Mgr  │    │ HDMI Command Mgr         │ │
│  │ (TwistMotionTrack) │    │ (RobotObjectTracking)    │ │
│  └─────────┬──────────┘    └───────────┬──────────────┘ │
│            │                            │                 │
│            ▼                            ▼                 │
│  ┌────────────────────┐    ┌──────────────────────────┐ │
│  │ TWIST Obs Adapter  │    │ HDMI Obs Manager         │ │
│  │ - proprio_hist     │    │ - ref_joint_pos_action   │ │
│  │ - ref_motion_win   │    │ - object_pos_b           │ │
│  └─────────┬──────────┘    └───────────┬──────────────┘ │
│            │                            │                 │
│            ▼                            ▼                 │
│  ┌────────────────────┐    ┌──────────────────────────┐ │
│  │ Frozen TWIST       │    │ HDMI Student Policy      │ │
│  │ Teacher Policy     │    │ (Training)               │ │
│  │ ✓ No gradients     │    │ ✓ Learn residual         │ │
│  └─────────┬──────────┘    └───────────┬──────────────┘ │
│            │                            │                 │
│            │ ref_action                 │ residual        │
│            └────────────┬───────────────┘                 │
│                         ▼                                 │
│               ┌──────────────────┐                        │
│               │ final_action =   │                        │
│               │ ref + residual   │                        │
│               └──────────────────┘                        │
└──────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. `FrozenPolicyWrapper` (`frozen_policy_wrapper.py`)

负责加载和管理冻结的 TWIST teacher policy：

```python
frozen_policy = FrozenPolicyWrapper(
    checkpoint_path="run:hdmi/abc123",
    device="cuda:0"
)
frozen_policy.eval()  # 冻结模式

# 推理
ref_action = frozen_policy(twist_obs)
```

#### 2. `DualCommandManager` (`dual_command_manager.py`)

维护两个独立的 command manager：

```python
dual_manager = DualCommandManager(
    env=env,
    hdmi_config=hdmi_cfg,
    twist_config=twist_cfg,
    shared_motion_path="data/motion/g1/omomo/sub1_suitcase_011"
)

# 默认代理到 HDMI manager（向后兼容）
dual_manager.reset(env_ids)
dual_manager.update()

# 访问特定 manager
dual_manager.hdmi_manager
dual_manager.twist_manager
```

#### 3. `TwistObservationAdapter` (`dual_command_manager.py`)

为 TWIST policy 构建观察：

```python
twist_obs_adapter = TwistObservationAdapter(
    env=env,
    twist_command_manager=twist_cmd,
    cfg=twist_obs_cfg
)

twist_obs_adapter.update()
twist_obs_tensor = twist_obs_adapter.get_observation_tensor()
```

#### 4. PPO-ROA 集成 (`ppo_roa.py`)

修改了 PPO-ROA 以支持冻结策略 reference：

```python
class PPOROA:
    def __init__(self, cfg, ...):
        if cfg.use_frozen_policy_ref:
            # 加载冻结策略
            self.frozen_policy = FrozenPolicyWrapper(...)
            self.use_frozen_ref = True

    def set_twist_obs_adapter(self, adapter):
        self.twist_obs_adapter = adapter

    def compute_frozen_policy_reference(self, tensordict):
        # 构建 TWIST 观察并推理
        self.twist_obs_adapter.update()
        twist_obs = self.twist_obs_adapter.get_observation_tensor()
        ref_action = self.frozen_policy(twist_obs)
        return ref_action

    def _update_ppo(self, tensordict):
        if self.use_frozen_ref:
            ref_action = self.compute_frozen_policy_reference(tensordict)
            self._frozen_ref_cache = ref_action  # 缓存给 residual module
        # ... PPO 训练 ...
```

---

## 训练流程

### 完整流程

1. **环境初始化**

   ```python
   env = SimpleEnv(cfg.task)
   # 包含 HDMI command manager
   ```
2. **策略初始化**

   ```python
   policy = PPOROA(cfg.algo, ...)
   # 如果 use_frozen_policy_ref=True，加载 frozen TWIST policy
   ```
3. **Dual Manager 设置**（在 `helpers.py` 中）

   ```python
   if cfg.algo.use_frozen_policy_ref:
       dual_manager = DualCommandManager(...)
       env.command_manager = dual_manager

       twist_obs_adapter = TwistObservationAdapter(...)
       policy.set_twist_obs_adapter(twist_obs_adapter)
   ```
4. **训练循环**

   ```python
   for step in range(total_steps):
       # 环境 step
       env.step()

       # Collect rollout
       carry = rollout_policy(carry)  # HDMI student policy

       if use_frozen_ref:
           # 在 PPO 更新时自动调用
           ref_action = policy.compute_frozen_policy_reference(tensordict)
           final_action = ref_action + student_residual_action

       # 训练
       policy.train_op(tensordict)
   ```

### 关键时机

- **冻结策略推理时机**: 在每个 PPO mini-batch 更新之前
- **TWIST 观察更新时机**: 每次计算 reference action 时
- **Dual manager 更新时机**: 每个环境 step

---

## 常见问题

### Q1: Checkpoint 格式要求

**A:** Checkpoint 必须包含以下键：

- `policy`: 策略状态字典
- （可选）`vecnorm`: 观察归一化参数

支持的路径格式：

- WandB: `run:project/run_id`
- 本地: `/path/to/checkpoint.pt`

### Q2: Motion 数据格式

**A:** HDMI 和 TWIST 使用相同的 NPZ motion 文件，但读取方式不同：

- HDMI: 通过 `RobotObjectTracking` 读取
- TWIST: 通过 `TwistMotionTracking` 读取

确保你的 motion 文件包含所有必要的字段（body_pos_w, joint_pos, 等）。

### Q3: 观察维度不匹配

**A:** 确保：

1. TWIST teacher 训练时的机器人配置与 HDMI 任务一致（关节数量）
2. TWIST observation 配置正确（history_length=11, past_frames=10, future_frames=10）
3. Checkpoint 中的策略网络输入维度与当前观察维度一致

### Q4: 如何禁用 TWIST reference

**A:** 有两种方式：

**方式 1**: 使用原始 HDMI base 配置

```yaml
defaults:
  - base/hdmi-base  # 不使用 twist-ref
```

**方式 2**: 在命令行覆盖

```bash
python scripts/train.py \
    algo=ppo_roa_train \  # 不使用 twist_ref variant
    task=G1/hdmi/move_suitcase  # 原始任务
```

### Q5: 可以使用 PPO (非 ROA) 训练的 TWIST teacher 吗？

**A:** 可以！设置 `frozen_policy_type: ppo`：

```yaml
reference:
  twist_policy:
    policy_type: ppo  # 或 "ppo_roa"
```

---

## 调试技巧

### 1. 检查 TWIST 观察维度

```python
# 在 TwistObservationAdapter 中添加打印
def compute(self):
    obs_dict = {}
    for name, obs_fn in self.obs_functions.items():
        obs = obs_fn.compute()
        print(f"[DEBUG] {name}: {obs.shape}")
        obs_dict[name] = obs
    return obs_dict
```

### 2. 验证 Frozen Policy 输出

```python
# 在 PPOROA.compute_frozen_policy_reference 中
ref_action = self.frozen_policy(frozen_policy_input)
print(f"[DEBUG] Frozen policy ref_action: shape={ref_action.shape}, "
      f"mean={ref_action.mean()}, std={ref_action.std()}")
```

### 3. 检查 Dual Manager 状态

```python
# 在环境 step 后
print(f"HDMI motion time: {dual_manager.hdmi_manager.t}")
print(f"TWIST motion time: {dual_manager.twist_manager.t}")
```

### 4. 启用详细日志

```bash
export HYDRA_FULL_ERROR=1
python scripts/train.py ... --cfg job
```

---

## 最佳实践

### 1. Checkpoint 管理

- 使用 WandB 存储 TWIST teacher checkpoints
- 记录每个 checkpoint 对应的训练配置
- 定期验证 checkpoint 的推理性能

### 2. 训练策略

- **阶段 1**: 训练 TWIST teacher（纯 motion tracking，无物体）
- **阶段 2**: 使用冻结的 TWIST teacher 训练 HDMI student（带物体交互）
- **阶段 3**: Fine-tune HDMI student（可选）

### 3. 超参数调优

- 初始学习率: `3e-4`（与原 HDMI 相同）
- Entropy coefficient: `0.001`
- Residual action scale: 可能需要调整 `action_scaling`

### 4. Motion 数据

- 确保 motion 数据质量高
- 使用足够长的 motion 序列（>5 秒）
- 检查 motion 的连续性和平滑性

---

## 示例：完整训练命令

```bash
# Step 1: 训练 TWIST teacher（如果还没有）
python scripts/train.py \
    algo=ppo \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096 \
    wandb.project=hdmi \
    wandb.name=twist_teacher_move_suitcase

# 记录 run ID: hdmi/abc123xyz

# Step 2: 使用 TWIST reference 训练 HDMI student
python scripts/train.py \
    algo=ppo_roa_train_twist_ref \
    task=G1/hdmi/move_suitcase_twist_ref \
    task.reference.twist_policy.checkpoint_path="run:hdmi/abc123xyz" \
    task.num_envs=4096 \
    wandb.project=hdmi \
    wandb.name=hdmi_student_twist_ref

# Step 3: 评估
python scripts/play.py \
    algo=ppo_roa_finetune \
    task=G1/hdmi/move_suitcase_twist_ref \
    checkpoint_path="run:hdmi/def456uvw" \
    task.reference.twist_policy.checkpoint_path="run:hdmi/abc123xyz"
```

---

## 文件清单

### 新增文件

1. `active_adaptation/learning/ppo/frozen_policy_wrapper.py`

   - `FrozenPolicyWrapper`: 冻结策略包装器
   - `DualObservationBuilder`: 双观察构建器（已废弃，使用 TwistObservationAdapter）
2. `active_adaptation/envs/mdp/commands/dual_command_manager.py`

   - `DualCommandManager`: 双 command manager
   - `TwistObservationAdapter`: TWIST 观察适配器
3. `cfg/task/base/hdmi-base-twist-ref.yaml`

   - TWIST reference 基础配置
4. `cfg/task/G1/hdmi/move_suitcase_twist_ref.yaml`

   - move_suitcase 任务 + TWIST reference 示例
5. `TWIST_POLICY_REFERENCE_GUIDE.md`

   - 本文档

### 修改文件

1. `active_adaptation/learning/ppo/ppo_roa.py`

   - 添加 `use_frozen_policy_ref` 配置项
   - 添加 `set_twist_obs_adapter()` 方法
   - 添加 `compute_frozen_policy_reference()` 方法
   - 修改 `_update_ppo()` 集成冻结策略
   - 注册 `ppo_roa_train_twist_ref` 算法配置
2. `scripts/helpers.py`

   - 添加 `_setup_frozen_policy_reference()` 函数
   - 在 `make_env_policy()` 中集成设置逻辑

---

## 版本历史

- **v1.0** (2025-11-05): 初始实现
  - 支持冻结 TWIST teacher policy 作为 reference
  - 双 command manager 系统
  - TWIST observation adapter
  - 完整的训练和评估支持

---

## 联系与支持

如遇问题，请检查：

1. 配置文件是否正确
2. Checkpoint 路径是否有效
3. Motion 数据格式是否兼容
4. 观察维度是否匹配

Happy training! 🚀
