# TWIST Frozen Policy 集成说明

本文档详细说明了如何使用 TWIST 预训练策略作为 frozen reference 来训练 HDMI 任务的 residual policy。

## 目录
- [背景介绍](#背景介绍)
- [核心概念](#核心概念)
- [代码修改说明](#代码修改说明)
- [使用方法](#使用方法)
- [配置文件说明](#配置文件说明)
- [关键技术细节](#关键技术细节)
- [故障排查](#故障排查)

---

## 背景介绍

根据论文 "Residual Refinement Policy" 的方法：

1. **Stage I: GMT (General Motion Tracking)**
   - 预训练一个通用的运动跟踪策略 πGMT (即 TWIST)
   - 使用大规模的 AMASS 数据集训练
   - 学习基础的运动跟踪能力

2. **Stage II: Residual Refinement**
   - 冻结 GMT 策略作为 reference
   - 训练一个 residual policy πRes
   - 最终动作：`a_t = a_gmt_t + Δa_res_t`
   - 使用任务特定的参考轨迹 `(ŝ_r_t, ŝ_o_t)`

**关键点**：GMT 和 residual policy 都使用**相同的任务相关参考轨迹**（例如 suitcase motion），而不是 GMT 训练时的 AMASS 数据集。

---

## 核心概念

### 1. 原始训练流程 (无 frozen policy)

```bash
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase
```

**训练流程：**
- 使用 HDMI 的 `RobotObjectTracking` command manager
- 加载任务特定的 NPZ motion 数据（例如 suitcase）
- 直接训练 teacher-student 策略

**数据流：**
```
NPZ Motion Data (suitcase)
    ↓
RobotObjectTracking (HDMI)
    ↓
Observations (HDMI format)
    ↓
PPO-ROA (Teacher-Student)
```

### 2. 新的训练流程 (使用 TWIST frozen policy)

```bash
python scripts/train.py \
    algo=ppo_roa_train_twist_ref \
    task=G1/hdmi/move_suitcase_twist_ref \
    task.reference.twist_policy.checkpoint_path=<TWIST_checkpoint_path>
```

**训练流程：**
- 加载 TWIST frozen policy checkpoint
- 为 TWIST 创建独立的 `TwistMotionTracking` command manager
- TWIST manager 使用**相同的 suitcase NPZ 数据**
- TWIST frozen policy 提供 reference action
- Residual policy 学习基于 reference 的修正

**数据流：**
```
NPZ Motion Data (suitcase)
    ↓
    ├─→ RobotObjectTracking (HDMI) → HDMI Observations → Residual Policy
    │
    └─→ TwistMotionTracking (TWIST) → TWIST Observations → Frozen TWIST Policy
                                                                    ↓
                                                            Reference Action
                                                                    ↓
                                    Final Action = TWIST Action + Residual Action
```

---

## 代码修改说明

### 1. 核心文件修改

#### 1.1 `scripts/helpers.py` - Frozen Policy 设置

**新增函数：** `_setup_frozen_policy_reference()`

**主要功能：**
```python
def _setup_frozen_policy_reference(env, policy, cfg):
    """设置 TWIST frozen policy reference"""

    # 1. 创建 TWIST 专用的 command manager
    twist_manager = TwistMotionTracking(
        env=base_env,
        data_path=hdmi_data_path  # 使用 HDMI 任务的 motion 数据
    )

    # 2. 创建 TWIST observation adapter
    twist_obs_adapter = TwistObservationAdapter(
        env=base_env,
        twist_command_manager=twist_manager,
        cfg=twist_obs_cfg
    )

    # 3. 将 adapter 设置到 policy
    policy.set_twist_obs_adapter(twist_obs_adapter)
```

**关键修改：**
- 使用 `hydra.utils.get_original_cwd()` 解析相对路径（避免 Hydra 工作目录问题）
- 自动将相对路径转换为绝对路径

#### 1.2 `active_adaptation/envs/mdp/commands/twist/command.py` - NPZ 格式支持

**修改内容：**

```python
# 自动检测数据格式
if isinstance(data_path, str):
    path_obj = Path(data_path)
    # 检查是否为包含 motion.npz 的目录
    if path_obj.is_dir() and (path_obj / "motion.npz").exists():
        is_npz_format = True

if is_npz_format:
    # 使用 MotionDataset 加载 NPZ 数据
    self.dataset = MotionDataset.create_from_path(...)
else:
    # 使用 TwistMotionDataset 加载 PKL 数据
    self.dataset = TwistMotionDataset.create_from_path(...)
```

**功能：**
- 自动检测 NPZ 和 PKL 格式
- 兼容 HDMI (NPZ) 和 TWIST (PKL) 两种数据格式

#### 1.3 `active_adaptation/utils/motion.py` - 数据兼容性

**修改内容：**

```python
class MotionData(TensorClass):
    # 原有字段
    body_pos_w: torch.Tensor      # [T, num_bodies, 3]
    body_lin_vel_w: torch.Tensor  # [T, num_bodies, 3]
    body_ang_vel_w: torch.Tensor  # [T, num_bodies, 3]
    # ... 其他字段

    # 新增：兼容 TwistMotionData 的 root 速度字段
    root_lin_vel_w: torch.Tensor  # [T, 3] - 提取自 body_lin_vel_w[:, 0, :]
    root_ang_vel_w: torch.Tensor  # [T, 3] - 提取自 body_ang_vel_w[:, 0, :]
```

**功能：**
- 从 `body_lin_vel_w[:, 0, :]` 提取 root body 速度
- 使 NPZ 的 `MotionData` 与 PKL 的 `TwistMotionData` 结构一致

#### 1.4 `active_adaptation/envs/mdp/commands/dual_command_manager.py` - 观察适配器

**新增类：** `TwistObservationAdapter`

**核心机制：临时替换 command manager**

```python
def _init_observations(self):
    # 临时替换 env.command_manager 为 TWIST manager
    actual_env = getattr(self.env, 'base_env', self.env)
    original_command_manager = actual_env.command_manager
    actual_env.command_manager = self.command_manager  # TWIST manager

    try:
        # 使用 TWIST manager 初始化观察函数
        self.obs_functions["proprio_history_combined"] = ...
        self.obs_functions["ref_motion_windowed"] = ...
    finally:
        # 恢复原始 HDMI manager
        actual_env.command_manager = original_command_manager
```

**为什么需要临时替换？**
- TWIST 观察函数在初始化时会从 `env.command_manager` 读取配置
- 观察函数需要使用 TWIST 的 joint 配置（29个关节）而不是 HDMI 的配置
- 通过临时替换，确保观察函数使用正确的数据源

**相同机制应用于：**
- `reset(env_ids)`: 重置观察时
- `update()`: 更新观察时
- `compute()`: 计算观察时

---

## 使用方法

### Step 1: 训练 TWIST Teacher Policy

首先训练一个 TWIST policy 作为 GMT（如果还没有）：

```bash
python scripts/train.py \
    algo=ppo_roa_train \
    task=G1/twist/0927_twist_teacher_new
```

训练完成后记录 checkpoint 路径，例如：
```
/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/outputs/2025-11-03/21-56-06-G1TwistTeacherAligned-ppotest_1014_twist/wandb/run-20251103_215613-mmk3woo1/files/checkpoint_9000.pt
```

### Step 2: 使用 TWIST Frozen Policy 训练 Residual Policy

```bash
python scripts/train.py \
    algo=ppo_roa_train_twist_ref \
    task=G1/hdmi/move_suitcase_twist_ref \
    task.reference.twist_policy.checkpoint_path=<TWIST_checkpoint_path>
```

**完整示例：**
```bash
python scripts/train.py \
    algo=ppo_roa_train_twist_ref \
    task=G1/hdmi/move_suitcase_twist_ref \
    task.reference.twist_policy.checkpoint_path=/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/outputs/2025-11-03/21-56-06-G1TwistTeacherAligned-ppotest_1014_twist/wandb/run-20251103_215613-mmk3woo1/files/checkpoint_9000.pt
```

---

## 配置文件说明

### 新增配置文件

#### 1. `cfg/task/G1/hdmi/move_suitcase_twist_ref.yaml`

```yaml
defaults:
  - base/hdmi-base-twist-ref  # 继承 TWIST reference 基础配置
  - _self_

name: G1TrackSuitcase_TwistRef

# TWIST Policy Reference 配置
reference:
  source: twist_policy  # 使用 TWIST policy 作为 reference

  twist_policy:
    checkpoint_path: ???  # 必须通过命令行指定
    policy_type: ppo

    # TWIST command manager 配置
    # 注意：data_path 会自动设置为 HDMI 任务的 motion 数据
    command:
      # 这里可以指定 TWIST 特定的参数
      # 例如 tracking_keypoint_names, future_steps 等

# 任务特定配置
command:
  data_path: data/motion/g1/omomo/sub1_suitcase_011  # HDMI 和 TWIST 共享
  # ... 其他 HDMI 配置
```

#### 2. `cfg/task/base/hdmi-base-twist-ref.yaml`

基础配置文件，定义了 TWIST reference 的通用设置。

#### 3. `cfg/algo/ppo_roa_train_twist_ref.yaml`

算法配置（在代码中注册，不是实际文件）：

```python
# 在 ppo_roa.py 中注册
ConfigStore.instance().store(
    group="algo",
    name="ppo_roa_train_twist_ref",
    node=dict(
        _target_="active_adaptation.learning.ppo.ppo_roa.PPOROA",
        phase="train",
        use_frozen_policy_ref=True,  # 启用 frozen policy reference
        enable_residual_distillation=True,
        # ... 其他参数
    )
)
```

---

## 关键技术细节

### 1. 数据格式差异处理

**问题：**
- TWIST 训练时使用 PKL 格式的 `TwistMotionData`
- HDMI 任务使用 NPZ 格式的 `MotionData`
- 两者字段不同：
  - `TwistMotionData`: `root_lin_vel_w` (仅 root)
  - `MotionData`: `body_lin_vel_w` (所有 body)

**解决方案：**
在 `MotionDataset.create_from_path()` 中：
```python
# 提取 root body 速度（index 0）
root_lin_vel_w[start_idx:end_idx] = motion["body_lin_vel_w"][:, 0, :]
root_ang_vel_w[start_idx:end_idx] = motion["body_ang_vel_w"][:, 0, :]
```

### 2. 路径解析问题

**问题：**
Hydra 会改变工作目录到 `outputs/YYYY-MM-DD/HH-MM-SS-<task_name>/`，导致相对路径失效。

**解决方案：**
```python
from hydra.utils import get_original_cwd

# 将相对路径转换为绝对路径
if not os.path.isabs(hdmi_data_path):
    hdmi_data_path_abs = str(Path(get_original_cwd()) / hdmi_data_path)
```

### 3. Command Manager 切换机制

**问题：**
- TWIST 观察函数需要访问 TWIST command manager（29个关节配置）
- 但环境的 `command_manager` 是 HDMI 的（可能不同配置）

**解决方案：**
临时替换机制：
```python
# 保存原始 manager
original_cm = env.command_manager

# 临时替换为 TWIST manager
env.command_manager = twist_manager

try:
    # 使用 TWIST manager 的操作
    obs_fn.update()
finally:
    # 恢复原始 manager
    env.command_manager = original_cm
```

这样确保：
- TWIST 观察函数使用 TWIST 的 joint 配置
- HDMI 观察函数使用 HDMI 的配置
- 两者互不干扰

### 4. 观察维度匹配

**TWIST 观察维度计算：**
```python
# ref_motion_dim = root_pos(3) + root_ori(6) + joint_pos(num_joints)
ref_motion_dim = 3 + 6 + 29 = 38  # HDMI (29 joints with wrists)
ref_motion_dim = 3 + 6 + 23 = 32  # TWIST (23 joints without wrists)
```

**关键：**
- TWIST frozen policy 期望 29 个关节（与其训练时一致）
- 通过使用 TWIST 专用的 command manager，确保维度正确

---

## 故障排查

### 问题 1: `AttributeError: 'root_lin_vel_w'`

**原因：** NPZ 数据没有 `root_lin_vel_w` 字段

**解决：** 已修复，`MotionData` 现在包含此字段

### 问题 2: `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 32 but got size 38`

**原因：** TWIST 观察函数使用了 HDMI 的 command manager（joint 数量不对）

**解决：** 已修复，使用临时替换机制确保使用正确的 command manager

### 问题 3: `FileNotFoundError: No such file or directory`

**原因：** Hydra 改变了工作目录，相对路径失效

**解决：** 已修复，使用 `get_original_cwd()` 解析绝对路径

### 问题 4: `AttributeError: 'SimpleEnv' object has no attribute 'base_env'`

**原因：** 不同环境类型的属性访问方式不同

**解决：** 已修复，使用 `getattr(env, 'base_env', env)` 兼容两种情况

---

## 性能对比

### 无 Frozen Policy (原始方法)

```bash
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase
```

**特点：**
- 从头训练 teacher-student policy
- 训练时间较长
- 可能需要更多样本

### 有 Frozen Policy (新方法)

```bash
python scripts/train.py \
    algo=ppo_roa_train_twist_ref \
    task=G1/hdmi/move_suitcase_twist_ref \
    task.reference.twist_policy.checkpoint_path=<path>
```

**特点：**
- 利用预训练的 TWIST policy
- Residual policy 只需学习任务特定的修正
- 理论上训练更快、效果更好
- 符合论文方法

---

## 总结

### 主要改动

1. **数据层面：**
   - 支持 NPZ 格式的 motion 数据
   - 添加兼容性字段 `root_lin_vel_w`, `root_ang_vel_w`

2. **Command Manager 层面：**
   - 创建 TWIST 专用的 `TwistMotionTracking` manager
   - 实现临时切换机制

3. **观察层面：**
   - 新增 `TwistObservationAdapter`
   - 确保 TWIST 观察使用正确的配置

4. **训练流程层面：**
   - 新增 frozen policy reference 机制
   - 实现 residual learning

### 向后兼容

**所有原有功能保持不变：**
```bash
# 原始方法仍然可用
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase

# 新方法是额外功能
python scripts/train.py algo=ppo_roa_train_twist_ref task=G1/hdmi/move_suitcase_twist_ref ...
```

### 下一步

1. 使用 TWIST frozen policy 训练 residual policy
2. 对比有/无 frozen policy 的性能差异
3. 评估在实际任务上的表现

---

## 参考

- 论文：Residual Refinement Policy with GMT
- TWIST 原始实现：`cfg/task/G1/twist/0927_twist_teacher_new.yaml`
- PPO-ROA 实现：`active_adaptation/learning/ppo/ppo_roa.py`
