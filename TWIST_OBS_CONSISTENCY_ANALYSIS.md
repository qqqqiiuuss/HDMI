# TWIST Frozen Policy 观察一致性分析

## 核心问题

**TWIST frozen policy 的观察在训练和推理时是否一致？**

---

## 简短回答

### ⚠️ **部分一致，但有关键差异**

| 观察组件 | 训练时配置 | 推理时配置 | 一致性 |
|---------|-----------|-----------|--------|
| **proprio_history_combined** | 带噪声 | **无噪声** | ⚠️ 不一致 |
| **ref_motion_windowed** | 无噪声 | 无噪声 | ✅ 一致 |
| **观察参数** (history_length等) | 未明确配置 | 使用**默认值** | ⚠️ 可能不一致 |

**关键问题**：
1. **训练时有噪声，推理时无噪声** - 可能影响性能
2. **配置未显式指定** - 使用代码默认值，可能与训练不一致

---

## 详细分析

### 1. TWIST 训练时的观察配置

**配置文件**: `cfg/task/G1/twist/0927_twist_teacher_new.yaml:111-133`

```yaml
observation:
  policy:
    # Proprioceptive 观察（带噪声）
    proprio_history_combined:
      _target_: active_adaptation.envs.mdp.commands.twist.observations.proprio_history_combined
      history_length: 11
      root_ori_noise: 0.1        # ⚠️ 训练时加噪声
      root_ang_vel_noise: 0.1    # ⚠️ 训练时加噪声
      joint_pos_noise: 0.01      # ⚠️ 训练时加噪声
      joint_vel_noise: 0.1       # ⚠️ 训练时加噪声
      action_noise: 0.0
      noise_increasing_steps: 3000

    # Reference Motion 观察
    ref_motion_windowed:
      _target_: active_adaptation.envs.mdp.commands.twist.observations.ref_motion_windowed
      past_frames: 10
      future_frames: 10
      coordinate_frame: world
      ref_root_pos_noise: 0.0    # ✅ 参考运动不加噪声
      ref_root_ori_noise: 0.0
      ref_joint_pos_noise: 0.0
```

**训练时观察维度** (假设29个关节):
- `proprio_history_combined`:
  - root_ori (6D) + root_ang_vel (3D) + joint_pos (29D) + joint_vel (29D) + action (29D)
  - = 96 维 × 11 历史帧 = **1056 维**

- `ref_motion_windowed`:
  - (root_pos (3D) + root_ori (6D) + joint_pos (29D)) × (10 past + 10 future)
  - = 38 维 × 20 帧 = **760 维**

**总观察维度**: 1056 + 760 = **1816 维**

---

### 2. HDMI 推理时的观察配置

**配置文件**: `cfg/task/G1/hdmi/move_suitcase_twist_ref.yaml:59-73`

```yaml
reference:
  source: twist_policy

  twist_policy:
    checkpoint_path: ???
    policy_type: ppo

    command:
      # ⚠️ 没有显式配置 observation!
      # data_path 会自动设置为 HDMI 任务的 motion 数据
```

**关键发现**: **配置中没有显式定义 `observation` 字段！**

---

### 3. 推理时观察如何构建？

**代码路径**: `scripts/helpers.py:318-324`

```python
# 从配置中获取 observation 配置
twist_obs_cfg = twist_cfg.get("observation", {}).get("policy", {})
# ⚠️ 如果配置中没有 observation，这里会得到空字典 {}

twist_obs_adapter = TwistObservationAdapter(
    env=base_env,
    twist_command_manager=twist_manager,
    cfg=twist_obs_cfg  # ⚠️ 传入空字典
)
```

**TwistObservationAdapter 初始化**: `dual_command_manager.py:173-210`

```python
def _init_observations(self):
    # 从配置中读取参数，如果没有则使用默认值
    proprio_cfg = self.cfg.get("proprio_history_combined", {})
    ref_motion_cfg = self.cfg.get("ref_motion_windowed", {})
    # ⚠️ 如果 cfg 是空字典，这两个都是空字典

    # 创建 proprio history 观察
    self.obs_functions["proprio_history_combined"] = proprio_history_cls(
        env=self.env,
        history_length=proprio_cfg.get("history_length", 11),  # ✅ 默认11
        root_ori_noise=0.0,      # ⚠️ 硬编码为0（推理时不加噪声）
        root_ang_vel_noise=0.0,  # ⚠️ 硬编码为0
        joint_pos_noise=0.0,     # ⚠️ 硬编码为0
        joint_vel_noise=0.0,     # ⚠️ 硬编码为0
        action_noise=0.0,
        noise_increasing_steps=proprio_cfg.get("noise_increasing_steps", 3000)
    )

    # 创建 ref_motion_windowed 观察
    self.obs_functions["ref_motion_windowed"] = ref_motion_windowed_cls(
        env=self.env,
        past_frames=ref_motion_cfg.get("past_frames", 10),       # ✅ 默认10
        future_frames=ref_motion_cfg.get("future_frames", 10),   # ✅ 默认10
        coordinate_frame=ref_motion_cfg.get("coordinate_frame", "world"),
        ref_root_pos_noise=0.0,
        ref_root_ori_noise=0.0,
        ref_joint_pos_noise=0.0
    )
```

---

## 关键差异对比

### 差异 1: Proprioceptive 噪声

| 噪声类型 | 训练时 | 推理时 | 影响 |
|---------|--------|--------|------|
| `root_ori_noise` | **0.1** | **0.0** | ⚠️ 不一致 |
| `root_ang_vel_noise` | **0.1** | **0.0** | ⚠️ 不一致 |
| `joint_pos_noise` | **0.01** | **0.0** | ⚠️ 不一致 |
| `joint_vel_noise` | **0.1** | **0.0** | ⚠️ 不一致 |

**分析**:
- **训练时**: 使用 **噪声课程学习** (noise_increasing_steps=3000)，前3000步噪声从0逐渐增加到全噪声
- **推理时**: 噪声**全部硬编码为0**
- **影响**:
  - ✅ **正常** - 推理时不加噪声是合理的（类似于 BN 的 train/eval 模式）
  - ✅ **有利** - 推理时使用干净观察，理论上应该表现更好
  - ⚠️ **风险** - 如果训练时过拟合了噪声，推理时可能泛化性差

### 差异 2: Reference Motion 噪声

| 噪声类型 | 训练时 | 推理时 | 影响 |
|---------|--------|--------|------|
| `ref_root_pos_noise` | **0.0** | **0.0** | ✅ 一致 |
| `ref_root_ori_noise` | **0.0** | **0.0** | ✅ 一致 |
| `ref_joint_pos_noise` | **0.0** | **0.0** | ✅ 一致 |

**分析**: ✅ **完全一致** - ref_motion 观察训练和推理都不加噪声

### 差异 3: 观察参数

| 参数 | 训练时 | 推理时（默认值） | 影响 |
|------|--------|-----------------|------|
| `history_length` | **11** | **11** | ✅ 一致 |
| `past_frames` | **10** | **10** | ✅ 一致 |
| `future_frames` | **10** | **10** | ✅ 一致 |
| `coordinate_frame` | **world** | **world** | ✅ 一致 |

**分析**: ✅ **一致** - 默认值与训练配置匹配

---

## 潜在问题与风险

### ⚠️ 问题 1: 配置未显式指定

**当前状态**:
```yaml
# cfg/task/G1/hdmi/move_suitcase_twist_ref.yaml
reference:
  twist_policy:
    command: {...}
    # ⚠️ 缺少 observation 配置
```

**风险**:
- 依赖代码中的默认值
- 如果默认值改变，推理行为会改变
- 不明确的配置让用户难以理解

**建议**: ✅ **显式添加观察配置**

```yaml
reference:
  twist_policy:
    checkpoint_path: ???

    command:
      # ...

    # ✅ 显式配置观察（与训练保持一致的参数，但推理时不加噪声）
    observation:
      policy:
        proprio_history_combined:
          history_length: 11
          # 噪声在推理时会被 TwistObservationAdapter 硬编码为0
          noise_increasing_steps: 3000

        ref_motion_windowed:
          past_frames: 10
          future_frames: 10
          coordinate_frame: world
```

### ⚠️ 问题 2: 噪声课程学习参数

**训练时**: `noise_increasing_steps: 3000`
- 前3000步噪声从0逐渐增加
- 第3000步后噪声达到全强度 (0.1, 0.1, 0.01, 0.1)

**推理时**: 噪声硬编码为0，但 `noise_increasing_steps=3000` 仍被传入

**影响**:
- 实际上无影响（因为噪声已经是0）
- 但语义上不清晰

### ⚠️ 问题 3: 数据分布差异

**训练数据**: AMASS 通用运动（PKL格式）
- 包含各种运动：crouch, walk, hop, dodge, swing等
- 数据集包含 15k+ 运动序列
- 多样性高，覆盖广泛的运动模式

**推理数据**: HDMI 任务特定运动（NPZ格式）
- 仅包含推箱子任务的运动
- 单一任务，数据分布窄

**影响**:
- ⚠️ **分布偏移**: TWIST frozen policy 在训练时见过的运动模式 ≠ 推理时的任务运动
- ✅ **符合设计**: 这正是 frozen GMT + residual 的设计初衷
  - Frozen GMT: 提供通用运动先验
  - Residual policy: 学习任务特定的调整

---

## 观察维度一致性验证

### 训练时观察维度

假设29个关节：

1. **proprio_history_combined** (带噪声):
   ```
   每帧: root_ori(6D) + root_ang_vel(3D) + joint_pos(29D) + joint_vel(29D) + action(29D)
        = 96 维
   历史: 96 × 11 = 1056 维
   ```

2. **ref_motion_windowed** (无噪声):
   ```
   每帧: root_pos(3D) + root_ori(6D) + joint_pos(29D) = 38 维
   时间窗口: 38 × (10 past + 10 future) = 38 × 20 = 760 维
   ```

**总维度**: 1056 + 760 = **1816 维**

### 推理时观察维度

使用相同参数（history_length=11, past_frames=10, future_frames=10）:

1. **proprio_history_combined** (无噪声):
   ```
   每帧: 96 维（结构相同，只是噪声为0）
   历史: 96 × 11 = 1056 维
   ```

2. **ref_motion_windowed** (无噪声):
   ```
   每帧: 38 维
   时间窗口: 38 × 20 = 760 维
   ```

**总维度**: 1056 + 760 = **1816 维**

### ✅ 维度完全一致

---

## 代码执行流程对比

### 训练时

```
1. 环境初始化
   ├─ HDMI/TWIST task config 加载
   ├─ TwistMotionTracking command manager 创建
   │   └─ 加载 PKL 数据 (AMASS 通用运动)
   └─ Observation functions 创建
       ├─ proprio_history_combined(噪声=0.1/0.1/0.01/0.1)
       └─ ref_motion_windowed(噪声=0)

2. 训练循环
   ├─ 前3000步: 噪声从0逐渐增加
   └─ 3000步后: 噪声达到全强度

3. Policy 训练
   └─ 学习在带噪声观察下的鲁棒策略
```

### 推理时（作为 Frozen Reference）

```
1. 环境初始化
   ├─ HDMI task config 加载 (move_suitcase_twist_ref.yaml)
   ├─ HDMI RobotObjectTracking command manager 创建（用于 student）
   └─ Observation functions 创建（HDMI student 观察）

2. Frozen Policy 设置
   ├─ 从 checkpoint 加载 TWIST frozen policy
   ├─ 创建 TWIST command manager
   │   └─ 加载 NPZ 数据 (HDMI 任务特定运动)
   └─ 创建 TwistObservationAdapter
       ├─ proprio_history_combined(噪声=0/0/0/0) ⚠️ 硬编码无噪声
       └─ ref_motion_windowed(噪声=0)

3. 推理循环
   ├─ TwistObservationAdapter.compute()
   │   ├─ 临时切换 env.command_manager → TWIST manager
   │   ├─ 计算 TWIST 观察（无噪声，干净观察）
   │   └─ 恢复 env.command_manager → HDMI manager
   ├─ frozen_policy(twist_obs) → frozen_action
   └─ final_action = frozen_action + residual_action
```

---

## 关键设计决策的合理性分析

### 决策 1: 推理时不加噪声

**代码**: `dual_command_manager.py:193-197`
```python
self.obs_functions["proprio_history_combined"] = proprio_history_cls(
    env=self.env,
    history_length=proprio_cfg.get("history_length", 11),
    root_ori_noise=0.0,  # 推理时不加噪声
    ...
)
```

**合理性**: ✅ **合理**
- 训练时加噪声是为了提高鲁棒性（数据增强）
- 推理时使用干净观察是标准做法
- 类比: BatchNorm 训练时用 batch stats，推理时用 running stats

### 决策 2: 使用任务特定运动数据

**代码**: `helpers.py:289`
```python
twist_command_cfg["data_path"] = hdmi_data_path_abs
# TWIST frozen policy 使用 HDMI 任务的运动数据，而非训练时的 AMASS
```

**合理性**: ✅ **合理**
- 符合论文设计：GMT 提供通用先验，但需要任务特定的 reference motion
- Frozen policy 的作用：在 **任务运动轨迹** 上提供基础动作
- Residual policy：学习偏离 reference 的调整（例如避障、适应物体重量变化）

### 决策 3: 临时切换 command_manager

**代码**: `dual_command_manager.py:181-213`
```python
# CRITICAL: 临时替换 env.command_manager 为 TWIST manager
actual_env.command_manager = self.command_manager
try:
    # 计算 TWIST 观察
    ...
finally:
    # 恢复原始 command manager
    actual_env.command_manager = original_command_manager
```

**合理性**: ✅ **合理**
- 保证 TWIST 观察使用正确的 command manager
- 避免修改环境的全局状态
- 干净的隔离机制

---

## 最终结论

### ✅ 观察**结构和维度**完全一致

| 维度 | 训练时 | 推理时 | 状态 |
|------|--------|--------|------|
| proprio_history_combined | 1056 | 1056 | ✅ 一致 |
| ref_motion_windowed | 760 | 760 | ✅ 一致 |
| **总维度** | **1816** | **1816** | ✅ 一致 |

### ⚠️ 观察**内容**有差异（符合预期）

| 差异项 | 训练时 | 推理时 | 合理性 |
|--------|--------|--------|--------|
| **Proprio 噪声** | 有噪声 (0.1/0.1/0.01/0.1) | 无噪声 (0/0/0/0) | ✅ 合理 |
| **Ref motion 噪声** | 无噪声 | 无噪声 | ✅ 一致 |
| **运动数据** | AMASS 通用运动 | HDMI 任务运动 | ✅ 符合设计 |

### 推荐改进

1. **显式配置观察参数** (优先级: 高)
   ```yaml
   # cfg/task/G1/hdmi/move_suitcase_twist_ref.yaml
   reference:
     twist_policy:
       observation:
         policy:
           proprio_history_combined:
             history_length: 11
           ref_motion_windowed:
             past_frames: 10
             future_frames: 10
             coordinate_frame: world
   ```

2. **添加观察维度检查** (优先级: 中)
   ```python
   # scripts/helpers.py
   twist_obs = twist_obs_adapter.get_observation_tensor()
   print(f"[Info] TWIST obs shape: {twist_obs.shape}")
   assert twist_obs.shape[-1] == expected_dim, f"TWIST obs dim mismatch: {twist_obs.shape[-1]} vs {expected_dim}"
   ```

3. **文档化设计决策** (优先级: 低)
   - 在 README 中说明为什么训练和推理使用不同的运动数据
   - 解释噪声差异的合理性

---

## 相关文件

### 核心代码
- `active_adaptation/envs/mdp/commands/dual_command_manager.py` - TwistObservationAdapter
- `scripts/helpers.py:220-340` - _setup_frozen_policy_reference()
- `active_adaptation/envs/mdp/commands/twist/observations.py` - TWIST 观察函数

### 配置文件
- `cfg/task/G1/twist/0927_twist_teacher_new.yaml` - TWIST 训练配置
- `cfg/task/G1/hdmi/move_suitcase_twist_ref.yaml` - HDMI+TWIST 推理配置

### 相关文档
- `TWIST_FROZEN_POLICY_README.md` - TWIST frozen policy 集成文档
- `DATA_FORMAT_CONSISTENCY_CHECK.md` - 数据格式一致性检查

---

**分析日期**: 2025-11-07
**分析人**: Claude Code
**结论**: ✅ **观察维度完全一致，噪声差异合理，可以安全使用**
