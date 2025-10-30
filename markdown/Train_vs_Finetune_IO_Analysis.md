# Train vs Finetune 阶段：输入输出和监督信号详解

本文档详细列出 PPO-ROA 架构中每个模块在 Train 和 Finetune 阶段的输入、输出和监督信号。

---

## 目录

1. [模块级详细对比](#模块级详细对比)
2. [数据流图](#数据流图)
3. [关键差异总结](#关键差异总结)

---

## 模块级详细对比

### 1. encoder_priv (Teacher 专用)

| 阶段 | 使用场景 | 输入 (OBS) | 输出 | 监督信号 | 是否训练 |
|------|---------|-----------|------|---------|---------|
| **Train** | Rollout | `OBS_PRIV_KEY` + `OBJECT_KEY`<br>(特权观测：地形、摩擦、物体质量等) | `PRIV_FEATURE_KEY` (256维)<br>(编码后的特权特征) | PPO loss 反向传播<br>(通过 actor 的梯度) | ✅ 训练 |
| **Train** | Training | `OBS_PRIV_KEY` + `OBJECT_KEY`<br>(来自 Teacher 轨迹) | `PRIV_FEATURE_KEY`<br>(作为 adapt_module 的 ground truth) | 用于监督 adapt_module | ✅ 训练 |
| **Finetune** | Rollout | ❌ 不使用 | - | - | ❌ 不训练 |
| **Finetune** | Training | `OBS_PRIV_KEY` + `OBJECT_KEY`<br>(来自 Student 轨迹) | `PRIV_FEATURE_KEY`<br>(作为 adapt_module 的 ground truth) | 用于监督 adapt_module | ❌ 不训练<br>(冻结参数) |

**代码位置**：
- 定义：`ppo_roa.py` 第201-205行
- Rollout：第410行
- Training：第511-512行、第657行、第700-701行

**关键点**：
- Finetune 的 Training 阶段仍需要 `encoder_priv` 生成监督信号
- 但 `encoder_priv` 本身不更新参数

---

### 2. actor (Teacher 策略)

| 阶段 | 使用场景 | 输入 (OBS) | 输出 | 监督信号 | 是否训练 |
|------|---------|-----------|------|---------|---------|
| **Train** | Rollout | `CMD_KEY` (命令)<br>`OBS_KEY` (本体观测：关节位置/速度)<br>`PRIV_FEATURE_KEY` (真实特权特征) | `action` (关节目标)<br>`sample_log_prob` (动作对数概率) | - | - |
| **Train** | Training | 同上<br>(来自 Teacher 自己的轨迹) | `action`, `log_prob`, `dist` | **PPO loss**:<br>- Policy loss (clipped surrogate)<br>- Entropy loss | ✅ 训练<br>(opt_policy) |
| **Finetune** | Rollout | ❌ 不使用 | - | - | - |
| **Finetune** | Training | ❌ 不使用 | - | - | ❌ 不训练 |

**代码位置**：
- 定义：第259-260行
- Rollout：第411行
- Training：第656-658行、第698行、第704行

**关键点**：
- Teacher 只在 Train 阶段使用
- 输入包含完美的特权特征 `PRIV_FEATURE_KEY`

---

### 3. actor_adapt (Student 策略)

| 阶段 | 使用场景 | 输入 (OBS) | 输出 | 监督信号 | 是否训练 |
|------|---------|-----------|------|---------|---------|
| **Train** | Rollout | ❌ 不使用 (Teacher 执行动作) | - | - | - |
| **Train** | Training | `CMD_KEY`<br>`OBS_KEY`<br>`PRIV_PRED_KEY` (推断的特权特征)<br>或 `PRIV_FEATURE_KEY` (取决于配置) | `action`, `dist` | **监督蒸馏**:<br>`MSE(action_student, action_teacher)` | ✅ 训练<br>(opt_adapt_actor)<br>第529-545行 |
| **Finetune** | Rollout | `CMD_KEY`<br>`OBS_KEY`<br>`PRIV_PRED_KEY` (来自 adapt_ema) | `action`<br>`sample_log_prob` | - | - |
| **Finetune** | Training | 同上<br>(来自 Student 自己的轨迹) | `action`, `log_prob`, `dist` | **PPO loss**:<br>- Policy loss<br>- Entropy loss | ✅ 训练<br>(opt_policy)<br>第659-660行 |

**代码位置**：
- 定义：第264-265行
- Rollout：第418行
- Training (蒸馏)：第529-545行
- Training (PPO)：第659-660行、第698行、第704行

**关键点**：
- Train 阶段：通过**监督学习**模仿 Teacher 的动作分布
- Finetune 阶段：通过 **PPO** 自主优化策略
- 训练方式发生根本转变

---

### 4. adapt_module (共享模块 - 可训练)

| 阶段 | 使用场景 | 输入 (OBS) | 输出 | 监督信号 | 是否训练 |
|------|---------|-----------|------|---------|---------|
| **Train** | Rollout | `OBS_KEY` (本体观测)<br>`CMD_KEY` (命令)<br>`OBJECT_KEY` (物体状态)<br>(来自 Teacher 轨迹) | `PRIV_PRED_KEY` (256维)<br>(推断的特权特征) | - | - |
| **Train** | Training | 同上 | `PRIV_PRED_KEY` | **监督学习**:<br>`MSE(PRIV_PRED, PRIV_FEATURE)` | ✅ 训练<br>(opt_adapt)<br>第516-522行 |
| **Finetune** | Rollout | ❌ 不使用 (用 adapt_ema) | - | - | - |
| **Finetune** | Training | `OBS_KEY`<br>`CMD_KEY`<br>`OBJECT_KEY`<br>(来自 Student 轨迹) | `PRIV_PRED_KEY` | **监督学习**:<br>`MSE(PRIV_PRED, PRIV_FEATURE)` | ✅ 训练<br>(opt_adapt)<br>第516-522行 |

**代码位置**：
- 定义：第207-220行
- Rollout：第412行
- Training：第516-522行

**关键点**：
- 两阶段都训练，但数据分布不同
- 训练目标完全相同：学习从可观测信息推断特权信息
- **Finetune 的 Training 仍需 `OBS_PRIV_KEY`** 生成监督信号
- 实现 Domain Adaptation

---

### 5. adapt_ema (共享模块 - EMA 版本)

| 阶段 | 使用场景 | 输入 (OBS) | 输出 | 监督信号 | 是否训练 |
|------|---------|-----------|------|---------|---------|
| **Train** | Rollout | ❌ 不使用 | - | - | - |
| **Train** | Training | - | - | EMA 更新:<br>`0.96 * adapt_ema + 0.04 * adapt_module` | ⚙️ EMA 更新<br>(第561行) |
| **Finetune** | Rollout | `OBS_KEY`<br>`CMD_KEY`<br>`OBJECT_KEY`<br>(来自 Student 执行动作时) | `PRIV_PRED_KEY` | - | - |
| **Finetune** | Training | - | - | EMA 更新:<br>`0.96 * adapt_ema + 0.04 * adapt_module` | ⚙️ EMA 更新<br>(第561行) |

**代码位置**：
- 定义：第345行
- Rollout：第417行
- EMA 更新：第561行

**关键点**：
- 不直接训练，通过 EMA 从 `adapt_module` 更新
- Finetune 的 Rollout 使用 `adapt_ema`（更稳定）
- EMA 公式：`params_ema = (1-τ) * params_ema + τ * params_module`，其中 τ=0.04

---

### 6. critic (共享模块)

| 阶段 | 使用场景 | 输入 (OBS) | 输出 | 监督信号 | 是否训练 |
|------|---------|-----------|------|---------|---------|
| **Train** | Rollout | - | - | - | - |
| **Train** | Training | `OBS_PRIV_KEY` (特权观测)<br>`OBS_KEY` (本体观测)<br>`CMD_KEY` (命令)<br>`OBJECT_KEY` (物体状态)<br>(来自 Teacher 轨迹) | `state_value` (状态价值) | **TD 误差**:<br>`MSE(value, return)`<br>return = GAE 计算的回报 | ✅ 训练<br>(opt_critic)<br>第689-690行 |
| **Finetune** | Rollout | - | - | - | - |
| **Finetune** | Training | `OBS_PRIV_KEY`<br>`OBS_KEY`<br>`CMD_KEY`<br>`OBJECT_KEY`<br>(来自 Student 轨迹) | `state_value` | **TD 误差**:<br>`MSE(value, return)` | ✅ 训练<br>(opt_critic)<br>第689-690行 |

**代码位置**：
- 定义：第267-272行
- Training：第689-690行、第699行、第705行

**关键点**：
- 两阶段都训练，输入包含特权观测
- 共享同一个 critic，利用两阶段的所有数据
- 只在训练时使用，部署时不需要

---

## 数据流图

### Train 阶段

#### Rollout（每步环境交互）

```
环境 → 观测
├─ OBS_PRIV_KEY (地形高度、摩擦系数、物体质量等)
├─ OBS_KEY      (关节位置、速度、IMU等)
├─ CMD_KEY      (目标速度、方向等)
└─ OBJECT_KEY   (物体位置、姿态、速度等)

执行流程:
1. encoder_priv([OBS_PRIV, OBJECT]) → PRIV_FEATURE (256维)
2. actor([CMD, OBS, PRIV_FEATURE]) → action ✅ 执行
3. adapt_module([OBS, CMD, OBJECT]) → PRIV_PRED (旁观推断，不影响动作)

收集的数据:
├─ OBS_KEY, OBS_PRIV_KEY, CMD_KEY, OBJECT_KEY
├─ PRIV_FEATURE_KEY (真实)
├─ PRIV_PRED_KEY (推断)
├─ action, log_prob
└─ reward, done
```

#### Training（每32步）

```
训练 1: train_policy (PPO)
├─ 输入: [CMD, OBS, PRIV_FEATURE] (来自 Teacher 轨迹)
├─ 前向: encoder_priv → PRIV_FEATURE
│         actor([CMD, OBS, PRIV_FEATURE]) → new_log_prob
│         critic([OBS_PRIV, OBS, CMD, OBJECT]) → value
├─ 损失: policy_loss = -min(adv*ratio, adv*clipped_ratio)
│        entropy_loss = -entropy_coef * entropy
│        value_loss = MSE(value, return)
└─ 更新: opt_policy.step() → 更新 actor + encoder_priv
         opt_critic.step() → 更新 critic

训练 2: train_adapt - 特权特征推断 (监督学习)
├─ 输入: [OBS, CMD, OBJECT] (来自 Teacher 轨迹)
├─ 前向: adapt_module([OBS, CMD, OBJECT]) → PRIV_PRED
│        encoder_priv([OBS_PRIV, OBJECT]) → PRIV_FEATURE (冻结)
├─ 损失: priv_loss = MSE(PRIV_PRED, PRIV_FEATURE)
└─ 更新: opt_adapt.step() → 更新 adapt_module

训练 3: train_adapt - Student 动作蒸馏 (监督学习)
├─ 条件: cfg.phase == "train" and cfg.enable_residual_distillation
├─ 输入: [CMD, OBS, PRIV_PRED] 或 [CMD, OBS, PRIV_FEATURE]
├─ 前向: actor([CMD, OBS, PRIV_FEATURE]) → dist_teacher (冻结)
│        actor_adapt([CMD, OBS, PRIV_PRED]) → dist_student
├─ 损失: adapt_loss = MSE(student.mean, teacher.mean)
└─ 更新: opt_adapt_actor.step() → 更新 actor_adapt

EMA 更新:
└─ adapt_ema ← 0.96 * adapt_ema + 0.04 * adapt_module
```

---

### Finetune 阶段

#### Rollout（每步环境交互）

```
环境 → 观测
├─ OBS_PRIV_KEY (仍然存在，但不用于推断)
├─ OBS_KEY
├─ CMD_KEY
└─ OBJECT_KEY

执行流程:
1. adapt_ema([OBS, CMD, OBJECT]) → PRIV_PRED (推断特权特征)
2. actor_adapt([CMD, OBS, PRIV_PRED]) → action ✅ 执行

收集的数据:
├─ OBS_KEY, OBS_PRIV_KEY, CMD_KEY, OBJECT_KEY
├─ PRIV_PRED_KEY (推断，来自 adapt_ema)
├─ action, log_prob
└─ reward, done
```

#### Training（每32步）

```
训练 1: train_policy (PPO)
├─ 输入: [CMD, OBS, PRIV_PRED] (来自 Student 轨迹)
├─ 前向: actor_adapt([CMD, OBS, PRIV_PRED]) → new_log_prob
│        critic([OBS_PRIV, OBS, CMD, OBJECT]) → value
├─ 损失: policy_loss = -min(adv*ratio, adv*clipped_ratio)
│        entropy_loss = -entropy_coef * entropy
│        value_loss = MSE(value, return)
└─ 更新: opt_policy.step() → 更新 actor_adapt
         opt_critic.step() → 更新 critic

训练 2: train_adapt - 特权特征推断 (监督学习)
├─ 输入: [OBS, CMD, OBJECT] (来自 Student 轨迹 ⚠️)
├─ 前向: adapt_module([OBS, CMD, OBJECT]) → PRIV_PRED
│        encoder_priv([OBS_PRIV, OBJECT]) → PRIV_FEATURE (冻结)
├─ 损失: priv_loss = MSE(PRIV_PRED, PRIV_FEATURE)
└─ 更新: opt_adapt.step() → 更新 adapt_module

⚠️ 不训练 actor_adapt 的蒸馏 (phase != "train")

EMA 更新:
└─ adapt_ema ← 0.96 * adapt_ema + 0.04 * adapt_module
```

---

## 关键差异总结

### 1. 输入观测（OBS）对比表

| 模块 | Train Rollout | Finetune Rollout | Train Training | Finetune Training |
|------|--------------|-----------------|----------------|------------------|
| **encoder_priv** | `[OBS_PRIV, OBJECT]`<br>(Teacher 轨迹) | ❌ 不使用 | `[OBS_PRIV, OBJECT]`<br>(Teacher 轨迹) | `[OBS_PRIV, OBJECT]`<br>(Student 轨迹) |
| **actor** | `[CMD, OBS, PRIV_FEATURE]`<br>(Teacher 轨迹) | ❌ 不使用 | `[CMD, OBS, PRIV_FEATURE]`<br>(Teacher 轨迹) | ❌ 不使用 |
| **actor_adapt** | ❌ 不使用 | `[CMD, OBS, PRIV_PRED]`<br>(Student 轨迹) | `[CMD, OBS, PRIV_PRED]`<br>(Teacher 轨迹) | `[CMD, OBS, PRIV_PRED]`<br>(Student 轨迹) |
| **adapt_module** | `[OBS, CMD, OBJECT]`<br>(Teacher 轨迹) | ❌ 不使用 | `[OBS, CMD, OBJECT]`<br>(Teacher 轨迹) | `[OBS, CMD, OBJECT]`<br>(Student 轨迹) |
| **adapt_ema** | ❌ 不使用 | `[OBS, CMD, OBJECT]`<br>(Student 轨迹) | EMA 更新 | EMA 更新 |
| **critic** | ❌ 不使用 | ❌ 不使用 | `[OBS_PRIV, OBS, CMD, OBJECT]`<br>(Teacher 轨迹) | `[OBS_PRIV, OBS, CMD, OBJECT]`<br>(Student 轨迹) |

---

### 2. 监督信号对比表

| 模块 | Train | Finetune | 监督信号来源 |
|------|-------|----------|------------|
| **encoder_priv** | PPO loss 反向传播 | ❌ 不训练 | actor 的策略梯度 |
| **actor** | PPO loss | ❌ 不训练 | 环境奖励 (policy gradient) |
| **actor_adapt** | 监督蒸馏 (MSE) | PPO loss | Train: Teacher 动作<br>Finetune: 环境奖励 |
| **adapt_module** | MSE | MSE | encoder_priv 的输出 (PRIV_FEATURE) |
| **critic** | MSE(value, return) | MSE(value, return) | GAE 计算的 return |

---

### 3. 数据分布差异

虽然输入的**键名和维度**在两个阶段完全相同，但**数值的统计分布**不同：

| 观测类型 | Train 阶段 | Finetune 阶段 |
|---------|-----------|--------------|
| **OBS_KEY** | 反映 Teacher 行为模式<br>(使用完美特权信息) | 反映 Student 行为模式<br>(使用推断的特权信息) |
| **OBJECT_KEY** | 物体状态在 Teacher 控制下 | 物体状态在 Student 控制下 |
| **数据质量** | 最优轨迹 | 次优轨迹（学习中） |
| **行为分布** | P_teacher(s\|π_teacher) | P_student(s\|π_student) |

**为什么这很重要？**
- adapt_module 在 Train 阶段学习 Teacher 分布下的推断
- 在 Finetune 阶段继续在 Student 分布下优化
- 这是 **Domain Adaptation** 的关键

---

### 4. 最关键的发现

#### ⚠️ Finetune 仍需特权观测进行训练

```python
Finetune Rollout:  ❌ 不需要 OBS_PRIV_KEY
Finetune Training: ✅ 需要 OBS_PRIV_KEY (用于监督 adapt_module)
```

**含义**：
- Finetune 阶段必须在仿真环境中进行（能访问特权观测）
- 不能在真实机器人上进行 Finetune
- 真机部署只用 `adapt_ema` 和 `actor_adapt`，不再更新

---

#### 🔄 actor_adapt 的训练方式转变

| 阶段 | 训练方式 | 损失函数 | 优化器 |
|------|---------|---------|--------|
| Train | 监督学习 | MSE(student_action, teacher_action) | opt_adapt_actor |
| Finetune | 强化学习 | PPO loss (policy gradient) | opt_policy |

**意义**：
- Train: 先模仿专家（快速获得基础能力）
- Finetune: 再自主探索（适应不完美的推断）

---

#### 📊 adapt_module 的持续学习

```python
# 两阶段都执行相同的训练代码
priv_loss = MSE(PRIV_PRED, PRIV_FEATURE)
opt_adapt.step()

# 但数据分布不同
Train:    OBS ~ P_teacher(s|π_teacher)
Finetune: OBS ~ P_student(s|π_student)
```

**目的**：Domain Adaptation
- 确保 adapt_module 在 Student 的实际行为分布下仍能准确推断
- 避免分布偏移（distribution shift）问题

---

## 部署阶段（真机）

**仅使用**：
```
adapt_ema([OBS, CMD, OBJECT]) → PRIV_PRED
actor_adapt([CMD, OBS, PRIV_PRED]) → action
```

**不需要**：
- encoder_priv (需要特权观测)
- actor (Teacher 策略)
- critic (价值网络)
- adapt_module (训练版本，用 EMA 版本)
- OBS_PRIV_KEY (特权观测)

**部署条件**：
- ✅ 必须有 OBS_KEY (本体观测)
- ✅ 必须有 CMD_KEY (命令)
- ✅ 必须有 OBJECT_KEY (物体状态) - 可能需要视觉估计器
- ❌ 不需要 OBS_PRIV_KEY

---

## 代码位置索引

| 功能 | 文件位置 |
|------|---------|
| **模块定义** | `ppo_roa.py` 第201-272行 |
| **Rollout 策略** | `get_rollout_policy()` 第406-439行 |
| **训练入口** | `train_op()` 第441-465行 |
| **PPO 训练** | `train_policy()` → `_update_ppo()` 第653-734行 |
| **Adapt 训练** | `train_adapt()` 第508-564行 |
| **EMA 更新** | `soft_copy_()` 第561行，定义在 `common.py` |
| **优化器配置** | 第347-396行 |
| **配置定义** | 第88-93行 |

---

**文档生成时间**: 2025-01-XX
**基于代码**: `active_adaptation/learning/ppo/ppo_roa.py`
**参考文档**: `PPO_ROA_Architecture_Analysis.md`
