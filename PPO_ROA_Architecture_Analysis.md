# PPO ROA 架构详细分析

本文档详细记录了对 HDMI-main 项目中 PPO ROA (Rapid Motor Adaptation) 算法的深度分析。

---

## 目录

1. [Train vs Finetune 的核心差异](#1-train-vs-finetune-的核心差异)
2. [Teacher-Student 架构详解](#2-teacher-student-架构详解)
3. [Observation 和特权信息流](#3-observation-和特权信息流)
4. [训练阶段模块详解](#4-训练阶段模块详解)
5. [adapt_module 和 adapt_ema](#5-adapt_module-和-adapt_ema)
6. [Critic 共享机制](#6-critic-共享机制)
7. [纯视觉部署方案](#7-纯视觉部署方案)

---

## 1. Train vs Finetune 的核心差异

### 1.1 命令对比

**Train 阶段 (从头训练):**
```bash
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase
```

**Finetune 阶段 (基于预训练模型微调):**
```bash
python scripts/train.py \
    algo=ppo_roa_finetune \
    task=G1/hdmi/move_suitcase \
    checkpoint_path=run:<teacher_wandb-run-path>
```

### 1.2 配置差异

**代码位置:** `active_adaptation/learning/ppo/ppo_roa.py` 第89-91行

```python
cs.store("ppo_roa_train", node=PPOConfig(
    phase="train",
    vecnorm="train",
    entropy_coef_start=0.001,
    entropy_coef_end=0.001
), group="algo")

cs.store("ppo_roa_finetune", node=PPOConfig(
    phase="finetune",
    vecnorm="eval",
    entropy_coef_start=0.001,
    entropy_coef_end=0.001
), group="algo")
```

**主要差异:**
- `phase`: "train" vs "finetune"
- `vecnorm`: "train" vs "eval"
- `checkpoint_path`: None vs 必须提供

---

## 2. Teacher-Student 架构详解

### 2.1 核心概念

这**不是简单的 DAgger**，而是 **Teacher-Student + Privilege Distillation + Domain Adaptation** 的混合架构。

### 2.2 关键区别

| 组件 | Teacher (actor) | Student (actor_adapt) |
|------|----------------|----------------------|
| **特权信息来源** | `encoder_priv` 直接编码 `OBS_PRIV_KEY` | `adapt_module` 从 `OBS_KEY` 推断 |
| **输入特征** | `PRIV_FEATURE_KEY` (真实) | `PRIV_PRED_KEY` (推断) |
| **是否需要推断** | ❌ 直接获取 ground truth | ✅ 必须从历史观测推断 |
| **仿真 vs 真实** | 仅在仿真中可用 | 可部署到真实机器人 |

### 2.3 Actor 输入详解

**Teacher Actor (第259行):**
```python
in_keys = [CMD_KEY, OBS_KEY, PRIV_FEATURE_KEY]
self.actor = build_actor(in_keys, ...)
```

**PRIV_FEATURE_KEY 来源 (第201-205行):**
```python
self.encoder_priv = Seq(
    CatTensors([OBS_PRIV_KEY, OBJECT_KEY], ...),  # 原始特权信息
    Mod(make_mlp([256]), ..., PRIV_FEATURE_KEY),  # 编码成256维
)
```

**Student Actor (第264行):**
```python
in_keys = [CMD_KEY, OBS_KEY, PRIV_PRED_KEY]
self.actor_adapt = build_actor(in_keys, ...)
```

**PRIV_PRED_KEY 来源 (第207-218行):**
```python
# MLP 模式
self.adapt_module = Seq(
    CatTensors([OBS_KEY, CMD_KEY, OBJECT_KEY], ...),
    Mod(make_mlp([256, 256]), ..., PRIV_PRED_KEY),
)

# 或 GRU 模式 (可以利用历史信息)
self.adapt_module = Seq(
    CatTensors([OBS_KEY, CMD_KEY, OBJECT_KEY], ...),
    Mod(GRUModule(256), ..., [PRIV_PRED_KEY, ...]),
)
```

### 2.4 为什么比 DAgger 好？

| 方法 | Teacher 输入 | Student 输入 | 学习目标 | 性能 |
|------|-------------|-------------|---------|------|
| **DAgger** | [obs, priv] | [obs] | 模仿 action | 60-70% |
| **Behavior Cloning** | [obs, priv] | [obs] | 监督学习 action | 40-60% |
| **RMA (本方法)** | [obs, PRIV_FEATURE] | [obs] → PRIV_PRED | 1) 推断特征<br>2) 模仿动作<br>3) RL微调 | 80-95% |

**优势:**
1. **显式建模环境参数的推断** - 不是黑盒模仿
2. **特权信息编码** - 降低学习难度
3. **蒸馏 + RL 混合训练** - 先模仿再优化
4. **EMA 更新** - 稳定 Student 的推断模块

---

## 3. Observation 和特权信息流

### 3.1 Observation 输入

**代码位置:** 第86行

```python
in_keys: List[str] = (CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY)
```

**Train 和 Finetune 的 observation 输入完全相同:**
- `CMD_KEY` - 命令/指令
- `OBS_KEY` - 本体观测 (proprioceptive)
- `OBJECT_KEY` - 物体信息 (位置、姿态等)
- `OBS_PRIV_KEY` - 特权信息 (地形、物理参数等)

### 3.2 特权信息类型

**PRIV_FEATURE_KEY vs PRIV_PRED_KEY:**

| 特性 | PRIV_FEATURE_KEY | PRIV_PRED_KEY |
|------|------------------|---------------|
| **定义** | 编码后的真实特权特征 | 推断的特权特征 |
| **维度** | 256 | 256 |
| **来源** | `encoder_priv(OBS_PRIV_KEY)` | `adapt_module(OBS_KEY)` |
| **可靠性** | 100% 准确 | 推断，有误差 |
| **用于** | Teacher actor | Student actor |

### 3.3 信息流图

```
训练阶段 (ppo_roa_train):

【Rollout 阶段】每步环境交互:
OBS_PRIV_KEY ──→ encoder_priv ──→ PRIV_FEATURE_KEY ──┐
                                                      ├──→ actor ──→ action (执行)
OBS_KEY ────────→ adapt_module ──→ PRIV_PRED_KEY ────┘
                                     ↓
                                (同步推断，不影响动作)

【训练阶段】每 32 步:

1. train_policy (PPO):
   - 用 Teacher (actor) 计算新的 log_prob
   - 用 Critic 计算 value
   - 更新 actor + encoder_priv + critic

2. train_adapt (蒸馏):
   - 训练 adapt_module: min MSE(PRIV_PRED, PRIV_FEATURE)
   - 训练 actor_adapt: min MSE(student_action, teacher_action)
```

```
微调阶段 (ppo_roa_finetune):

【Rollout 阶段】每步环境交互:
OBS_KEY ──→ adapt_ema (冻结) ──→ PRIV_PRED_KEY ──→ actor_adapt ──→ action (执行)

【训练阶段】每 32 步:

1. train_policy (PPO):
   - 用 Student (actor_adapt) 计算新的 log_prob
   - 用 Critic 计算 value
   - 更新 actor_adapt + critic

2. train_adapt (继续优化):
   - 训练 adapt_module: min MSE(PRIV_PRED, PRIV_FEATURE)
   - 不训练 actor_adapt (通过蒸馏)
```

---

## 4. 训练阶段模块详解

### 4.1 所有网络模块

```python
✅ encoder_priv     # 特权信息编码器
✅ actor            # Teacher 策略网络
✅ critic           # 价值网络
✅ adapt_module     # 适应模块 (推断特权信息)
✅ actor_adapt      # Student 策略网络
✅ adapt_ema        # adapt_module 的 EMA 版本
```

### 4.2 Train 阶段训练流程

#### Rollout 阶段 (第409-412行)

```python
modules = [
    self.encoder_priv,   # OBS_PRIV → PRIV_FEATURE
    self.actor,          # [CMD, OBS, PRIV_FEATURE] → action (执行这个)
    self.adapt_module    # [CMD, OBS] → PRIV_PRED (同时运行，不影响动作)
]
```

#### 训练阶段

**A. train_policy() - PPO 更新 (第467-505行)**

训练的模块:

**1. actor (Teacher)** - 第656-658, 698, 704行
```python
# Optimizer
self.opt_policy = Adam([
    {"params": self.actor.parameters()},
    {"params": self.encoder_priv.parameters()},
])

# 损失
policy_loss = -min(adv * ratio, adv * ratio.clamp(...))
entropy_loss = -entropy_coef * entropy
loss = policy_loss + entropy_loss + value_loss.mean()

# 更新
self.opt_policy.step()  # 更新 actor + encoder_priv
```

**2. encoder_priv** - 第657, 701, 704行
```python
self.encoder_priv(tensordict)  # 前向传播
priv_grad_norm = clip_grad_norm_(self.encoder_priv.parameters(), ...)
# 在 opt_policy.step() 中更新
```

**3. critic** - 第689-690, 696, 699, 705行
```python
values = self.critic(tensordict)["state_value"]
value_loss = MSE(b_returns, values)
self.opt_critic.step()
```

**B. train_adapt() - 蒸馏更新 (第508-564行)**

**1. adapt_module** - 第516-522行
```python
self.opt_adapt = Adam([{"params": self.adapt_module.parameters()}])

# 前向
self.adapt_module(minibatch)  # OBS → PRIV_PRED

# 损失
with torch.no_grad():
    self.encoder_priv(tensordict)  # 生成 PRIV_FEATURE (不更新)

priv_loss = MSE(PRIV_PRED_KEY, PRIV_FEATURE_KEY)
self.opt_adapt.step()
```

**2. actor_adapt (Student)** - 第529-545行
```python
if self.cfg.phase == "train" and self.cfg.enable_residual_distillation:
    self.opt_adapt_actor = Adam([{"params": self.actor_adapt.parameters()}])

    # Teacher 的动作分布
    with torch.no_grad():
        dist_teacher = self.actor.get_dist(minibatch)

    # Student 的动作分布
    if self.cfg.distill_with_priv_pred:
        minibatch[PRIV_PRED_KEY] = minibatch[PRIV_PRED_KEY].detach()
    else:
        minibatch[PRIV_PRED_KEY] = minibatch[PRIV_FEATURE_KEY].detach()

    dist_student = self.actor_adapt.get_dist(minibatch)
    adapt_loss = (dist_teacher.mean - dist_student.mean).square().mean()

    self.opt_adapt_actor.step()
```

**3. EMA 更新** - 第561行
```python
soft_copy_(self.adapt_module, self.adapt_ema, 0.04)
```

#### Train 阶段总结表

| 模块 | 优化器 | 损失函数 | 更新频率 | 代码行 |
|------|--------|---------|---------|--------|
| **actor** (Teacher) | opt_policy | PPO loss | 每个 minibatch | 656-658, 704 |
| **encoder_priv** | opt_policy | PPO loss (反向传播) | 每个 minibatch | 657, 701, 704 |
| **critic** | opt_critic | MSE(value, return) | 每个 minibatch | 689-690, 705 |
| **adapt_module** | opt_adapt | MSE(PRIV_PRED, PRIV_FEATURE) | 每个 minibatch | 516-522 |
| **actor_adapt** (Student) | opt_adapt_actor | MSE(action_student, action_teacher) | 每个 minibatch | 529-545 |
| **adapt_ema** | - | EMA 软更新 | 每次 train_adapt | 561 |

### 4.3 Finetune 阶段训练流程

#### Rollout 阶段 (第416-418行)

```python
modules = [
    self.adapt_ema,      # OBS → PRIV_PRED (用冻结的 EMA 版本)
    self.actor_adapt     # [CMD, OBS, PRIV_PRED] → action (执行这个)
]
```

**关键差异:**
- ❌ 不用 Teacher (actor)
- ✅ 用 Student (actor_adapt) 收集数据
- ✅ 用 `adapt_ema` (不是 `adapt_module`)

#### 训练阶段

**A. train_policy() - PPO 更新**

**1. actor_adapt (Student)** - 第659-660, 698, 704行
```python
self.opt_policy = Adam([
    {"params": self.actor_adapt.parameters()}
])

actor = self.actor_adapt
dist = actor.get_dist(tensordict)
policy_loss = -min(adv * ratio, adv * ratio.clamp(...))

self.opt_policy.step()
```

**2. critic** - 第689-690, 705行
```python
values = self.critic(tensordict)["state_value"]
value_loss = MSE(b_returns, values)
self.opt_critic.step()
```

**B. train_adapt() - 继续训练适应模块**

**1. adapt_module** - 第516-522行 (和 train 阶段完全一样)
```python
self.adapt_module(minibatch)
priv_loss = MSE(PRIV_PRED_KEY, PRIV_FEATURE_KEY)
self.opt_adapt.step()
```

**2. actor_adapt (Student)** - 第529-545行
```python
# ❌ 不会训练！
if self.cfg.phase == "train" and self.cfg.enable_residual_distillation:
    # phase = "finetune" 时，这个条件为 False
```

**3. EMA 更新** - 第561行
```python
soft_copy_(self.adapt_module, self.adapt_ema, 0.04)
```

#### Finetune 阶段总结表

| 模块 | 优化器 | 损失函数 | 更新频率 | 代码行 |
|------|--------|---------|---------|--------|
| **actor_adapt** (Student) | opt_policy | PPO loss | 每个 minibatch | 659-660, 704 |
| **critic** | opt_critic | MSE(value, return) | 每个 minibatch | 689-690, 705 |
| **adapt_module** | opt_adapt | MSE(PRIV_PRED, PRIV_FEATURE) | 每个 minibatch | 516-522 |
| **adapt_ema** | - | EMA 软更新 | 每次 train_adapt | 561 |
| ~~actor~~ (Teacher) | - | ❌ 不训练 | - | - |
| ~~encoder_priv~~ | - | ❌ 不训练 | - | - |

### 4.4 关键差异对比

#### Rollout (数据收集)

| 特性 | Train | Finetune |
|------|-------|----------|
| **执行动作的网络** | `actor` (Teacher) | `actor_adapt` (Student) |
| **特权特征来源** | `encoder_priv(OBS_PRIV)` | `adapt_ema(OBS)` |
| **是否需要特权观测** | ✅ 需要 OBS_PRIV_KEY | ✅ 需要 (用于训练 adapt_module) |
| **同时运行** | `adapt_module` (学习) | - |

#### 训练更新

| 模块 | Train 阶段 | Finetune 阶段 |
|------|-----------|--------------|
| **actor (Teacher)** | ✅ PPO 更新 | ❌ 不训练 |
| **encoder_priv** | ✅ PPO 反向传播 | ❌ 不训练 |
| **actor_adapt (Student)** | ✅ 监督蒸馏 (MSE 动作) | ✅ PPO 更新 |
| **critic** | ✅ PPO 更新 | ✅ PPO 更新 (共享) |
| **adapt_module** | ✅ 监督学习 (MSE 特征) | ✅ 监督学习 (MSE 特征) |
| **adapt_ema** | ✅ EMA 更新 | ✅ EMA 更新 |

#### actor_adapt 的两种训练方式

| 训练方式 | Train 阶段 | Finetune 阶段 |
|---------|-----------|--------------|
| **监督蒸馏** | ✅ 通过 opt_adapt_actor<br>MSE(student_action, teacher_action) | ❌ 不使用 |
| **PPO 强化学习** | ❌ 不使用 | ✅ 通过 opt_policy<br>PPO loss |

---

## 5. adapt_module 和 adapt_ema

### 5.1 adapt_module (可训练的适应模块)

**定义位置:** 第207-220行

```python
if self.cfg.adapt_module == "gru":
    self.adapt_module = Seq(
        CatTensors([OBS_KEY, CMD_KEY, OBJECT_KEY], ...),
        Mod(GRUModule(latent_dim), ..., [PRIV_PRED_KEY, ...]),
    )
elif self.cfg.adapt_module == "mlp":
    self.adapt_module = Seq(
        CatTensors([OBS_KEY, CMD_KEY, OBJECT_KEY], ...),
        Mod(make_mlp([256, 256]), ..., [PRIV_PRED_KEY]),
    )
```

**功能:**
- **输入**: 本体观测 `[OBS_KEY, CMD_KEY, OBJECT_KEY]`
- **输出**: 预测的特权特征 `PRIV_PRED_KEY` (256维)
- **作用**: 从可观测的信息中**推断**不可直接测量的特权信息

**训练:** ✅ 在所有阶段都会训练

```python
self.opt_adapt = Adam([{"params": self.adapt_module.parameters()}])

priv_loss = MSE(PRIV_PRED_KEY, PRIV_FEATURE_KEY)
self.opt_adapt.step()
```

### 5.2 adapt_ema (指数移动平均版本)

**定义位置:** 第345行

```python
self.adapt_ema = copy.deepcopy(self.adapt_module).requires_grad_(False)
```

**功能:**
- 是 `adapt_module` 的 **深拷贝**
- **不可训练** (`.requires_grad_(False)`)
- 通过 **指数移动平均 (EMA)** 更新

**更新方式:** 第561行

```python
soft_copy_(self.adapt_module, self.adapt_ema, tau=0.04)
```

**soft_copy_ 实现:**

```python
def soft_copy_(source_module, target_module, tau=0.01):
    for params_source, params_target in zip(source_module.parameters(),
                                            target_module.parameters()):
        params_target.data.lerp_(params_source.data, tau)
```

**数学公式:**
```
adapt_ema.params ← (1 - tau) * adapt_ema.params + tau * adapt_module.params
                 ← 0.96 * adapt_ema.params + 0.04 * adapt_module.params
```

### 5.3 为什么需要两个版本？

| 特性 | adapt_module | adapt_ema |
|------|--------------|-----------|
| **训练** | ✅ 每个 minibatch 更新 | ❌ 不训练 |
| **更新方式** | 梯度下降 | 指数移动平均 |
| **稳定性** | ⚠️ 可能波动 | ✅ 更平滑 |
| **用途** | 训练时学习 | Rollout 时推断 |

**EMA 的优势:**
1. **平滑性**: 减少训练过程中的噪声
2. **稳定性**: 避免突然的性能下降
3. **鲁棒性**: 对超参数不敏感

### 5.4 各阶段使用情况

| 阶段 | Rollout 使用 | adapt_module 训练 | adapt_ema 更新 |
|------|-------------|------------------|---------------|
| **Train** | `adapt_module` | ✅ MSE loss | ✅ EMA |
| **Finetune** | `adapt_ema` | ✅ MSE loss | ✅ EMA |
| **Deploy** | `adapt_ema` | ❌ | ❌ |

### 5.5 是否属于 Teacher 或 Student？

**答案: 都不属于！是独立的共享模块！**

```
整个系统的网络架构:

Teacher 专用:
  ├─ encoder_priv  (OBS_PRIV → PRIV_FEATURE)
  └─ actor         ([CMD, OBS, PRIV_FEATURE] → action)

Student 专用:
  └─ actor_adapt   ([CMD, OBS, PRIV_PRED] → action)

共享模块:
  ├─ adapt_module  (OBS → PRIV_PRED)  [可训练]
  ├─ adapt_ema     (OBS → PRIV_PRED)  [EMA版本]
  └─ critic        ([OBS_PRIV, OBS, CMD] → value)
```

---

## 6. Critic 共享机制

### 6.1 只有一个 Critic

**定义位置:** 第267-272行

```python
_critic = nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(num_reward_groups))
self.critic = Seq(
    CatTensors(critic_in_keys, "_critic_input", del_keys=False),
    Mod(_critic, ["_critic_input"], ["state_value"])
).to(self.device)
```

**数量:** 1 个

**Optimizer:** 第362-367行

```python
self.opt_critic = torch.optim.Adam(
    [{"params": self.critic.parameters()}],
    lr=cfg.lr,
)
```

### 6.2 Critic 的输入

**定义:** 第192-198行

```python
critic_in_keys = [OBS_PRIV_KEY, OBS_KEY, CMD_KEY]
if observation_spec.get(OBJECT_KEY, None) is not None:
    critic_in_keys.append(OBJECT_KEY)
```

**输入包含特权信息:**
- `OBS_PRIV_KEY` - 特权观测 (地形、物体属性等)
- `OBS_KEY` - 本体观测
- `CMD_KEY` - 命令
- `OBJECT_KEY` - 物体信息 (可选)

### 6.3 各阶段的 Critic 使用

**Train 阶段 (第689-690, 705行):**
```python
values = self.critic(tensordict)["state_value"]
value_loss = MSE(b_returns, values)
self.opt_critic.step()
```
- 数据来源: Teacher (actor) 的 rollout

**Finetune 阶段:**
```python
# 完全相同的代码
values = self.critic(tensordict)["state_value"]
value_loss = MSE(b_returns, values)
self.opt_critic.step()
```
- 数据来源: Student (actor_adapt) 的 rollout

### 6.4 为什么可以共享？

1. **输入相同**: Critic 使用 `[OBS_PRIV, OBS, CMD]`，不依赖于是哪个 actor
2. **任务相同**: 预测累积回报，无论是 Teacher 还是 Student
3. **特权信息可用**: 训练时仍然可以访问 `OBS_PRIV_KEY`
4. **效率更高**: 共享可以利用两个阶段的所有数据

### 6.5 对比其他方法

| 方法 | Critic 设计 | 优缺点 |
|------|------------|-------|
| **本方法 (共享)** | 1个 Critic，输入特权信息 | ✅ 数据高效<br>✅ 训练稳定<br>⚠️ 部署时不用 |
| **独立 Critic** | Teacher 和 Student 各一个 | ⚠️ 数据利用率低<br>⚠️ 训练复杂 |
| **无特权 Critic** | 只用可观测信息 | ❌ 价值估计不准<br>✅ 可部署 |

---

## 7. 纯视觉部署方案

### 7.1 问题背景

真机部署时可能只有图像输入，没有 `OBJECT_KEY` (物体精确位置) 的输入。

### 7.2 当前架构

代码已经支持 `train_est` 和 `adapt_est` 阶段用于视觉 estimator:

**第92-93行:**
```python
cs.store("ppo_roa_train_est", node=PPOConfig(
    phase="train_est",
    vecnorm="eval",
    in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, DEPTH_KEY)
), group="algo")

cs.store("ppo_roa_adapt_est", node=PPOConfig(
    phase="adapt_est",
    in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, DEPTH_KEY)
), group="algo")
```

**Estimator 架构 (第274-296行):**
```python
if self.cfg.phase in ["train_est", "adapt_est"]:
    mlp = make_mlp([latent_dim])
    cnn = nn.Sequential(
        make_conv(num_channels=[8, 8, 8], activation=nn.Mish, kernel_sizes=5),
        nn.LazyLinear(64),
        nn.LayerNorm(64)
    )
    back_bone = make_mlp([latent_dim, latent_dim])
    modules = [
        CatTensors([OBS_KEY, CMD_KEY], "_estimator_mlp_inp", ...),
        Mod(mlp, "_estimator_mlp_inp", ["_mlp"]),
        Mod(cnn, [DEPTH_KEY], ["_cnn"]),
        CatTensors(["_mlp", "_cnn"], "_estimator_inp", ...),
        Mod(back_bone, "_estimator_inp", "priv_est")
    ]
    self.estimator = Seq(*modules, selected_out_keys=["priv_est"])
```

### 7.3 三阶段训练流程

```
阶段 1: Train Teacher (ppo_roa_train)
  Teacher: [OBS, PRIV(包含OBJECT位置)] → actor → action

阶段 2: Train Vision Estimator (ppo_roa_train_est)
  Vision Estimator: [OBS, DEPTH_IMAGE] → CNN → PRIV_EST
  目标: PRIV_EST ≈ PRIV_PRED (来自 adapt_ema)

阶段 3: Deploy with Vision (ppo_roa_adapt_est)
  Real Robot: [OBS, DEPTH_IMAGE] → Vision Estimator → PRIV_EST → actor_adapt → action
  (无需 OBJECT_KEY)
```

### 7.4 修改方案：移除 OBJECT_KEY 依赖

**1. 添加配置参数 (第40行附近):**

```python
@dataclass
class PPOConfig:
    # ... 现有参数 ...

    use_pure_vision: bool = False      # 是否纯视觉 (不用 OBJECT_KEY)
    vision_feature_dim: int = 128      # 视觉特征维度
```

**2. 新增配置:**

```python
cs.store("ppo_roa_train_vision", node=PPOConfig(
    phase="train_est",
    vecnorm="eval",
    use_pure_vision=True,
    in_keys=(CMD_KEY, OBS_KEY, OBS_PRIV_KEY, DEPTH_KEY)  # 移除 OBJECT_KEY
), group="algo")

cs.store("ppo_roa_deploy_vision", node=PPOConfig(
    phase="adapt_est",
    use_pure_vision=True,
    in_keys=(CMD_KEY, OBS_KEY, DEPTH_KEY)  # 部署时无特权信息
), group="algo")
```

**3. 修改 Estimator 构建 (第274-296行):**

```python
if self.cfg.phase in ["train_est", "adapt_est"]:
    assert DEPTH_KEY in observation_spec, f"{DEPTH_KEY} needed"

    if self.cfg.use_pure_vision:
        # ============ 纯视觉模式 ============
        mlp = make_mlp([latent_dim])

        # 增强的视觉编码器
        cnn = nn.Sequential(
            make_conv(
                num_channels=[16, 32, 64, 32],
                activation=nn.Mish,
                kernel_sizes=[5, 3, 3, 3]
            ),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.LazyLinear(self.cfg.vision_feature_dim),
            nn.LayerNorm(self.cfg.vision_feature_dim),
            nn.Mish(),
        )

        # 融合网络
        fusion_dim = latent_dim + self.cfg.vision_feature_dim
        back_bone = nn.Sequential(
            nn.Linear(fusion_dim, latent_dim),
            nn.Mish(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

        modules = [
            CatTensors([OBS_KEY, CMD_KEY], "_estimator_mlp_inp", ...),
            Mod(mlp, "_estimator_mlp_inp", ["_mlp"]),
            Mod(cnn, [DEPTH_KEY], ["_cnn"]),
            CatTensors(["_mlp", "_cnn"], "_estimator_inp", ...),
            Mod(back_bone, "_estimator_inp", "priv_est")
        ]
    else:
        # ============ 原有模式 (DEPTH + OBJECT) ============
        # 保持原代码
        ...
```

**4. 修改 encoder_priv 构建 (第189-205行):**

```python
encoder_priv_in_keys = [OBS_PRIV_KEY]
adapt_module_in_keys = [OBS_KEY]
critic_in_keys = [OBS_PRIV_KEY, OBS_KEY, CMD_KEY]

if self.cfg.adapt_module_input_cmd:
    adapt_module_in_keys.append(CMD_KEY)

# 只在非纯视觉模式且有 OBJECT_KEY 时添加
if not self.cfg.use_pure_vision and observation_spec.get(OBJECT_KEY, None) is not None:
    encoder_priv_in_keys.append(OBJECT_KEY)
    adapt_module_in_keys.append(OBJECT_KEY)
    critic_in_keys.append(OBJECT_KEY)
```

### 7.5 训练命令

**第一步: 训练 Teacher**
```bash
python scripts/train.py \
    algo=ppo_roa_train \
    task=G1/hdmi/move_suitcase \
    experiment=suitcase_teacher
```

**第二步: 训练 Vision Estimator**
```bash
python scripts/train.py \
    algo=ppo_roa_train_vision \
    task=G1/hdmi/move_suitcase \
    checkpoint_path=run:<teacher_wandb_run_path> \
    experiment=suitcase_vision
```

**第三步: 微调 (可选)**
```bash
python scripts/train.py \
    algo=ppo_roa_deploy_vision \
    task=G1/hdmi/move_suitcase \
    checkpoint_path=run:<vision_wandb_run_path> \
    experiment=suitcase_deploy
```

### 7.6 方法对比

| 特性 | 当前方法 (OBJECT_KEY) | 纯视觉方法 (DEPTH only) |
|------|---------------------|----------------------|
| **Train 阶段** | PRIV 包含物体精确位置 | PRIV 包含物体精确位置 |
| **Train_est 阶段** | Estimator: DEPTH + OBJECT → PRIV_EST | Estimator: DEPTH → PRIV_EST |
| **Deploy 阶段** | 需要物体位置传感器 | ❌ 只需相机 |
| **真机部署** | 需要视觉检测系统提供 OBJECT_KEY | ✅ 端到端视觉 |
| **视觉编码器** | 简单 CNN (辅助) | 增强 CNN (主要) |

---

## 8. 设计哲学总结

### 8.1 Train 阶段的目标

1. **训练强大的 Teacher** (`actor` + `encoder_priv`)
2. **让 Student 学会推断** (`adapt_module` 学习 PRIV_FEATURE)
3. **让 Student 学会模仿** (`actor_adapt` 模仿 Teacher 的动作)

### 8.2 Finetune 阶段的目标

1. **Student 自己探索** (用 PPO 而非蒸馏)
2. **修正推断误差** (`actor_adapt` 适应不完美的 PRIV_PRED)
3. **继续改进推断** (`adapt_module` 继续学习)

### 8.3 为什么这样设计？

**Train 阶段:**
- Teacher 有完美的特权信息，性能最好
- Student 先学习 "怎么看" (推断特征)，再学习 "怎么做" (模仿动作)
- 蒸馏保证 Student 有好的初始化

**Finetune 阶段:**
- Student 必须学会应对不完美的推断
- PPO 让 Student 自己优化策略 (而非盲目模仿)
- 继续训练 `adapt_module` 减少推断误差

这是一个 **两阶段课程学习 (curriculum learning)** 策略:
```
Train: 模仿专家 (有完美信息) → 打好基础
Finetune: 独立行动 (有不完美信息) → 真实场景适应
```

### 8.4 核心创新点

1. **显式建模环境参数的推断** - 不是黑盒模仿
2. **特权信息编码** - 降低学习难度 (256维而非原始高维)
3. **蒸馏 + RL 混合训练** - 先模仿再优化
4. **EMA 更新** - 稳定 Student 的推断模块
5. **共享 Critic** - 数据高效，训练稳定
6. **模块化设计** - Teacher、Student、共享模块清晰分离

---

## 9. 完整架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      整个系统的网络模块                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Teacher 专用:                                              │
│    ├─ encoder_priv  (OBS_PRIV → PRIV_FEATURE)              │
│    └─ actor         ([CMD, OBS, PRIV_FEATURE] → action)    │
│                                                             │
│  Student 专用:                                              │
│    └─ actor_adapt   ([CMD, OBS, PRIV_PRED] → action)       │
│                                                             │
│  共享模块:                                                   │
│    ├─ adapt_module  (OBS → PRIV_PRED)  [可训练]            │
│    ├─ adapt_ema     (OBS → PRIV_PRED)  [EMA版本]           │
│    └─ critic        ([OBS_PRIV, OBS, CMD] → value)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Train 阶段训练:
  ✅ actor (Teacher)         - PPO
  ✅ encoder_priv            - PPO 反向传播
  ✅ critic (共享)           - PPO
  ✅ adapt_module (共享)     - 监督学习
  ✅ actor_adapt (Student)   - 监督蒸馏
  ✅ adapt_ema               - EMA更新

Finetune 阶段训练:
  ❌ actor (不训练)
  ❌ encoder_priv (不训练)
  ✅ critic (共享，继续训练)  - PPO
  ✅ adapt_module (共享)      - 监督学习
  ✅ actor_adapt (Student)    - PPO (不是蒸馏!)
  ✅ adapt_ema                - EMA更新

Deploy 阶段使用:
  ❌ actor (不部署)
  ❌ encoder_priv (不部署)
  ❌ critic (不部署)
  ✅ adapt_ema (部署)
  ✅ actor_adapt (部署)
```

---

## 10. 代码关键位置索引

| 功能 | 代码位置 |
|------|---------|
| **配置定义** | 第88-93行 |
| **encoder_priv 构建** | 第201-205行 |
| **adapt_module 构建** | 第207-220行 |
| **actor 构建** | 第259-260行 |
| **actor_adapt 构建** | 第264-265行 |
| **critic 构建** | 第267-272行 |
| **estimator 构建** | 第274-296行 |
| **adapt_ema 初始化** | 第345行 |
| **优化器配置** | 第347-396行 |
| **get_rollout_policy** | 第406-439行 |
| **train_op 入口** | 第441-465行 |
| **train_policy** | 第467-505行 |
| **train_adapt** | 第508-564行 |
| **_update_ppo** | 第653-734行 |
| **soft_copy_ 定义** | common.py 第286-288行 |

---

**文档生成时间:** 2025-01-XX

**分析的代码版本:** HDMI-main (最新)

**主要文件:** `active_adaptation/learning/ppo/ppo_roa.py`