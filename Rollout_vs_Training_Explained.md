# Rollout vs Training 详解

本文档详细解释强化学习中 Rollout 和 Training 两个阶段的区别。

---

## 目录

1. [核心概念](#核心概念)
2. [详细对比](#详细对比)
3. [代码流程](#代码流程)
4. [PPO-ROA 具体实现](#ppo-roa-具体实现)
5. [关键差异总结](#关键差异总结)

---

## 核心概念

### Rollout（数据收集/策略执行阶段）

**定义**：使用**当前策略**与环境交互，收集经验数据（轨迹）

**目的**：
- 生成训练数据（观测、动作、奖励、下一状态）
- 评估当前策略的性能
- 探索环境

**特点**：
- ✅ 策略网络处于**推理模式**（`inference_mode` / `eval` mode）
- ✅ 不计算梯度（`torch.no_grad()` 或 `torch.inference_mode()`）
- ✅ 执行动作并获取环境反馈
- ❌ 不更新网络参数

### Training（训练/优化阶段）

**定义**：使用 Rollout 收集的数据，通过梯度下降**更新策略网络参数**

**目的**：
- 优化策略（使期望回报最大化）
- 学习价值函数
- 改进模型性能

**特点**：
- ✅ 策略网络处于**训练模式**（`train` mode）
- ✅ 计算梯度（`loss.backward()`）
- ✅ 更新网络参数（`optimizer.step()`）
- ❌ 不与环境交互

---

## 详细对比

### 1. 执行模式

| 特性 | Rollout | Training |
|------|---------|----------|
| **网络模式** | `eval()` / `inference_mode()` | `train()` |
| **梯度计算** | ❌ `torch.no_grad()` | ✅ 启用梯度 |
| **随机性** | ✅ 探索（采样动作） | ❌ 确定性（重新评估旧动作） |
| **Dropout** | ❌ 关闭 | ✅ 激活（如果有） |
| **BatchNorm** | ✅ 使用 running stats | ✅ 更新 running stats |

---

### 2. 输入输出

| 方面 | Rollout | Training |
|------|---------|----------|
| **输入来源** | 环境实时观测 | 缓存的 Rollout 数据 |
| **输入类型** | 当前状态 `s_t` | 历史轨迹 `{s_t, a_t, r_t, ...}` |
| **输出用途** | 执行动作 → 环境 | 计算损失 → 更新参数 |
| **数据存储** | 存入 buffer | 从 buffer 读取 |

---

### 3. 时间消耗

| 阶段 | Rollout | Training |
|------|---------|----------|
| **主要操作** | 环境仿真 + 前向推理 | 反向传播 + 参数更新 |
| **瓶颈** | 环境步进速度 | GPU 计算 |
| **典型时间** | 较长（需等待环境） | 较短（纯计算） |
| **频率** | 每步环境交互 | 每 N 步（如 32 步） |

---

### 4. 使用的模块

#### PPO-ROA Train 阶段

| 模块 | Rollout 使用 | Training 使用 | 备注 |
|------|-------------|--------------|------|
| **encoder_priv** | ✅ 推理模式 | ✅ 训练模式 | 编码特权信息 |
| **actor** (Teacher) | ✅ 推理模式 | ✅ 训练模式 | 执行动作 + 更新 |
| **actor_adapt** (Student) | ❌ | ✅ 训练模式 | 仅训练（蒸馏） |
| **adapt_module** | ✅ 推理模式 | ✅ 训练模式 | 推断 + 更新 |
| **adapt_ema** | ❌ | ⚙️ EMA 更新 | 不直接使用 |
| **critic** | ❌ | ✅ 训练模式 | 仅训练 |

#### PPO-ROA Finetune 阶段

| 模块 | Rollout 使用 | Training 使用 | 备注 |
|------|-------------|--------------|------|
| **encoder_priv** | ❌ | ✅ 推理模式 | 生成监督信号（冻结） |
| **actor** (Teacher) | ❌ | ❌ | 不使用 |
| **actor_adapt** (Student) | ✅ 推理模式 | ✅ 训练模式 | 执行动作 + 更新 |
| **adapt_module** | ❌ | ✅ 训练模式 | 仅训练 |
| **adapt_ema** | ✅ 推理模式 | ⚙️ EMA 更新 | 执行推断 |
| **critic** | ❌ | ✅ 训练模式 | 仅训练 |

---

## 代码流程

### 完整训练循环

```python
# 初始化
env = make_env(...)
policy = make_policy(...)
data_buf = TensorDict({}, batch_size=[N, T])  # N=环境数, T=steps_per_update

carry = env.reset()  # 初始观测

for i in range(total_iters):
    # ============ ROLLOUT 阶段 ============
    with torch.inference_mode():  # 关闭梯度计算
        for step in range(train_every):  # 例如 32 步
            # 1. 策略推理（生成动作）
            carry = rollout_policy(carry)

            # 2. 环境步进
            td, carry = env.step_and_maybe_reset(carry)

            # 3. 存储数据
            data_buf[:, step] = td

        # 4. 计算价值函数（用于 GAE）
        values = policy.critic(data_buf)["state_value"]

    # ============ TRAINING 阶段 ============
    # 5. 更新策略
    info = policy.train_op(data_buf)  # 使用收集的数据训练

    # 6. 记录日志
    wandb.log(info)
```

---

### Rollout 详细流程

```python
# 代码位置: train.py 第180-226行

rollout_policy = policy.get_rollout_policy("train")

with torch.inference_mode():  # ❌ 不计算梯度
    for step in range(cfg.algo.train_every):  # 默认 32 步
        # ========== 策略推理 ==========
        carry = rollout_policy(carry)
        # carry 包含:
        #   - observation (OBS_KEY, OBS_PRIV_KEY, CMD_KEY, OBJECT_KEY)
        #   - action (rollout_policy 输出)
        #   - sample_log_prob (动作的对数概率)

        # ========== 环境步进 ==========
        td, carry = env.step_and_maybe_reset(carry)
        # td 包含:
        #   - reward (即时奖励)
        #   - done (是否结束)
        #   - next.observation (下一状态)

        # ========== 存储数据 ==========
        data_buf[:, step] = td

    # ========== 计算价值 ==========
    policy.critic(data_buf)  # 批量计算所有状态的价值
```

**关键点**：
- **不更新参数**，只收集数据
- **动作采样**：从策略分布中采样（探索）
- **环境交互**：执行动作，获取奖励和下一状态

---

### Training 详细流程

```python
# 代码位置: train.py 第239行 → ppo_roa.py train_op()

info = policy.train_op(data_buf)

# ========== train_op() 内部 ==========
# 代码位置: ppo_roa.py 第441-465行

def train_op(self, tensordict: TensorDict):
    info = {}

    if self.cfg.phase == "train":
        # Train 阶段
        info.update(self.train_policy(tensordict.copy()))  # PPO 更新
        info.update(self.train_adapt(tensordict.copy()))   # Adapt 更新

    elif self.cfg.phase == "finetune":
        # Finetune 阶段
        info.update(self.train_policy(tensordict.copy()))  # PPO 更新
        info.update(self.train_adapt(tensordict.copy()))   # Adapt 更新

    return info
```

#### train_policy（PPO 更新）

```python
# 代码位置: ppo_roa.py 第467-505行

def train_policy(self, tensordict: TensorDict):
    # 计算优势函数（GAE）
    with torch.no_grad():
        adv = compute_gae(...)
        ret = adv + values

    # 多轮更新（默认 5 epochs）
    for epoch in range(self.cfg.num_epochs):
        for minibatch in make_batch(tensordict, num_minibatches):
            # ========== PPO 更新 ==========
            info = self._update_ppo(minibatch)

    return info

# ========== _update_ppo() 内部 ==========
# 代码位置: ppo_roa.py 第653-734行

def _update_ppo(self, tensordict):
    if self.cfg.phase == "train":
        # 使用 Teacher
        self.encoder_priv(tensordict)  # 生成 PRIV_FEATURE
        actor = self.actor
    else:
        # 使用 Student
        actor = self.actor_adapt

    # 1. 前向传播（重新计算动作概率）
    dist = actor.get_dist(tensordict)
    log_probs = dist.log_prob(tensordict["action"])  # 已执行的动作

    # 2. 计算 PPO loss
    ratio = exp(log_probs - tensordict["sample_log_prob"])
    surr1 = adv * ratio
    surr2 = adv * ratio.clamp(1-ε, 1+ε)
    policy_loss = -min(surr1, surr2).mean()

    # 3. Critic loss
    values = self.critic(tensordict)["state_value"]
    value_loss = MSE(values, ret)

    # 4. 反向传播 + 更新
    loss = policy_loss + value_loss
    loss.backward()  # ✅ 计算梯度
    self.opt_policy.step()  # ✅ 更新参数
    self.opt_critic.step()
```

#### train_adapt（Adapt 更新）

```python
# 代码位置: ppo_roa.py 第508-564行

def train_adapt(self, tensordict: TensorDict):
    # 生成监督信号
    with torch.no_grad():
        self.encoder_priv(tensordict)  # PRIV_FEATURE (ground truth)

    # 多轮更新
    for epoch in range(2):
        for minibatch in make_batch(tensordict, num_minibatches):
            # ========== 1. 训练 adapt_module ==========
            self.adapt_module(minibatch)  # 推断 PRIV_PRED
            priv_loss = MSE(PRIV_PRED, PRIV_FEATURE)
            priv_loss.backward()
            self.opt_adapt.step()  # 更新 adapt_module

            # ========== 2. 训练 actor_adapt (仅 Train 阶段) ==========
            if self.cfg.phase == "train":
                dist_teacher = self.actor.get_dist(minibatch)  # 冻结
                dist_student = self.actor_adapt.get_dist(minibatch)
                adapt_loss = MSE(dist_student.mean, dist_teacher.mean)
                adapt_loss.backward()
                self.opt_adapt_actor.step()  # 更新 actor_adapt

    # ========== 3. EMA 更新 ==========
    soft_copy_(self.adapt_module, self.adapt_ema, tau=0.04)
```

**关键点**：
- **更新参数**：通过梯度下降
- **使用历史数据**：Rollout 收集的轨迹
- **不与环境交互**：纯计算过程

---

## PPO-ROA 具体实现

### Train 阶段

#### Rollout（32步环境交互）

```
时间: t0 → t31

每步执行:
1. encoder_priv([OBS_PRIV, OBJECT]) → PRIV_FEATURE
2. actor([CMD, OBS, PRIV_FEATURE]) → action (采样) ✅ 执行
3. adapt_module([OBS, CMD, OBJECT]) → PRIV_PRED (旁观)
4. env.step(action) → reward, next_obs

收集数据:
├─ OBS_KEY, OBS_PRIV_KEY, CMD_KEY, OBJECT_KEY
├─ PRIV_FEATURE_KEY, PRIV_PRED_KEY
├─ action, sample_log_prob
└─ reward, done, next
```

#### Training（使用32步数据）

```
阶段 1: train_policy (5 epochs × 4 minibatches)
├─ 输入: 32步历史数据
├─ 前向: encoder_priv → actor → new_log_prob
│        critic → value
├─ 损失: PPO loss + value loss
└─ 更新: actor, encoder_priv, critic

阶段 2: train_adapt (2 epochs × 4 minibatches)
├─ 输入: 32步历史数据
├─ 损失1: MSE(PRIV_PRED, PRIV_FEATURE)
├─ 更新1: adapt_module
├─ 损失2: MSE(actor_adapt, actor)
├─ 更新2: actor_adapt
└─ EMA: adapt_ema ← 0.96*adapt_ema + 0.04*adapt_module
```

---

### Finetune 阶段

#### Rollout（32步环境交互）

```
时间: t0 → t31

每步执行:
1. adapt_ema([OBS, CMD, OBJECT]) → PRIV_PRED (推断)
2. actor_adapt([CMD, OBS, PRIV_PRED]) → action (采样) ✅ 执行
3. env.step(action) → reward, next_obs

收集数据:
├─ OBS_KEY, OBS_PRIV_KEY, CMD_KEY, OBJECT_KEY
├─ PRIV_PRED_KEY (来自 adapt_ema)
├─ action, sample_log_prob
└─ reward, done, next
```

#### Training（使用32步数据）

```
阶段 1: train_policy (5 epochs × 4 minibatches)
├─ 输入: 32步历史数据
├─ 前向: actor_adapt → new_log_prob
│        critic → value
├─ 损失: PPO loss + value loss
└─ 更新: actor_adapt, critic

阶段 2: train_adapt (2 epochs × 4 minibatches)
├─ 输入: 32步历史数据
├─ 前向: adapt_module → PRIV_PRED
│        encoder_priv → PRIV_FEATURE (冻结)
├─ 损失: MSE(PRIV_PRED, PRIV_FEATURE)
├─ 更新: adapt_module
└─ EMA: adapt_ema ← 0.96*adapt_ema + 0.04*adapt_module
```

---

## 关键差异总结

### 1. 核心目的

| 方面 | Rollout | Training |
|------|---------|----------|
| **主要目的** | 收集经验数据 | 优化策略参数 |
| **次要目的** | 评估当前策略 | 学习价值函数 |

---

### 2. 计算图

| 特性 | Rollout | Training |
|------|---------|----------|
| **梯度计算** | ❌ `torch.no_grad()` | ✅ `loss.backward()` |
| **前向传播** | ✅ 一次（推理） | ✅ 多次（重新评估） |
| **反向传播** | ❌ 无 | ✅ 有 |
| **参数更新** | ❌ 无 | ✅ `optimizer.step()` |

---

### 3. 数据流向

```
Rollout:
环境 → 观测 → 策略网络 → 动作 → 环境 → 奖励 → Buffer

Training:
Buffer → 数据 → 策略网络 → 损失 → 梯度 → 优化器 → 更新参数
```

---

### 4. 时间分布

```
训练循环（1 iteration）:

[Rollout: 32 steps]
├─ step 0:  env.step() + policy.forward()
├─ step 1:  env.step() + policy.forward()
├─ ...
└─ step 31: env.step() + policy.forward()
    ↓
[Training: 5 epochs × 4 minibatches]
├─ epoch 0: [mb0, mb1, mb2, mb3] → backward + update
├─ epoch 1: [mb0, mb1, mb2, mb3] → backward + update
├─ ...
└─ epoch 4: [mb0, mb1, mb3, mb3] → backward + update
```

**时间比例**（典型值）：
- Rollout: 70-80% 总时间（环境瓶颈）
- Training: 20-30% 总时间（GPU 计算）

---

### 5. 网络状态

#### Rollout

```python
with torch.inference_mode():  # 禁用梯度
    rollout_policy.eval()      # 评估模式
    carry = rollout_policy(carry)
```

**特点**：
- BatchNorm 使用 running statistics
- Dropout 关闭
- 不构建计算图

#### Training

```python
# 自动启用梯度
policy.train()  # 训练模式（如果有 BN/Dropout）
loss.backward()  # 构建计算图
optimizer.step()  # 更新参数
```

**特点**：
- BatchNorm 更新 running statistics
- Dropout 激活
- 构建完整计算图

---

### 6. 动作选择

| 方面 | Rollout | Training |
|------|---------|----------|
| **动作生成** | 从分布**采样**（探索） | 重新计算**对数概率** |
| **随机性** | ✅ 有（exploration） | ❌ 无（使用已执行的动作） |
| **目的** | 生成新经验 | 评估旧动作的质量 |

**代码对比**：

```python
# Rollout: 采样新动作
with torch.inference_mode():
    dist = policy.get_dist(obs)
    action = dist.sample()  # ✅ 采样（探索）
    log_prob = dist.log_prob(action)

# Training: 评估旧动作
dist = policy.get_dist(obs)
log_prob_new = dist.log_prob(old_action)  # ✅ 重新计算（评估）
ratio = exp(log_prob_new - log_prob_old)  # PPO 比率
```

---

### 7. 频率

| 方面 | Rollout | Training |
|------|---------|----------|
| **执行频率** | 每步环境交互 | 每 N 步（如 32 步） |
| **典型设置** | 持续执行 | 间歇更新 |
| **并行度** | 环境并行 | Batch 并行 |

**示例**：
```
Iteration 0:
├─ Rollout: step 0, 1, 2, ..., 31 (32次环境步进)
└─ Training: 5 epochs × 4 minibatches (20次参数更新)

Iteration 1:
├─ Rollout: step 32, 33, 34, ..., 63
└─ Training: 5 epochs × 4 minibatches
...
```

---

### 8. 关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| **Rollout 主循环** | `train.py` | 180-226 |
| **Training 入口** | `train.py` | 239 |
| **get_rollout_policy** | `ppo_roa.py` | 406-439 |
| **train_op** | `ppo_roa.py` | 441-465 |
| **train_policy** | `ppo_roa.py` | 467-505 |
| **train_adapt** | `ppo_roa.py` | 508-564 |
| **_update_ppo** | `ppo_roa.py` | 653-734 |

---

## 类比理解

### 现实类比

**Rollout = 考试答题**
- 学生在考场上答题（执行策略）
- 收集答案和成绩（收集数据）
- 不允许修改知识（不更新参数）

**Training = 考后总结**
- 分析错题（计算损失）
- 理解知识点（反向传播）
- 改进学习方法（更新参数）

---

### 代码类比

```python
# Rollout = 函数调用（只用，不改）
result = model(input)  # 推理

# Training = 函数优化（改进）
loss = criterion(model(input), target)
loss.backward()   # 分析问题
optimizer.step()  # 改进模型
```

---

## 常见误区

### ❌ 误区 1：Rollout 也在训练

**错误理解**：Rollout 时网络参数会更新

**正确理解**：
- Rollout 只是**推理**（inference）
- 参数完全不变
- 只有 Training 阶段才更新参数

---

### ❌ 误区 2：Training 需要环境交互

**错误理解**：Training 时需要执行动作

**正确理解**：
- Training 使用**缓存数据**
- 不需要环境交互
- 纯计算过程

---

### ❌ 误区 3：Rollout 和 Training 用的是不同网络

**错误理解**：两个阶段用不同的模型

**正确理解**：
- 是**同一个网络**
- Rollout 用当前参数推理
- Training 用同一网络更新参数

---

## 总结图表

```
┌─────────────────────────────────────────────────────────────┐
│                    强化学习训练循环                            │
└─────────────────────────────────────────────────────────────┘

Iteration i:

┌─────────────────── ROLLOUT ───────────────────┐
│                                               │
│  with torch.inference_mode():                │
│    for step in range(32):                    │
│      carry = policy(carry)  ←──┐ 推理模式     │
│      td, carry = env.step(carry) │ 不计算梯度 │
│      buffer.store(td)        ←──┘ 收集数据    │
│                                               │
└───────────────────┬───────────────────────────┘
                    │
                    ↓ buffer (32 steps data)
                    │
┌─────────────────── TRAINING ──────────────────┐
│                                               │
│  for epoch in range(5):                      │
│    for minibatch in buffer:                  │
│      loss = compute_loss(policy, mb) ←──┐    │
│      loss.backward()                    │训练模式│
│      optimizer.step()                ←──┘计算梯度│
│                                          更新参数│
└───────────────────────────────────────────────┘
```

---

**关键总结**：

1. **Rollout = 数据收集**：与环境交互，不更新参数
2. **Training = 参数优化**：使用历史数据，更新参数
3. **交替执行**：Rollout → Training → Rollout → Training ...
4. **同一网络**：Training 更新的参数会被下次 Rollout 使用
5. **时间分配**：Rollout 占大部分时间（环境瓶颈）

---

**文档生成时间**: 2025-01-XX
**参考代码**: `scripts/train.py`, `active_adaptation/learning/ppo/ppo_roa.py`