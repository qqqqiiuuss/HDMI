# TWIST-MASTER vs HDMI ppo_twist 完整训练流程对比

## 执行摘要

**结论**: ⚠️ **不完全一致，存在几个关键差异**

虽然PPO超参数已对齐，但以下方面存在差异：
1. ❌ **观察空间归一化** - TWIST有，HDMI没有
2. ❌ **Action std调度** - TWIST有动态调度，HDMI是固定的
3. ❌ **Motion curriculum** - TWIST有课程学习，HDMI没有
4. ✅ **PPO算法核心** - 完全一致
5. ✅ **奖励计算** - 一致（不裁剪负奖励）
6. ✅ **数据收集** - 一致（24步rollout）

---

## 详细对比

### 1. 训练循环主流程

| 组件 | TWIST-MASTER | HDMI ppo_twist | 状态 |
|------|--------------|----------------|------|
| 主循环 | `on_policy_runner_mimic.py` | `scripts/train.py` | ✅ 等价 |
| Rollout步数 | `num_steps_per_env=24` | `train_every=24` | ✅ 一致 |
| 更新频率 | 每24步更新一次 | 每24步更新一次 | ✅ 一致 |
| Episode长度 | `episode_length_s=10` (500 steps) | `max_episode_length=500` | ✅ 一致 |

**代码对比**:

```python
# TWIST (on_policy_runner_mimic.py)
for it in range(current_learning_iteration, self.max_iterations):
    # Rollout
    for _ in range(self.num_steps_per_env):  # 24 steps
        actions = self.alg.act(obs, critic_obs)
        obs, rewards, dones, infos = self.env.step(actions)
        self.alg.process_env_step(rewards, dones, infos)

    # Update
    self.alg.compute_returns(critic_obs)
    self.alg.update()

# HDMI (train.py + ppo_twist)
for i in range(total_iters):
    # Rollout
    for step in range(cfg.algo.train_every):  # 24 steps
        carry = rollout_policy(carry)
        td, carry = env.step_and_maybe_reset(carry)
        data_buf[:, step] = td

    # Update
    policy.train_op(data_buf)
```

---

### 2. 🔴 **观察空间归一化** (关键差异)

#### TWIST-MASTER

```python
# g1_mimic_distill_config.py:40
normalize_obs = True

# humanoid_mimic.py
class HumanoidMimicEnv:
    def __init__(self):
        if self.cfg.env.normalize_obs:
            self.obs_rms = RMS(device=self.device, shape=[self.cfg.env.num_observations])

    def step(self):
        obs = self._compute_observations()
        if self.cfg.env.normalize_obs:
            mean, var = self.obs_rms(obs)
            obs = (obs - mean) / torch.sqrt(var + 1e-8)
        return obs
```

**作用**:
- 在线计算观察空间的running mean和running std
- 归一化到均值0、方差1
- 稳定训练，加快收敛

#### HDMI ppo_twist

```python
# ❌ 没有观察空间归一化!
# env.step() 直接返回原始观察
```

**影响**:
- 观察空间的尺度未归一化
- 可能影响网络训练稳定性
- 不同维度的重要性未平衡

---

### 3. 🔴 **Action Std调度** (关键差异)

#### TWIST-MASTER

```python
# g1_mimic_distill_config.py:444
std_schedule = [1.0, 0.4, 4000, 1500]
# 含义: [init_std, final_std, warmup_iters, decay_iters]

# ppo.py:231-233
if self.fix_std:
    std_stage = min(max((self.counter - std_schedule[2]), 0) / std_schedule[3], 1)
    std_coef = std_stage * (std_schedule[1] - std_schedule[0]) + std_schedule[0]
    self.actor_critic.update_std(std_coef)
```

**调度曲线**:
```
Iteration:    0     4000    5500    (total_iters)
Action std: 1.0 →  1.0  →  0.4
            \_____/\__________/
            warmup   decay
```

**作用**:
- 初期(0-4000 iters): std=1.0，高探索
- 衰减期(4000-5500): std=1.0→0.4，逐渐降低探索
- 后期(>5500): std=0.4，exploitation

#### HDMI ppo_twist

```python
# ppo_twist.py
init_noise_scale: float = 1.0  # 固定初始化
# ❌ 没有std_schedule，训练过程中std不变
```

**影响**:
- 探索-利用平衡可能不optimal
- TWIST通过降低std来逐渐转向exploitation
- HDMI可能在后期探索过多或过少

---

### 4. 🔴 **Motion Curriculum Learning** (差异)

#### TWIST-MASTER

```python
# g1_mimic_distill_config.py
motion_curriculum = True
motion_curriculum_gamma = 0.01

# humanoid_mimic.py
if self.cfg.env.motion_curriculum:
    # 初期重置到motion开始附近，逐渐扩展
    self.motion_start_range *= (1 + motion_curriculum_gamma)
```

**作用**:
- 初期从motion开始位置附近重置
- 逐渐扩展重置范围到整个motion
- 课程学习：从简单到困难

#### HDMI

```python
# ❌ 没有motion curriculum
# reset总是在motion的随机位置
```

**影响**:
- 初期可能从困难的motion phase开始
- 学习效率可能降低

---

### 5. ✅ **PPO算法核心** (完全一致)

| 参数 | TWIST | ppo_twist | 状态 |
|------|-------|-----------|------|
| `train_every` | 24 | 24 | ✅ |
| `ppo_epochs` | 5 | 5 | ✅ |
| `num_minibatches` | 4 | 4 | ✅ |
| `lr` | 2e-4 | 2e-4 | ✅ |
| `clip_param` | 0.2 | 0.2 | ✅ |
| `entropy_coef` | 0.01 | 0.01 | ✅ |
| `desired_kl` | 0.008 | 0.008 | ✅ |
| `gamma` | 0.99 | 0.99 | ✅ |
| `lmbda` | 0.95 | 0.95 | ✅ |
| `max_grad_norm` | 1.0 | 1.0 | ✅ |

**PPO更新逻辑**:

```python
# TWIST ppo.py:197-202
ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
surrogate = -advantages_batch * ratio
surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

# HDMI ppo_twist.py:368-372
ratio = torch.exp(log_ratio)
surr1 = adv * ratio
surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
policy_loss = - (torch.min(surr1, surr2) * valid).mean()
```

✅ **逻辑完全一致** (只是写法不同，数学等价)

---

### 6. ✅ **奖励处理** (一致)

#### TWIST-MASTER

```python
# ppo.py:142-147 (process_env_step)
self.transition.rewards = rewards.clone()  # 不裁剪
# ❌ 没有 clamp_min(0.)

# storage.py:compute_returns
# 使用原始rewards计算GAE
```

#### HDMI ppo_twist

```python
# ppo_twist.py:331-335
if self.cfg.clamp_negative_rewards:
    rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True).clamp_min(0.)
else:
    rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True)  # ✅ 默认不裁剪

# 配置
clamp_negative_rewards: bool = False  # ✅ 与TWIST一致
```

✅ **都不裁剪负奖励**

---

### 7. ✅ **GAE计算** (一致)

#### TWIST-MASTER

```python
# storage.py:compute_returns
def compute_returns(self, last_values, gamma, lam):
    advantage = 0
    for step in reversed(range(self.num_transitions_per_env)):
        delta = self.rewards[step] + gamma * next_values * (1 - self.dones[step]) - self.values[step]
        advantage = delta + gamma * lam * (1 - self.dones[step]) * advantage
        self.advantages[step] = advantage
        self.returns[step] = advantage + self.values[step]
```

#### HDMI ppo_twist

```python
# common.py:GAE class
class GAE:
    def __init__(self, gamma, lmbda):
        self.gamma = gamma
        self.lmbda = lmbda

    def __call__(self, rewards, terms, dones, values, next_values, discount):
        # 标准GAE计算
```

✅ **算法完全一致** (gamma=0.99, lambda=0.95)

---

### 8. ✅ **Value Function Loss** (一致)

#### TWIST-MASTER

```python
# ppo.py:205-212
if self.use_clipped_value_loss:
    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-clip_param, clip_param)
    value_losses = (value_batch - returns_batch).pow(2)
    value_losses_clipped = (value_clipped - returns_batch).pow(2)
    value_loss = torch.max(value_losses, value_losses_clipped).mean()
```

#### HDMI ppo_twist

```python
# ppo_twist.py:383-385
value_loss = self.critic_loss_fn(b_returns, values)  # MSE
value_loss = (value_loss * valid).mean()
```

⚠️ **差异**: TWIST使用clipped value loss，HDMI使用标准MSE

**影响**: 理论上clipped value loss更稳定，但实践中差异可能不大

---

## 关键差异总结

### ❌ 差异1: 观察空间归一化

**TWIST**:
```python
normalize_obs = True  # Running mean/std normalization
```

**HDMI**:
```python
# 没有归一化
```

**建议**:
```python
# 在HDMI中添加VecNorm transform
from torchrl.envs.transforms import VecNorm
vecnorm = VecNorm(in_keys=[OBS_KEY])
```

---

### ❌ 差异2: Action Std调度

**TWIST**:
```python
std_schedule = [1.0, 0.4, 4000, 1500]  # 动态衰减
```

**HDMI**:
```python
init_noise_scale = 1.0  # 固定
```

**建议**: 在`ppo_twist.py`中添加std调度逻辑

---

### ❌ 差异3: Motion Curriculum

**TWIST**:
```python
motion_curriculum = True
```

**HDMI**:
```python
# 没有课程学习
```

**建议**: 在环境的reset()中实现类似逻辑

---

## 总体评估

### 相同之处 (✅)
1. **PPO核心算法** - 完全一致
2. **训练超参数** - 完全对齐
3. **网络架构** - 一致 ([512, 512, 256, 128])
4. **数据收集** - 一致 (24步rollout)
5. **GAE计算** - 一致 (γ=0.99, λ=0.95)
6. **奖励处理** - 一致 (不裁剪负值)
7. **学习率调度** - 一致 (adaptive KL)

### 差异之处 (❌)
1. **观察空间归一化** - TWIST有，HDMI没有
2. **Action std调度** - TWIST动态，HDMI固定
3. **Motion curriculum** - TWIST有，HDMI没有
4. **Value loss** - TWIST clipped，HDMI standard MSE

### 影响评估

**关键程度排序**:
1. **观察空间归一化** (⚠️ 中等) - 可能影响训练稳定性和收敛速度
2. **Action std调度** (⚠️ 中等) - 影响探索-利用平衡
3. **Motion curriculum** (⚠️ 低) - 影响初期学习效率
4. **Value loss** (⚠️ 低) - 实践影响通常不大

### 预期效果

使用当前的`ppo_twist`:
- ✅ **应该能工作** - PPO核心算法一致
- ⚠️ **可能不如TWIST** - 缺少归一化和调度
- ✅ **比ppo_motion_encoder好** - 没有信息损失

### 改进建议优先级

**Priority 1 (立即实现)**:
```python
# 添加观察空间归一化
vecnorm = VecNorm(in_keys=[OBS_KEY], stats_keys=["mean", "std"])
```

**Priority 2 (建议实现)**:
```python
# 添加action std调度
std_schedule = [1.0, 0.4, 4000, 1500]
```

**Priority 3 (可选)**:
```python
# Motion curriculum (需要修改环境)
motion_curriculum = True
```

---

## 使用建议

### 方案1: 直接使用ppo_twist (推荐尝试)

```bash
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096
```

**预期**: 效果应该接近TWIST，但可能略差（因为缺少归一化）

### 方案2: 添加VecNorm后使用 (最佳)

```bash
# 修改train.py添加VecNorm
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    vecnorm=train  # 启用归一化
```

**预期**: 效果应该非常接近TWIST

### 方案3: 完全对齐TWIST (最完整)

实现所有差异点后训练。

**预期**: 效果应该与TWIST-MASTER一致

---

## 结论

**回答你的问题**:

> "是不是除了我的obs改了以及motionencoder没有，其他整体的训练流程完全一致？"

**答案**: ❌ **不完全一致**

虽然**PPO算法核心是一致的**，但存在以下关键差异：

1. ❌ **观察空间归一化** - 这可能是最重要的差异
2. ❌ **Action std调度** - 影响探索策略
3. ❌ **Motion curriculum** - 影响初期学习

**但是**:
- ✅ PPO超参数完全对齐
- ✅ 网络架构一致
- ✅ 没有Motion Encoder的信息损失
- ✅ 奖励计算一致

**建议**: 先用当前的`ppo_twist`训练，然后逐步添加归一化和调度来进一步改进。
