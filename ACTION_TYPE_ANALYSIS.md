# HDMI Action类型分析: Residual vs Absolute

## 🎯 核心结论

**HDMI-todesk输出的action是: 基于`default_joint_pos`的相对偏移量 (Residual Action)**

但这里的"Residual"**不是**基于上一帧实际joint position的增量,而是相对于default pose的偏移。

---

## 📊 完整数据流分析

### 1. Policy网络输出 (Actor)

**位置**: `active_adaptation/learning/ppo/common.py:145-164`

```python
class Actor(nn.Module):
    def __init__(self, action_dim: int, init_noise_scale: float=1.0,
                 predict_std: bool=False, load_noise_scale: float | None=None):
        super().__init__()
        if predict_std:
            self.actor_mean = nn.LazyLinear(action_dim * 2)  # 预测均值和方差
        else:
            self.actor_mean = nn.LazyLinear(action_dim)      # 只预测均值
            self.actor_std = nn.Parameter(torch.ones(action_dim) * init_noise_scale)

    def forward(self, features: torch.Tensor):
        if self.predict_std:
            loc, scale = self.actor_mean(features).chunk(2, dim=-1)
        else:
            loc = self.actor_mean(features)  # ← 这是网络直接输出的action
            scale = torch.ones_like(loc) * self.actor_std
        return loc, scale  # loc就是action的均值
```

**关键点**:
- `loc` 是网络**直接输出**的action值
- **没有任何与default_joint_pos或previous action的关系**
- 这是一个**原始的、未经处理的action**

---

### 2. Action Manager接收action

**位置**: 你提供的代码 `JointPosition.__call__()`

```python
def __call__(self, action: torch.Tensor, substep: int):
    if substep == 0:
        if isinstance(action, TensorDictBase):
            action = action["action"]  # ← 从TensorDict中提取action
        self.action_buf[:, :, 1:] = self.action_buf[:, :, :-1]  # 历史action移位
        self.action_buf[:, :, 0] = action  # 存储当前action
```

**关键点**:
- `action` 是从policy网络采样得到的**原始action值**
- 存储在`action_buf`中用于延迟处理

---

### 3. Delay和平滑处理

```python
    # 计算延迟后的action索引
    action_dim = (self.delay - substep + self.env.decimation - 1) // self.env.decimation
    action = self.action_buf.take_along_dim(action_dim.unsqueeze(1), dim=-1)

    # 使用EMA平滑
    self.applied_action.lerp_(action.squeeze(-1), self.alpha)
```

**参数**:
- `delay`: 动作延迟步数 (模拟真实硬件延迟)
- `alpha`: EMA平滑系数 (在`[alpha_range[0], alpha_range[1]]`之间随机)

**示例** (假设delay=2, decimation=4, alpha=0.5):
- `substep=0`: 取`action_buf[:,:,1]` (上一次的action)
- `substep=1`: 取`action_buf[:,:,1]`
- `substep=2`: 取`action_buf[:,:,0]` (当前action)
- `substep=3`: 取`action_buf[:,:,0]`

然后用EMA平滑:
```python
applied_action = applied_action * (1 - alpha) + delayed_action * alpha
```

**关键点**:
- `applied_action` 是**平滑后**的action
- 仍然是**原始action值,未与任何joint position相关**

---

### 4. **关键步骤**: 转换为目标关节位置

```python
    # ← 这里是关键!
    pos_target = self.default_joint_pos + self.offset
    pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
    self.asset.set_joint_position_target(pos_target)
```

**数学表达**:
```
pos_target[joint_i] = default_joint_pos[joint_i] + offset[joint_i]
                      + applied_action[i] * action_scaling[i]
```

**具体例子** (假设某个关节):
- `default_joint_pos[knee_joint]` = 0.0 (站立姿态)
- `action_scaling[knee_joint]` = 0.5
- `applied_action[knee_joint]` = 1.2 (policy输出)
- `offset[knee_joint]` = 0.0 (通常为0)

计算:
```
pos_target[knee_joint] = 0.0 + 0.0 + 1.2 * 0.5 = 0.6 rad
```

**关键洞察**:
- ✅ **policy输出的action是相对于default_joint_pos的偏移量**
- ✅ `action_scaling`用于限制action的范围 (例如0.5意味着action=±1对应±0.5 rad)
- ❌ **NOT** 基于上一帧的实际joint position
- ❌ **NOT** 绝对joint position

---

## 🔍 与TWIST对比

### TWIST的action类型

查看TWIST的actor实现:
```python
# TWIST: rsl_rl/modules/actor_critic_mimic.py
class Actor(nn.Module):
    def forward(self, observations):
        # ...
        actions_mean = self.actor(observations)  # 直接输出action
        return actions_mean
```

在环境中应用:
```python
# TWIST环境
target_q = self._default_dof_pos + self.cfg.control.action_scale * actions
self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(target_q))
```

**数学表达**:
```
target_q = default_dof_pos + action_scale * action
```

**结论**: ✅ **TWIST和HDMI完全一致!都是基于default pose的residual action**

---

## 📋 Action类型总结

### Residual Action (相对于default pose) ← **HDMI使用这种**

```python
pos_target = default_joint_pos + action * scaling
```

**优点**:
1. ✅ **训练稳定**: action=0时机器人处于安全的default pose
2. ✅ **易于探索**: 初始随机action不会导致极端姿态
3. ✅ **物理意义**: action表示"相对于默认姿态的偏移"
4. ✅ **范围控制**: 通过`action_scaling`轻松限制action范围
5. ✅ **泛化能力**: 对不同任务,只需改变default pose

**缺点**:
- ❌ 无法直接达到远离default pose的姿态 (受限于scaling)
- ❌ 需要精心设计default pose

---

### Absolute Position Action (绝对位置) ← **HDMI不使用**

```python
pos_target = action  # action直接是目标关节角度
```

**优点**:
- ✅ 可以直接到达任何姿态
- ✅ 不依赖default pose

**缺点**:
- ❌ 训练不稳定: action=0时所有关节都在0度 (通常是危险姿态)
- ❌ 探索困难: 初始随机action可能导致极端姿态
- ❌ 需要额外归一化

---

### Residual Action (相对于上一帧) ← **HDMI也不使用**

```python
pos_target = current_joint_pos + action * scaling
```

**优点**:
- ✅ 平滑运动: 每次只改变一小步
- ✅ 适合速度控制

**缺点**:
- ❌ **误差累积**: 如果有测量噪声,误差会不断积累
- ❌ **不可恢复**: 一旦偏离目标姿态,难以自动纠正
- ❌ 依赖准确的状态估计

**为什么HDMI不用这种?**
- HDMI使用**位置控制** (PD controller),不是速度控制
- 基于default pose的residual更适合模仿学习 (motion tracking)

---

## 🧪 验证方法

### 实验1: 固定action=0

```python
# 在环境中测试
action = torch.zeros(num_envs, action_dim)
env.step(action)
# 预期: 机器人应该收敛到default_joint_pos
```

如果机器人收敛到default pose,说明是residual action (基于default)。

### 实验2: 固定action=1.0

```python
action = torch.ones(num_envs, action_dim) * 1.0
env.step(action)
# 预期: 机器人关节位置应该是 default_joint_pos + 1.0 * action_scaling
```

### 实验3: 检查action范围

```python
# 在训练过程中记录
actions_history = []
for step in range(1000):
    action = policy.get_action(obs)
    actions_history.append(action)

print(f"Action mean: {torch.stack(actions_history).mean()}")
print(f"Action std: {torch.stack(actions_history).std()}")
print(f"Action min: {torch.stack(actions_history).min()}")
print(f"Action max: {torch.stack(actions_history).max()}")
```

**预期结果**:
- Residual action (based on default): mean ≈ 0, std ≈ 0.5-2.0
- Absolute action: mean ≈ default_joint_pos值, std ≈ 0.1-0.5

---

## 💡 HDMI中的特殊处理

### 1. Action Scaling (关节级别不同scaling)

```python
action_scaling: float | Dict[str, float] = 0.5

# 可以为不同关节设置不同scaling:
action_scaling:
  hip_pitch: 0.5
  knee: 0.3
  ankle: 0.2
```

**作用**:
- 限制每个关节的最大偏移范围
- 例如`scaling=0.5`意味着action∈[-1,1]对应关节偏移∈[-0.5,0.5] rad

### 2. Action Delay (模拟真实延迟)

```python
min_delay: int = 0
max_delay: int = 0
```

**作用**:
- 模拟真实硬件的控制延迟 (传感器→计算→执行)
- 每个环境随机delay∈[min_delay, max_delay]
- 单位: 仿真步数 (不是decimation后的步数)

**示例** (delay=2, decimation=4):
- t=0: 生成action_0
- t=1: 等待
- t=2: 应用action_0 ← 延迟2步
- t=4: 生成action_1
- t=6: 应用action_1

### 3. Action Smoothing (EMA平滑)

```python
alpha: float | Tuple[float, float] = 0.5
```

**作用**:
- 平滑action变化,避免抖动
- `alpha=1.0`: 立即应用新action (无平滑)
- `alpha=0.1`: 缓慢过渡到新action (强平滑)

**数学**:
```python
applied_action_t = (1 - alpha) * applied_action_{t-1} + alpha * new_action_t
```

**示例**:
```
alpha=0.5:
t=0: applied=0.0, new=1.0 → applied=0.5
t=1: applied=0.5, new=1.0 → applied=0.75
t=2: applied=0.75, new=1.0 → applied=0.875
...
```

### 4. Offset (额外偏移)

```python
self.offset = torch.zeros_like(self.default_joint_pos)
```

**作用**:
- 允许在default pose基础上添加额外的静态偏移
- 通常为0,在某些任务中可能用于调整初始姿态

**最终公式**:
```python
pos_target = default_joint_pos + offset + applied_action * action_scaling
```

---

## 🎓 总结

### HDMI Action类型: **Residual Action (基于Default Pose)**

| 属性 | 值 |
|------|-----|
| **Policy输出** | 相对于default pose的偏移量 |
| **Action含义** | `Δjoint_pos` (相对偏移) |
| **目标计算** | `target = default + action * scaling` |
| **action=0时** | 机器人处于default pose |
| **action范围** | 通常 [-2, +2] (取决于初始化) |
| **实际关节范围** | `default ± (action * scaling)` |

### 与TWIST对比

| 维度 | TWIST | HDMI | 一致性 |
|------|-------|------|--------|
| **Action类型** | Residual (基于default) | Residual (基于default) | ✅ 一致 |
| **公式** | `target = default + action * scale` | `target = default + offset + action * scaling` | ✅ 一致 |
| **Scaling方式** | 统一scalar | 可per-joint配置 | ⚠️ HDMI更灵活 |
| **延迟模拟** | 无 | 支持随机delay | ⚠️ HDMI更真实 |
| **Action平滑** | 无 | 支持EMA平滑 | ⚠️ HDMI更稳定 |

### 关键要点

1. ✅ **Policy网络输出**: 纯粹的residual action,与default pose无关
2. ✅ **Action Manager转换**: 将residual action映射到目标关节位置
3. ✅ **物理意义**: action表示"期望偏离默认姿态多远"
4. ✅ **训练友好**: action=0是安全姿态,便于初期探索
5. ✅ **与TWIST一致**: 两者都使用基于default pose的residual action

### 常见误解 ❌

- ❌ "Action是绝对关节角度" - **错误!**是相对偏移
- ❌ "Action是基于上一帧的增量" - **错误!**是基于default pose
- ❌ "applied_action直接是关节位置" - **错误!**需要乘以scaling后加到default pose

### 代码关键行

```python
# 1. Policy输出原始action (相对偏移)
loc = self.actor_mean(features)  # common.py:161

# 2. 延迟和平滑处理
self.applied_action.lerp_(delayed_action, self.alpha)  # 你的代码

# 3. 转换为目标关节位置 (关键!)
pos_target = self.default_joint_pos + self.offset
pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling  # ← 这里!
self.asset.set_joint_position_target(pos_target)  # 你的代码
```

---

## 📚 参考

- HDMI代码: `active_adaptation/learning/ppo/common.py:145-164`
- Action Manager: 你提供的代码
- TWIST对比: `rsl_rl/modules/actor_critic_mimic.py`

**最后更新**: 2025-11-01
