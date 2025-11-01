# HDMI Action Manager中的`offset`详细分析

## 🎯 核心结论

**`self.offset`的作用**: **Domain Randomization - 随机关节位置偏移**

用于在训练时为每个环境的default joint position添加随机噪声,提高策略的鲁棒性,帮助sim-to-real迁移。

---

## 📊 完整数据流

### 1. 初始化 (全零)

**位置**: `active_adaptation/envs/mdp/action.py:68`

```python
class JointPosition(ActionManager):
    def __init__(self, env, action_scaling, ...):
        # ...
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)  # ← 初始化为全零
        # Shape: [num_envs, num_joints]
```

**初始状态**:
- `offset[env_i, joint_j]` = 0.0 (所有环境,所有关节)
- 如果没有domain randomization,`offset`始终为0

---

### 2. 随机化设置 (Domain Randomization)

**位置**: `active_adaptation/envs/mdp/randomizations.py:731-744`

```python
class random_joint_offset(Randomization):
    """
    在每次环境reset时,为关节的default position添加随机偏移
    """
    def __init__(self, env, **offset_range: Tuple[float, float]):
        super().__init__(env)
        self.asset = self.env.scene["robot"]

        # 解析配置中的offset范围
        # 例如: {".*": [-0.01, 0.01]} 表示所有关节偏移范围为±0.01 rad
        self.joint_ids, _, self.offset_range = \
            string_utils.resolve_matching_names_values(
                dict(offset_range),
                self.asset.joint_names
            )

        # offset_range: [num_envs, num_joints, 2]
        # offset_range[env_i, joint_j, :] = [min_offset, max_offset]
        self.offset_range = torch.tensor(self.offset_range, device=self.device) \
                                 .unsqueeze(0).expand(self.num_envs, -1, -1)

        self.action_manager = self.env.action_manager

    def reset(self, env_ids: torch.Tensor):
        """每次环境reset时调用,为指定的环境生成新的随机offset"""
        # 从均匀分布中采样
        offset = uniform(
            self.offset_range[env_ids, :, 0],  # min值
            self.offset_range[env_ids, :, 1]   # max值
        )
        # Shape: [len(env_ids), num_joints]

        # 将随机offset写入action_manager
        self.action_manager.offset[env_ids.unsqueeze(1), self.joint_ids] = offset
```

**配置示例** (`cfg/task/G1/twist/0927_twist_teacher_new.yaml:299-300`):
```yaml
randomization:
  random_joint_offset:
    .*: [-0.01, 0.01]  # 所有关节偏移范围为±0.01 rad (约±0.57度)
```

**具体例子**:
```python
# 环境0 reset时:
env_ids = torch.tensor([0])
offset = uniform([-0.01, -0.01, ...], [0.01, 0.01, ...])
# 可能得到: offset = [0.003, -0.007, 0.005, ...]

# 环境1024 reset时:
env_ids = torch.tensor([1024])
offset = uniform([-0.01, -0.01, ...], [0.01, 0.01, ...])
# 可能得到: offset = [-0.004, 0.009, -0.002, ...]
```

**关键点**:
- ✅ **每个环境独立采样**: 不同环境有不同的offset
- ✅ **每次reset重新采样**: 同一环境每次重置都会得到新的offset
- ✅ **在整个episode期间保持不变**: reset后,offset固定,直到下次reset

---

### 3. 应用offset到目标关节位置

**位置**: `active_adaptation/envs/mdp/action.py:117-119`

```python
def __call__(self, action: torch.Tensor, substep: int):
    # ... (delay和平滑处理)

    # ← 关键步骤: 计算目标关节位置
    pos_target = self.default_joint_pos + self.offset
    pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
    self.asset.set_joint_position_target(pos_target)
```

**数学公式**:
```
pos_target[env_i, joint_j] = default_joint_pos[env_i, joint_j]  # 默认姿态
                            + offset[env_i, joint_j]            # 随机偏移
                            + action[i, joint_j] * scaling[j]   # policy输出
```

**具体例子** (某个环境的膝关节):
- `default_joint_pos[knee]` = 0.0 rad (站立姿态)
- `offset[knee]` = 0.005 rad (本episode的随机偏移)
- `action[knee]` = 1.2 (policy输出)
- `action_scaling[knee]` = 0.5

计算:
```
pos_target[knee] = 0.0 + 0.005 + (1.2 × 0.5)
                 = 0.005 + 0.6
                 = 0.605 rad
```

**如果没有offset** (offset=0):
```
pos_target[knee] = 0.0 + 0.0 + (1.2 × 0.5) = 0.6 rad
```

**offset的影响**:
- ✅ 改变了"默认姿态"的基准点
- ✅ policy的action仍然是相对于(default + offset)的偏移
- ✅ 增加训练时的姿态多样性

---

### 4. 在Observation中补偿offset

**位置**: `active_adaptation/envs/mdp/observations/common.py:136,168`

```python
class joint_pos_history(Observation):
    def __init__(self, env, joint_names=".*", history_steps=[1], ...):
        super().__init__(env)
        from active_adaptation.envs.mdp.action import JointPosition
        action_manager: JointPosition = self.env.action_manager

        # 获取offset的引用 (共享同一个tensor)
        self.joint_pos_offset = action_manager.offset  # ← 引用,不是复制
        # ...

    def compute(self):
        # 从buffer中获取历史关节位置
        # buffer存储的是真实的关节位置 (包含了offset的影响)

        # 减去offset,得到相对于default_joint_pos的偏移
        joint_pos = self.buffer - self.joint_pos_offset[:, self.joint_ids].unsqueeze(1)
        #           ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #           真实位置       减去随机offset

        joint_pos = joint_pos * self.joint_mask
        joint_pos_selected = joint_pos[:, self.history_steps]
        return joint_pos_selected.reshape(self.num_envs, -1)
```

**为什么要减去offset?**

因为我们希望observation是**相对于default pose的偏移**,而不是绝对位置。

**数学分析**:

假设某个关节:
- `default_joint_pos` = 0.0
- `offset` = 0.005 (本episode的随机偏移)
- `action` = 1.2
- `action_scaling` = 0.5

实际应用后:
```
actual_joint_pos = 0.0 + 0.005 + (1.2 × 0.5) = 0.605 rad
```

在observation中:
```
observed_joint_pos = actual_joint_pos - offset
                   = 0.605 - 0.005
                   = 0.6 rad
```

这样,observation中看到的是`0.6 rad`,与`action * scaling = 1.2 × 0.5 = 0.6`一致!

**关键洞察**:
- ✅ **物理世界**: 关节实际位置受offset影响
- ✅ **Observation空间**: 减去offset,恢复到"无偏移"的参考系
- ✅ **Policy视角**: policy看到的是相对于default pose的偏移,与offset无关

**这样做的好处**:
1. ✅ **训练时增加多样性**: 物理世界中机器人姿态有offset噪声
2. ✅ **Policy不知道offset**: observation中已补偿,policy学习的是"相对关节位置"
3. ✅ **Sim-to-Real**: 真实硬件的零位可能有偏差,通过offset模拟这种偏差

---

## 🔍 与TWIST对比

### TWIST是否使用offset?

让我检查TWIST的配置:

**TWIST配置** (`cfg/task/base/twist-base.yaml:190`):
```yaml
randomization:
  random_joint_offset: {.*: [-0.01, 0.01]}
```

✅ **TWIST也使用相同的random_joint_offset!**

**对比**:

| 维度 | TWIST | HDMI | 一致性 |
|------|-------|------|--------|
| **使用offset** | ✅ 是 | ✅ 是 | ✅ 一致 |
| **offset范围** | ±0.01 rad | ±0.01 rad | ✅ 一致 |
| **应用方式** | `target = default + offset + action * scale` | `target = default + offset + action * scale` | ✅ 一致 |
| **Observation补偿** | (需要检查TWIST代码) | ✅ 在`joint_pos_history`中减去offset | - |

---

## 💡 Domain Randomization的作用

### 为什么需要offset?

**问题**: 真实硬件的关节零位可能与仿真不一致

**现象**:
- 仿真中: `joint_pos=0.0` 表示站立姿态
- 真实硬件: 由于装配误差、标定误差,`joint_pos=0.0` 可能对应略微不同的姿态

**解决方案**: 在训练时添加随机offset

```python
# 仿真训练时:
# 环境A: offset = +0.005 rad → 机器人"以为"自己在0.0,实际在0.005
# 环境B: offset = -0.003 rad → 机器人"以为"自己在0.0,实际在-0.003
# 环境C: offset = +0.008 rad → 机器人"以为"自己在0.0,实际在0.008
```

**训练效果**:
- ✅ Policy学会对零位偏差鲁棒
- ✅ 部署到真实硬件时,即使零位有偏差,policy仍能工作
- ✅ 减少sim-to-real gap

---

## 📊 完整示例

### 场景: 训练一个环境

**初始化**:
```python
# 环境创建
env = create_env()
action_manager = JointPosition(env, action_scaling=0.5)

# offset初始化为0
action_manager.offset = torch.zeros(num_envs, num_joints)
```

**第1次Reset** (env_id=0):
```python
# randomization.reset([0]) 被调用
offset = uniform([-0.01], [0.01])  # 假设采样到 0.005
action_manager.offset[0, :] = 0.005  # 所有关节偏移0.005 rad
```

**Episode 1** (offset=0.005):
```python
# Step 1:
action = policy(obs)  # 假设输出 action[knee] = 1.0
pos_target[knee] = 0.0 + 0.005 + (1.0 × 0.5) = 0.505 rad

# Step 2:
action = policy(obs)  # 假设输出 action[knee] = -0.5
pos_target[knee] = 0.0 + 0.005 + (-0.5 × 0.5) = -0.245 rad

# ... episode continues ...
```

**第2次Reset** (env_id=0):
```python
# randomization.reset([0]) 再次调用
offset = uniform([-0.01], [0.01])  # 假设采样到 -0.003
action_manager.offset[0, :] = -0.003  # 新的offset
```

**Episode 2** (offset=-0.003):
```python
# Step 1:
action = policy(obs)  # 假设输出相同 action[knee] = 1.0
pos_target[knee] = 0.0 + (-0.003) + (1.0 × 0.5) = 0.497 rad
#                                   ^^^^^^^^ 与Episode 1不同!

# 物理世界中,机器人姿态略有不同
# 但observation中,由于减去了offset,policy看到的是相同的"相对位置"
```

---

## 🎓 总结

### `offset`的本质

| 属性 | 值 |
|------|-----|
| **作用** | Domain Randomization - 模拟关节零位偏差 |
| **初始值** | 全零 (如果没有randomization) |
| **设置时机** | 每次环境reset时,由`random_joint_offset`随机采样 |
| **取值范围** | 通常 ±0.01 rad (约±0.57度) |
| **持续时间** | 整个episode (reset前保持不变) |
| **影响** | 改变default joint position的基准点 |
| **Observation补偿** | 在`joint_pos_history`中减去offset,policy不感知 |

### 数学总结

**目标关节位置计算**:
```
pos_target = default_joint_pos + offset + (action × action_scaling)
             ^^^^^^^^^^^^^^^^   ^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
             默认姿态(站立)      随机噪声   policy输出的偏移
```

**Observation计算**:
```
observed_joint_pos = actual_joint_pos - offset
                     ^^^^^^^^^^^^^^^^   ^^^^^^
                     包含offset的真实位置  减去offset,恢复到无偏移参考系
```

**Policy视角**:
```
policy输入: observed_joint_pos (已补偿offset)
policy输出: action (相对于default_joint_pos的偏移)
```

### 关键要点

1. ✅ **offset不是bug,是feature**: 故意引入的随机噪声,用于提高鲁棒性
2. ✅ **与TWIST一致**: TWIST也使用相同的offset机制和范围
3. ✅ **Policy不感知offset**: observation中已补偿,policy学习的是"相对位置"
4. ✅ **物理世界受影响**: 实际关节位置包含offset,增加训练多样性
5. ✅ **Sim-to-Real**: 模拟真实硬件的零位标定误差

### 配置建议

**训练阶段** (提高鲁棒性):
```yaml
randomization:
  random_joint_offset:
    .*: [-0.01, 0.01]  # 较大范围,增加多样性
```

**微调阶段** (如果需要更精确):
```yaml
randomization:
  random_joint_offset:
    .*: [-0.005, 0.005]  # 较小范围
```

**测试阶段** (评估性能):
```yaml
randomization:
  random_joint_offset:
    .*: [0.0, 0.0]  # 无offset,纯净评估
```

---

## 📚 代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| **offset初始化** | `active_adaptation/envs/mdp/action.py` | 68 |
| **offset应用** | `active_adaptation/envs/mdp/action.py` | 117 |
| **offset随机化** | `active_adaptation/envs/mdp/randomizations.py` | 731-744 |
| **offset补偿** | `active_adaptation/envs/mdp/observations/common.py` | 168 |
| **配置示例** | `cfg/task/G1/twist/0927_twist_teacher_new.yaml` | 299-300 |

---

## 🔍 常见误解

❌ **"offset是bug,应该去掉"**
- ✅ offset是domain randomization,提高sim-to-real迁移能力

❌ **"offset会影响policy学习"**
- ✅ observation中已补偿offset,policy看到的是相对位置

❌ **"offset应该在整个训练过程保持不变"**
- ✅ offset在每次reset时重新采样,每个episode不同

❌ **"TWIST没有offset,HDMI多此一举"**
- ✅ TWIST也使用相同的offset机制

---

**最后更新**: 2025-11-01
