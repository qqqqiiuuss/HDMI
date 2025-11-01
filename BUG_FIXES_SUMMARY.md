# Bug 修复总结 - 2025-11-01

## 1. ✅ `_update_action_std` 方法缩进错误

### 问题
```
AttributeError: 'PPOTwistPolicy' object has no attribute '_update_action_std'
```

### 根本原因
`_update_action_std()` 方法被错误地缩进在全局函数 `get_activation()` 的 `else` 块内部，而不是 `PPOTwistPolicy` 类的方法。

### 修复
**文件**: `active_adaptation/learning/ppo/ppo_twist.py`

**修改**:
- 删除了第486-522行错误位置的 `_update_action_std()` 定义
- 在第419行（`load_state_dict()` 方法后）正确添加了 `_update_action_std()` 作为类方法

**修复后位置**: `ppo_twist.py:419-466`

---

## 2. ✅ Actor std 访问路径错误

### 问题
```
AttributeError: 'SafeProbabilisticModule' object has no attribute 'module'
```

### 根本原因
原代码假设 `self.actor.module[1].module.actor_std` 可以访问 `Actor` 实例，但实际上 `ProbabilisticActor` 的嵌套结构不同。

### Actor 结构
```python
self.actor = ProbabilisticActor(
    module = TensorDictSequential([
        TensorDictModule(actor_backbone, ...),  # [0]
        TensorDictModule(Actor(...), ...)       # [1] <- Actor 在这里
    ])
)
```

### 修复
**文件**: `active_adaptation/learning/ppo/ppo_twist.py`

**两处修改**:

1. **`_update_action_std()` 方法** (第454-466行):
```python
# 修复前（错误）
actor_module = self.actor.module[1].module  # ❌ SafeProbabilisticModule 无 .module
actor_module.actor_std.fill_(target_std)

# 修复后（正确）
try:
    actor_module = self.actor[1]             # TensorDictModule wrapping Actor
    actor_instance = actor_module.module     # Actor class instance
    with torch.no_grad():
        actor_instance.actor_std.fill_(target_std)
except (AttributeError, IndexError, TypeError):
    # Fallback: 通过参数名搜索
    for name, param in self.actor.named_parameters():
        if 'actor_std' in name:
            with torch.no_grad():
                param.fill_(target_std)
            break
```

2. **WandB 日志记录** (第288-298行):
```python
# 修复前（错误）
infos["actor/action_std"] = self.actor.module[1].module.std.mean().item()  # ❌

# 修复后（正确）
try:
    actor_module = self.actor[1]
    actor_instance = actor_module.module
    infos["actor/action_std"] = actor_instance.actor_std.mean().item()
except (AttributeError, IndexError, TypeError):
    for name, param in self.actor.named_parameters():
        if 'actor_std' in name:
            infos["actor/action_std"] = param.mean().item()
            break
```

**注意**: 原代码还有个错误：`actor_module.std` 应该是 `actor_module.actor_std`（见 `common.py:153`）

---

## 3. ℹ️ Reward 权重显示 0.00 **不是 Bug**

### 问题描述
用户看到以下输出：
```
Reward group: regularization
    feet_contact_forces_twist:     -0.00,     True
    dof_vel_twist:     -0.00,     True
    dof_acc_twist:     -0.00,     True
```

### 原因分析

**这不是 Bug！** 是**显示精度问题**。

这些奖励项的权重非常小，在配置文件中定义为：

```yaml
# cfg/task/G1/twist/0927_twist_teacher_new.yaml

# feet_contact_forces = -5e-4 (TWIST line 200)
feet_contact_forces_twist:
  weight: -0.0005         # -5e-4

# dof_vel = -1e-4 (TWIST line 207)
dof_vel_twist:
  weight: -0.0001         # -1e-4

# dof_acc = -5e-8 (TWIST line 208)
dof_acc_twist:
  weight: -0.00000005     # -5e-8

# ankle_dof_acc = -1e-7 (TWIST line 225)
ankle_dof_acc_twist:
  weight: -0.0000001      # -1e-7

# ankle_dof_vel = -2e-4 (TWIST line 226)
ankle_dof_vel_twist:
  weight: -0.0002         # -2e-4
```

**打印时使用了有限精度格式化**（如 `{:.2f}`），导致：
- `-0.0005` → 显示为 `-0.00`
- `-0.0001` → 显示为 `-0.00`
- `-5e-8` → 显示为 `-0.00`

### 如何验证权重是否生效

**方法1**: 检查 WandB 日志中的详细奖励值

查看以下指标：
```
reward/feet_contact_forces_twist
reward/dof_vel_twist
reward/dof_acc_twist
```

如果有数值（即使很小），说明权重生效。

**方法2**: 修改打印格式

如果想看到完整权重，可以在打印奖励权重的地方使用科学计数法：
```python
# 修改打印格式为科学计数法
print(f"{name}: {weight:.2e}, {enabled}")  # 使用 .2e 代替 .2f
```

这样会显示：
```
feet_contact_forces_twist:     -5.00e-04,     True
dof_vel_twist:     -1.00e-04,     True
dof_acc_twist:     -5.00e-08,     True
```

### TWIST-MASTER 原版也是这样

这些微小权重**完全对齐 TWIST-MASTER 原版配置**：

```python
# TWIST-master/legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py

class G1MimicDistillCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            dof_vel = -1e-4              # ✅ 与 HDMI 一致
            dof_acc = -5e-8              # ✅ 与 HDMI 一致
            feet_contact_forces = -5e-4  # ✅ 与 HDMI 一致
```

**结论**: 这些奖励项的作用是**轻微正则化**，防止过大的关节速度、加速度和接触力，但不应过度惩罚。权重小是**正常的**。

---

## 4. 之前修复的 Bug (回顾)

### Bug 4.1: future_steps dtype 错误
**文件**: `active_adaptation/envs/mdp/commands/twist/command.py:142`

```python
# 修复前
self.future_steps = torch.tensor(future_steps)  # 可能推断为float

# 修复后
self.future_steps = torch.tensor(future_steps, dtype=torch.long)  # 明确long
```

### Bug 4.2: future_steps 数量错误
**文件**: `cfg/task/G1/twist/0927_twist_teacher_new.yaml`

```yaml
# 修复前（只有9个）
future_steps: [1, 2,3,4,5,6,7,8,9]

# 修复后（10个）
future_steps: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

---

## 测试验证

### 1. 语法检查
```bash
python -m py_compile active_adaptation/learning/ppo/ppo_twist.py
# ✅ Syntax check passed
```

### 2. 重新训练测试
```bash
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096 \
    suffix=test_1014_twist_fixed
```

**预期**:
1. 训练应该正常启动，不再有 `AttributeError`
2. WandB 应该记录 `actor/action_std` 指标
3. Std schedule 应该正确执行：
   - Iteration 0-4000: std=1.0
   - Iteration 4000-5500: std=1.0→0.4
   - Iteration >5500: std=0.4

### 3. 监控 Std Schedule
在 WandB 查找指标 `actor/action_std`，应该看到类似曲线：

```
Iteration:     0     4000   5500   10000
Action Std:  1.0 →  1.0  →  0.4  →  0.4
             |-----|-------|--------|
             warmup  decay   stable
```

---

## 文件修改清单

1. ✅ `active_adaptation/learning/ppo/ppo_twist.py`
   - 修复 `_update_action_std()` 缩进（第419行）
   - 修复 std 访问路径（第454-466行，第288-298行）

2. ✅ `active_adaptation/envs/mdp/commands/twist/command.py`
   - 修复 `future_steps` dtype（第142行）
   - 添加 motion curriculum 实现

3. ✅ `cfg/task/G1/twist/0927_twist_teacher_new.yaml`
   - 修复 `future_steps` 数量（9→10）
   - 添加 motion curriculum 配置

---

## 下一步

1. **重新训练** ppo_twist 并监控 WandB
2. **验证 std schedule** 是否正确工作
3. **对比实验**:
   - ppo_twist (有 std schedule + curriculum)
   - ppo (无 std schedule)
4. **如果仍有问题**: 检查 `actor/action_std` 日志，确认 fallback 是否被触发

---

## 总结

**主要 Bug**:
1. ✅ 缩进错误 - `_update_action_std` 不是类方法
2. ✅ 访问路径错误 - `self.actor` 的嵌套结构理解错误

**不是 Bug**:
3. ℹ️ Reward 权重显示 0.00 - 正常，权重本身就是 1e-4, 5e-8 级别

**所有修复已完成，可以重新训练测试！**
