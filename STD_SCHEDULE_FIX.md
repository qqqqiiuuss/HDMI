# Std Schedule Bug Fix - 修复记录

## 问题诊断

用户报告 `ppo_twist` 效果比 `ppo` 差很多，经过分析可能的原因：

### 最可能的原因：Std Schedule 实现 Bug ✅ **已修复**

**问题位置**: `ppo_twist.py:484`

**Bug 描述**:
```python
# 错误代码
actor_module = self.actor.module[1].module
with torch.no_grad():
    actor_module.std.fill_(target_std)  # ❌ Actor 类没有 'std' 属性
```

**正确实现** (已修复):
```python
# 修复后
actor_module = self.actor.module[1].module  # TensorDictModule wrapping Actor
with torch.no_grad():
    actor_module.actor_std.fill_(target_std)  # ✅ 正确访问 self.actor_std
```

**根据** `common.py:153`:
```python
class Actor(nn.Module):
    def __init__(self, action_dim: int, init_noise_scale: float=1.0, ...):
        # ...
        self.actor_std = nn.Parameter(torch.ones(action_dim) * init_noise_scale)
        #    ^^^^^^^^^ 正确的属性名
```

---

## 其他可能原因分析

### 2. Motion Curriculum 可能干扰训练 ⚠️

**现象**: 如果 HDMI 的 reset 流程不调用 `_update_motion_difficulty()`，curriculum 不会生效

**调试方法**:
```yaml
# 临时禁用 curriculum 测试
motion_curriculum: false
```

### 3. 网络更深需要更多迭代 ⏱️

**ppo_twist**: 4 层 MLP `[512, 512, 256, 128]`
**ppo**: 3 层 MLP `[512, 256, 128]`

**影响**: ppo_twist 可能需要更多 iterations 才能显示优势

### 4. 熵系数差异 🎲

**ppo_twist**: `entropy_coef = 0.01` (TWIST 默认)
**ppo**: `entropy_coef = 0.001` (HDMI 默认)

**影响**: ppo_twist 保持更高的随机性，初期可能学习慢

---

## 验证 Std Schedule 是否工作

### 方法 1: 检查 WandB Logs

在 WandB 界面搜索指标：
- `actor/action_std` - 应该显示调度曲线

**预期曲线**:
```
Iteration:     0     4000   5500
Action Std:  1.0 →  1.0  →  0.4
             |-----|-------|
             warmup  decay
```

### 方法 2: 运行验证脚本

使用下面创建的 `verify_std_schedule.py`

---

## 重新训练测试

### 测试 1: 完整 ppo_twist (推荐)

```bash
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096 \
    suffix=ppo_twist_fixed_std
```

**预期**: std schedule 现在应该正确工作

### 测试 2: 禁用 Std Schedule

如果仍有问题，测试是否是 std schedule 导致：

```bash
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096 \
    algo.use_std_schedule=false \
    suffix=ppo_twist_no_std_schedule
```

### 测试 3: 禁用 Motion Curriculum

```bash
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096 \
    task.motion_curriculum=false \
    suffix=ppo_twist_no_curriculum
```

### 测试 4: 对齐熵系数

```bash
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096 \
    algo.entropy_coef=0.001 \
    suffix=ppo_twist_low_entropy
```

---

## 对比基线

同时运行普通 ppo 作为对比：

```bash
python scripts/train.py \
    algo=ppo \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096 \
    suffix=ppo_baseline
```

---

## 预期效果差异

### ppo_twist vs ppo (理论分析)

| 维度 | ppo_twist | ppo | 预期影响 |
|------|-----------|-----|---------|
| **探索策略** | std 1.0→0.4 (动态) | std 1.5 (固定) | ppo_twist 后期更focused |
| **网络深度** | 4 层 | 3 层 | ppo_twist 表达能力更强，但收敛慢 |
| **熵系数** | 0.01 | 0.001 | ppo_twist 初期更random |
| **激活函数** | SiLU | ELU | 影响较小 |
| **课程学习** | 有 | 无 | ppo_twist 应该更稳定 |

**总体预期**:
- 初期 (0-4000 iters): ppo_twist 可能稍慢（高熵+深网络）
- 中期 (4000-5500): ppo_twist 应该开始超越（std 开始下降）
- 后期 (>5500): ppo_twist 应该明显更好（std=0.4，精确exploitation）

---

## 修复文件总结

1. ✅ **ppo_twist.py:484** - 修复 std update 的属性访问错误
2. ✅ **command.py** - 添加 motion curriculum 实现
3. ✅ **0927_twist_teacher_new.yaml** - 添加 curriculum 配置，修复 future_steps

---

## 如果仍然效果差

### 排查优先级

1. **验证 std schedule 是否工作** (检查 WandB 或运行 verify_std_schedule.py)
2. **对比相同 iteration** (ppo_twist 可能需要更多时间)
3. **检查 motion curriculum 是否被调用** (打印日志)
4. **尝试降低熵系数** (`entropy_coef=0.001`)
5. **尝试浅网络** (`actor_hidden_dims=[512, 256, 128]`)

### 如果 std schedule 不工作

可能是 `self.actor` 的嵌套结构不同，需要调试：

```python
# 在 _update_action_std() 中添加
print(f"Actor structure: {self.actor}")
print(f"Actor module[1]: {self.actor.module[1]}")
print(f"Actor module[1].module: {self.actor.module[1].module}")
print(f"Actor attributes: {dir(self.actor.module[1].module)}")
```

---

## 结论

**主要修复**: std schedule 的属性访问错误 (`std` → `actor_std`)

**下一步**: 重新训练并监控 WandB 的 `actor/action_std` 指标，确认调度生效
