# 修复 PD 增益问题 - 渐进式方案

## 问题总结

**现状**：
- ✅ ppo 训练正常
- ❌ ppo_twist 训练异常（penalty 大，velocity tracking 差）
- **原因**：ppo 的高噪声 (std=1.5) 掩盖了 PD 增益不匹配问题

## 已完成的代码修改

### 1. 创建了新的 TWIST-aligned robot 配置 ✅

**文件**: `active_adaptation/assets/g1.py`

- 新增 `G1_CYLINDER_CFG_TWIST` (第244-404行)
- 使用 TWIST-MASTER 的 PD 增益：
  - Hip: K=100, D=2 (原来 K=40.2, D=2.6)
  - Knee: K=150, D=4 (原来 K=99.1, D=6.3)
  - **Waist: K=150, D=4** (原来 K=28.5, D=1.8) ← 最关键！
  - Ankle: K=40, D=2 (原来 K=28.5, D=1.8)

### 2. 注册了新 robot ✅

**文件**: `active_adaptation/assets/__init__.py`

```python
ROBOTS = {
    "g1": G1_CYLINDER_CFG,           # 原有配置（ppo 使用）
    "g1_twist": G1_CYLINDER_CFG_TWIST,  # TWIST 配置（ppo_twist 使用）
}
```

---

## 使用方案

### 方案 1: 只修改 PD 增益（最小改动）⭐ 推荐

**修改**: `cfg/task/G1/twist/0927_twist_teacher_new.yaml`

```yaml
robot:
  name: g1_twist  # 改这一行即可
  robot_type: g1_29dof_nohand-feet_sphere
```

**其他配置保持不变**（不改 delay 和 alpha）

**预期效果**：
- Waist 刚度提升 5.3倍 (28.5→150)
- Hip pitch/yaw 刚度提升 2.5倍 (40.2→100)
- 运动更稳定，penalty 降低
- **但可能仍然不完美**（因为还有 delay）

---

### 方案 2: PD 增益 + 移除 delay（完全对齐 TWIST）

**修改**: `cfg/task/G1/twist/0927_twist_teacher_new.yaml`

```yaml
robot:
  name: g1_twist  # 使用 TWIST PD 增益
  robot_type: g1_29dof_nohand-feet_sphere

action:
  # ... action_scaling 保持不变 ...
  min_delay: 0    # 移除延迟
  max_delay: 0    # 移除延迟
  alpha: [1.0, 1.0]  # 禁用 EMA 滤波
```

**预期效果**：
- 完全对齐 TWIST-MASTER 的控制流程
- 效果应该最好
- **但改动较大，可能需要重新调参**

---

### 方案 3: 临时提高 ppo_twist 的 std（快速测试）

**不修改配置文件**，只在训练命令中override：

```bash
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    algo.init_noise_scale=1.5 \
    algo.use_std_schedule=false \
    suffix=ppo_twist_high_std
```

**预期效果**：
- ppo_twist 应该能正常训练（类似 ppo）
- **但这不是正确的解决方案**（治标不治本）

---

## 推荐实施步骤

### Step 1: 测试方案 1（只改 PD 增益）

```yaml
# cfg/task/G1/twist/0927_twist_teacher_new.yaml
robot:
  name: g1_twist  # ← 只改这一行
```

```bash
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096 \
    suffix=ppo_twist_fixed_pd
```

**监控指标**：
- `reward/tracking_root_vel` 应该提升 ↑
- `reward/tracking_joint_vel` 应该提升 ↑
- `reward/joint_torque_limits_twist` penalty 应该减少 ↑
- `reward/dof_vel_twist` penalty 应该减少 ↑

**如果效果不理想** → 继续 Step 2

---

### Step 2: 测试方案 2（PD + 移除 delay）

```yaml
# cfg/task/G1/twist/0927_twist_teacher_new.yaml
robot:
  name: g1_twist

action:
  min_delay: 0
  max_delay: 0
  alpha: [1.0, 1.0]
```

```bash
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096 \
    suffix=ppo_twist_full_aligned
```

**如果还不行** → 问题可能在其他地方（需要进一步诊断）

---

## 对比实验（验证诊断）

### 实验 A: ppo 使用低 std

```bash
python scripts/train.py \
    algo=ppo \
    task=G1/twist/0927_twist_teacher_new \
    algo.init_noise_scale=0.5 \
    suffix=ppo_low_std
```

**预测**: ppo 应该会出现类似 ppo_twist 的问题

### 实验 B: ppo_twist 使用高 std

```bash
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    algo.init_noise_scale=1.5 \
    algo.use_std_schedule=false \
    suffix=ppo_twist_high_std
```

**预测**: ppo_twist 应该能正常训练

**如果预测都对** → 证明我的诊断正确（PD 增益 + noise std 的相互作用）

---

## 回滚（如果需要）

### 撤销 PD 增益修改

```yaml
# cfg/task/G1/twist/0927_twist_teacher_new.yaml
robot:
  name: g1  # 改回原来的
```

### 删除新增代码（可选）

```bash
# 回滚 g1.py 的修改
git diff active_adaptation/assets/g1.py
git checkout active_adaptation/assets/g1.py

# 回滚 __init__.py 的修改
git checkout active_adaptation/assets/__init__.py
```

**注意**: 回滚后，`g1_twist` robot 将不可用

---

## PD 增益对比表

| 关节 | 原有 (g1) | TWIST (g1_twist) | 提升倍数 | 重要性 |
|------|-----------|------------------|----------|---------|
| **Hip pitch/yaw** | K=40.2, D=2.6 | K=100, D=2.0 | 2.5x | ⭐⭐⭐ |
| **Hip roll** | K=99.1, D=6.3 | K=100, D=2.0 | 1.0x K, 0.3x D | ⭐⭐ |
| **Knee** | K=99.1, D=6.3 | K=150, D=4.0 | 1.5x K, 0.6x D | ⭐⭐⭐ |
| **Ankle** | K=28.5, D=1.8 | K=40, D=2.0 | 1.4x | ⭐⭐ |
| **Waist** | K=28.5, D=1.8 | K=150, D=4.0 | **5.3x** | ⭐⭐⭐⭐⭐ |
| **Shoulder** | K≈14.3, D≈0.9 | K=40, D=5.0 | 2.8x | ⭐⭐ |
| **Elbow** | K≈14.3, D≈0.9 | K=40, D=5.0 | 2.8x | ⭐⭐ |

**最关键**: Waist 增益提升 **5.3倍**

---

## 为什么改完 reward 更低了？

### 可能的原因

#### 1. 初始不稳定期（正常）

**现象**: 训练初期 reward 突然下降

**原因**:
- 旧策略是在低 PD 增益下训练的
- 突然换成高 PD 增益，旧策略不适用
- 需要重新探索

**解决**: 耐心等待，给几千次 iteration

#### 2. PD 增益过高（不太可能）

**现象**: reward 持续很低，无法恢复

**原因**:
- TWIST 的 PD 增益可能不适合当前任务
- 或者其他参数需要同步调整

**解决**:
- 降低 PD 增益（介于原有和 TWIST 之间）
- 或者先用方案 3（提高 std）

#### 3. Delay 干扰（如果同时改了 delay）

**现象**: 机器人震荡、不稳定

**原因**:
- 移除 delay 后，策略需要重新学习
- 之前策略依赖 delay 的平滑效果

**解决**:
- 先只改 PD 增益（方案 1）
- 不要同时改 delay

---

## 建议的测试顺序

1. **先用方案 3 验证诊断** (5分钟)
   ```bash
   python scripts/train.py algo=ppo_twist task=G1/twist/0927_twist_teacher_new \
       algo.init_noise_scale=1.5 algo.use_std_schedule=false suffix=test_high_std
   ```
   - 如果能正常训练 → 诊断正确
   - 如果还是有问题 → 问题在别处

2. **再用方案 1 测试 PD 增益** (1小时)
   ```yaml
   robot.name: g1_twist  # 只改这个
   ```
   - 监控 1000 iterations
   - 如果 reward 回升 → 成功
   - 如果持续低 → 可能需要调整

3. **最后考虑方案 2** (完全对齐)
   - 只在方案 1 有效的前提下
   - 一次只改一个参数（先 delay，再 alpha）

---

## 总结

**代码已经修改好了**，现在有 3 个选择：

1. ⭐ **方案 1**: `robot.name: g1_twist` (最小改动，推荐)
2. **方案 2**: g1_twist + 移除 delay (完全对齐，激进)
3. **方案 3**: 提高 std (快速测试，不治本)

**我的建议**：
- 先用**方案 3 快速验证**诊断是否正确
- 确认后再用**方案 1 逐步修正** PD 增益
- ppo 保持不变，不受影响

**预期**：方案 1 应该能显著改善 ppo_twist 的训练效果。
