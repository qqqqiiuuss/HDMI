# HDMI vs TWIST Episode Length 差异分析

## 问题描述

训练曲线显示：
- **HDMI**: episode_length 最大约 **80 steps**
- **TWIST-MAIN**: episode_length 最大约 **400 steps**

配置显示两者的`max_episode_length`都是500，但实际平均episode length相差5倍！

---

## 根本原因：终止条件差异

### 1. ✅ First Step保护缺失（已修复）

**TWIST-MAIN** (`humanoid_mimic.py:333`):
```python
first_step = self.episode_length_buf == 0
self.reset_buf[first_step] = 0  # Do not reset on first step
```
**TWIST永远不会在第一步触发终止**

**HDMI原配置**:
```yaml
cum_body_pos_error_local:
  min_steps: 1  # 第一步就可以触发终止
```

**修复方案**: 已将`min_steps`从1改为2

```yaml
cum_body_pos_error_local:
  min_steps: 2  # 模拟first_step保护
  threshold: 0.7

cum_body_z_error:
  min_steps: 2  # 模拟first_step保护
  threshold: 0.2
```

---

### 2. ⚠️ 误差计算方式（已验证一致）

**TWIST-MAIN** (`humanoid_mimic.py:310-318`):
```python
body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1)  # 平方和
body_pos_dist = torch.max(body_pos_dist, dim=-1)[0]
pose_fail = body_pos_dist > self._pose_termination_dist ** 2  # Σ(diff²) > 0.49
```

**HDMI** (`terminations.py:144`):
```python
body_pos_error = (ref_body_pos_local - robot_body_pos_local).norm(dim=-1)  # L2范数
self.error[:] = body_pos_error.max(dim=1).values
# 比较: ||diff|| > 0.7
```

**数学等价性**:
- TWIST: `Σ(diff²) > threshold²`
- HDMI: `√(Σ(diff²)) > threshold`
- 等价！✅

---

### 3. ⚠️ 可能的其他差异

即使修复了first_step保护，如果episode length仍然差距很大，可能存在以下差异：

#### 差异1: 初始化噪声

**TWIST-MAIN配置**:
```python
randomize_start_pos = True
randomize_start_yaw = False  # 不随机化yaw
```

**HDMI配置** (`0927_twist_teacher_new.yaml:79-96`):
```yaml
pose_range:
  x: [-0.05, 0.05]
  y: [-0.05, 0.05]
  z: [-0.01, 0.01]
  roll: [-0.05, 0.05]
  pitch: [-0.05, 0.05]
  yaw: [0.0, 0.0]  # ✅ 已对齐，不随机化

velocity_range:
  x: [-0.2, 0.2]
  y: [-0.2, 0.2]
  z: [-0.2, 0.2]
  roll: [-0.5, 0.5]
  pitch: [-0.5, 0.5]
  yaw: [-0.5, 0.5]

init_joint_pos_noise: 0.1
init_joint_vel_noise: 0.1
```

**对比TWIST**: TWIST的初始化噪声可能更小，导致更不容易触发终止条件。

---

#### 差异2: 运动数据质量

如果HDMI使用的运动数据与TWIST不同：
- 运动质量差
- 参考轨迹不平滑
- 机器人难以跟踪

会导致tracking error增大，更容易触发终止。

---

#### 差异3: Curriculum Learning

**TWIST-MAIN** (`humanoid_mimic.py:260`):
```python
motion_difficulty_ratio = (self.motion_difficulty - 1) / 8
self.motion_termination_dist = (self._pose_termination_dist - 0.1) * motion_difficulty_ratio + 0.1
# 当motion difficulty = 1时，termination dist = 0.1（更严格）
# 当motion difficulty = 9时，termination dist = 0.7（更宽松）
```

TWIST有**curriculum learning**机制：
- 训练初期：termination distance更小（0.1），容易触发终止
- 训练后期：termination distance更大（0.7），更难触发终止

**HDMI**: 没有这个机制，threshold始终是0.7

**影响**: TWIST在训练后期会有更长的episode，而HDMI始终一样。

---

#### 差异4: 其他终止条件

**TWIST-MAIN的额外终止条件** (`humanoid_mimic.py:275-299`):
```python
# 1. 躯干接触
contact_force_termination = torch.any(
    torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
    dim=1
)

# 2. 根节点高度误差
height_cutoff = torch.abs(self.root_states[:, 2] - self._ref_root_pos[:, 2]) > 0.2

# 3. Roll/Pitch超限
roll_cut = torch.abs(self.roll) > termination_roll
pitch_cut = torch.abs(self.pitch) > termination_pitch

# 4. 速度过大
vel_too_large = torch.norm(self.root_states[:, 7:10], dim=-1) > 5.

# 5. 运动结束
motion_end = self.episode_length_buf * self.dt >= motion_length
```

**HDMI配置**:
```yaml
cum_body_pos_error_local: enabled ✅
cum_body_z_error: enabled ✅
crash (torso contact): disabled ❌ (被注释)
roll_cut: ❌ 未配置
pitch_cut: ❌ 未配置
vel_too_large: ❌ 未配置
```

**分析**: HDMI缺少一些终止条件，但这应该会导致episode**更长**，而不是更短。

---

## 诊断步骤

### 步骤1: 验证first_step修复是否有效

重新训练后，检查episode_length是否增加：
- 如果从80增加到接近400 → 问题解决 ✅
- 如果仍然在80左右 → 还有其他问题 ⚠️

### 步骤2: 打印终止原因统计

在训练中添加logging，统计每种终止条件触发的频率：

```python
# 在check_termination()中添加
termination_reasons = {
    'body_pos_error': (cum_body_pos_error > threshold).sum().item(),
    'body_z_error': (cum_body_z_error > threshold).sum().item(),
    'timeout': (episode_length > max_length).sum().item(),
}
print(f"Termination reasons: {termination_reasons}")
```

**预期**:
- 如果`body_pos_error`占比很高 → tracking困难，检查初始化噪声或运动数据
- 如果`body_z_error`占比很高 → 高度控制问题
- 如果`timeout`占比很高 → 正常

---

### 步骤3: 对比tracking error曲线

在WandB中查看：
- `error/tracking_keybody_pos` - 关键点位置误差
- `error/cum_body_pos_error` - 终止条件的误差值

**对比**:
- HDMI的tracking error是否远大于TWIST？
- HDMI的error是否经常超过0.7？

---

### 步骤4: 检查初始化噪声

可以尝试减小HDMI的初始化噪声：

```yaml
# 当前配置
init_joint_pos_noise: 0.1
init_joint_vel_noise: 0.1

# 可以尝试
init_joint_pos_noise: 0.01  # 减小10倍
init_joint_vel_noise: 0.01
```

---

### 步骤5: 检查运动数据

确认HDMI和TWIST使用的是**同一批运动数据**：
- HDMI: `data_path: /home/jane/workspace/xmh/HDMI/small_dataset`
- TWIST: `motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/twist_dataset.yaml"`

如果数据不同，可能导致质量差异。

---

## 修改总结

### ✅ 已修改

1. **cum_body_pos_error_local**: `min_steps: 1 → 2`
2. **cum_body_z_error**: `min_steps: 1 → 2`

### ⚠️ 待验证

1. 初始化噪声是否过大
2. 运动数据质量是否一致
3. 是否需要添加curriculum learning

---

## 预期结果

修改`min_steps=2`后，训练曲线应该显示：
- Episode length从80增加到**至少200+**
- 如果增加到400左右 → 完全对齐 ✅
- 如果仍然<200 → 需要进一步调查初始化噪声或运动数据

---

## 下一步行动

1. ✅ 重新训练HDMI（使用修改后的配置）
2. 📊 对比新的episode_length曲线
3. 🔍 如果仍有差异，执行诊断步骤2-5
4. 📝 记录最终结果
