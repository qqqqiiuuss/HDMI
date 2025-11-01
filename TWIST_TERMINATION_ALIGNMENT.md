# TWIST-MAIN终止条件完全对齐方案

## TWIST-MAIN的完整终止条件

根据代码分析（`humanoid_mimic.py:275-333`），TWIST-MAIN有以下终止条件：

### 1. ❌ 躯干接触力终止（HDMI缺失）
```python
# TWIST: humanoid_mimic.py:276-277
contact_force_termination = torch.any(
    torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
    dim=1
)
self.reset_buf = contact_force_termination
```
**配置**: `terminate_after_contacts_on = ['torso_link']`

---

### 2. ✅ 根节点高度差异终止（HDMI已有）
```python
# TWIST: humanoid_mimic.py:280
height_cutoff = torch.abs(self.root_states[:, 2] - self._ref_root_pos[:, 2]) > 0.2
self.reset_buf |= height_cutoff
```
**配置**: `root_height_diff_threshold = 0.2`

**HDMI对应**: `cum_body_z_error` (threshold: 0.3) - **需要改回0.2**

---

### 3. ❌ Roll/Pitch超限终止（HDMI缺失）
```python
# TWIST: humanoid_mimic.py:282-285
roll_cut = torch.abs(self.roll) > self.cfg.rewards.termination_roll
pitch_cut = torch.abs(self.pitch) > self.cfg.rewards.termination_pitch
self.reset_buf |= roll_cut
self.reset_buf |= pitch_cut
```
**配置**: `termination_roll` 和 `termination_pitch` (默认值未设置，可能不触发)

---

### 4. ❌ Motion结束终止（HDMI缺失）
```python
# TWIST: humanoid_mimic.py:286-290, 294
motion_end = self.episode_length_buf * self.dt >= motion_length
if self.viewer is None:
    self.reset_buf |= motion_end
    self.time_out_buf |= motion_end
```
**这是关键**: 当motion播放完毕时自动重置

---

### 5. ✅ Timeout终止（HDMI已有）
```python
# TWIST: humanoid_mimic.py:292
self.time_out_buf = self.episode_length_buf > self.max_episode_length
self.reset_buf |= self.time_out_buf
```
**配置**: `max_episode_length = 500`

**HDMI对应**: `base.py:876` 自动处理

---

### 6. ❌ 速度过大终止（HDMI缺失）
```python
# TWIST: humanoid_mimic.py:298-299
vel_too_large = torch.norm(self.root_states[:, 7:10], dim=-1) > 5.
self.reset_buf |= vel_too_large
```
**阈值**: 5 m/s

---

### 7. ✅ Pose误差终止（HDMI已有但不完全一致）
```python
# TWIST: humanoid_mimic.py:301-328
if self._pose_termination:
    body_pos_dist = torch.sum(body_pos_diff ** 2, dim=-1)
    body_pos_dist = torch.max(body_pos_dist, dim=-1)[0]
    pose_fail = body_pos_dist > self._pose_termination_dist ** 2  # 0.7**2
    self.reset_buf |= pose_fail
```

**HDMI对应**: `cum_body_pos_error_local` (threshold: 1.0, min_steps: 2)

**差异**:
- TWIST: 立即触发（只要error > 0.7）
- HDMI: 需要连续2步超限才触发

---

### 8. ✅ First Step保护（HDMI已有）
```python
# TWIST: humanoid_mimic.py:330-333
first_step = self.episode_length_buf == 0
self.reset_buf[first_step] = 0
```

**HDMI对应**: `min_steps: 2` 实现了相同效果

---

## 对齐方案

### 方案A: 完全对齐TWIST-MAIN（推荐）

恢复所有TWIST的终止条件：

```yaml
termination:
  # 1. ✅ Pose误差终止 - 改为立即触发
  cum_body_pos_error_local:
    _target_: active_adaptation.envs.mdp.commands.twist.terminations.cum_body_pos_error_local
    body_names: [".*_hip_(pitch|yaw)_link", ".*_knee_link", ".*_ankle_roll_link", "pelvis", "torso_link", ".*_shoulder_pitch_link", ".*_elbow_link", ".*_wrist_yaw_link"]
    min_steps: 2         # 保持2（模拟first_step保护）
    threshold: 0.7       # 恢复到0.7（对齐TWIST）
    enabled: true

  # 2. ✅ 根节点高度误差 - 恢复原始阈值
  cum_body_z_error:
    _target_: active_adaptation.envs.mdp.commands.twist.terminations.cum_body_z_error
    body_names: "pelvis"
    min_steps: 2
    threshold: 0.2       # 恢复到0.2（对齐TWIST）
    enabled: true

  # 3. ❌ 躯干接触（TWIST有，但可能不常触发）
  # crash:
  #   _target_: active_adaptation.envs.mdp.terminations.crash
  #   body_names_expr: "torso_link"
  #   force_threshold: 1.0
  #   enabled: true

  # 4. ❌ Motion结束（关键！HDMI缺失）
  # 需要在terminations.py中实现
  # motion_end:
  #   _target_: active_adaptation.envs.mdp.commands.twist.terminations.motion_end
  #   enabled: true
```

---

### 方案B: 临时禁用所有终止条件（测试用）

**目的**: 验证motion的实际长度

```yaml
termination:
  # 全部禁用
  cum_body_pos_error_local:
    enabled: false

  cum_body_z_error:
    enabled: false
```

**预期结果**:
- 如果motion长度是8秒（400步@50Hz） → episode_len应该等于400
- 如果没有motion_end终止 → episode_len应该等于max_episode_length (500)

---

## 回答你的问题

### Q: 如果取消terminate，episode_len是不是就是400？

**A: 不一定！取决于是否有motion_end终止条件。**

#### 情况1: 完全没有motion_end终止
```python
# 没有这个条件
motion_end = episode_length * dt >= motion_length
```

**结果**: episode_len = max_episode_length = **500步**（不是400）

#### 情况2: 有motion_end终止
```python
# TWIST有这个条件
motion_end = episode_length * dt >= motion_length
if motion_length = 8秒:
    motion_end在 400步 触发
```

**结果**: episode_len = motion_length = **400步**

---

## 验证Motion长度

### 方法1: 打印Motion长度

```python
# 在command manager的reset中添加
print(f"Motion lengths: min={self.motion_len.min():.2f}s, "
      f"mean={self.motion_len.mean():.2f}s, "
      f"max={self.motion_len.max():.2f}s")
print(f"Corresponding steps @ 50Hz: {self.motion_len.mean() * 50:.0f}")
```

### 方法2: 临时禁用所有终止

使用方案B，观察episode_len：
- 如果变成500 → motion没有自动结束
- 如果是400 → motion长度确实是8秒

---

## 实现Motion End终止条件

如果HDMI缺少motion_end，需要添加：

### Step 1: 在terminations.py中添加

```python
# active_adaptation/envs/mdp/commands/twist/terminations.py

class motion_end(RobotTrackTermination):
    """
    TWIST: motion_end终止条件
    当motion播放完毕时触发
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self):
        # 获取当前时间
        current_time = self.env.episode_length_buf.float() * self.env.step_dt

        # 获取motion长度
        if hasattr(self.command_manager, 'motion_len'):
            motion_lengths = self.command_manager.motion_len
        else:
            # Fallback: 使用dataset的长度
            motion_lengths = torch.tensor([
                self.command_manager.dataset.lengths[mid]
                for mid in self.command_manager.motion_ids
            ], device=self.device)

        # 检查是否超过motion长度
        motion_finished = current_time >= motion_lengths

        return motion_finished.unsqueeze(-1)
```

### Step 2: 在配置中启用

```yaml
termination:
  motion_end:
    _target_: active_adaptation.envs.mdp.commands.twist.terminations.motion_end
    enabled: true
```

---

## 推荐的验证流程

### 1. 首先：临时禁用所有终止（方案B）

**目的**: 确认motion的实际长度

```yaml
termination:
  cum_body_pos_error_local:
    enabled: false
  cum_body_z_error:
    enabled: false
```

**观察**: `train/state/episode_len`
- 如果是500 → 说明没有motion_end
- 如果是400 → 说明有motion_end或motion确实是8秒

---

### 2. 然后：恢复TWIST的阈值（方案A）

```yaml
termination:
  cum_body_pos_error_local:
    threshold: 0.7  # 恢复原始值
    min_steps: 2
    enabled: true

  cum_body_z_error:
    threshold: 0.2  # 恢复原始值
    min_steps: 2
    enabled: true
```

**观察**: `train/state/episode_len`
- 如果训练初期≈80，训练后期增长到200-400 → 正常训练进度
- 如果始终≈80 → tracking质量没有提升

---

### 3. 最后：添加motion_end（如果需要）

如果验证步骤1发现episode_len=500（而不是400），说明缺少motion_end终止。

需要实现上面的motion_end终止条件。

---

## 总结

### TWIST vs HDMI终止条件对比

| 终止条件 | TWIST | HDMI | 需要修改 |
|---------|-------|------|---------|
| Pose误差 | threshold=0.7, 立即 | threshold=1.0, 连续2步 | ✅ 改回0.7 |
| 高度误差 | threshold=0.2 | threshold=0.3 | ✅ 改回0.2 |
| 躯干接触 | ✅ force>1.0 | ❌ 无 | ⚠️ 可选 |
| Motion结束 | ✅ 有 | ❌ 无 | ✅ 需要实现 |
| 速度过大 | ✅ vel>5.0 | ❌ 无 | ⚠️ 可选 |
| Roll/Pitch | ⚠️ 可能未设置 | ❌ 无 | ⚠️ 可选 |
| Timeout | ✅ 500 | ✅ 500 | ✅ 已对齐 |
| First Step保护 | ✅ 有 | ✅ min_steps=2 | ✅ 已对齐 |

### 关键差异

1. **Threshold放宽了** (0.7→1.0, 0.2→0.3)
2. **缺少motion_end** - 这可能是最关键的差异

### 回答原问题

**"如果取消terminate，episode_len是不是就是400？"**

- ❌ 不一定！
- 如果没有motion_end终止 → episode_len = 500
- 如果有motion_end终止 → episode_len = motion长度（可能是400）

**建议**: 先用方案B测试，确认motion的实际长度，然后再决定如何对齐。
