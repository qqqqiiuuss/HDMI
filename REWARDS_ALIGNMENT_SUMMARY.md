# HDMI vs TWIST-MAIN Rewards 对齐总结

## ✅ 已完成的修改

### 1. action_rate_l2_twist ✅
**修改**: L2 平方和 → L2 范数（开方）

```python
# 修改前
return (action_diff ** 2).sum(dim=-1).unsqueeze(1)

# 修改后 ✅
return torch.norm(action_diff, dim=-1).unsqueeze(1)
```

**TWIST-MAIN 对应**:
```python
def _reward_action_rate(self):
    return torch.norm(self.last_actions - self.actions, dim=1)
```

---

### 2. feet_air_time_twist ✅
**修改**:
1. 接触检测: `current_contact_time > 0.02` → `force > 5.0`
2. 移动阈值: `> 0.1` → `> 0.05`

```python
# 修改前
in_contact = self.contact_sensor.data.current_contact_time[:, self.body_indices] > 0.02
is_moving = torch.norm(ref_root_vel[..., :2], dim=-1) > 0.1

# 修改后 ✅
contact_forces_z = self.contact_sensor.data.net_forces_w[:, self.body_indices, 2]
in_contact = contact_forces_z > 5.0
is_moving = torch.norm(ref_root_vel[..., :2], dim=-1) > 0.05
```

**TWIST-MAIN 对应**:
```python
def _reward_feet_air_time(self):
    contact = self.contact_forces[:, self.feet_indices, 2] > 5.
    # ...
    rew_airtime *= torch.norm(self._ref_root_vel[:, :2], dim=1) > 0.05
    return rew_airtime
```

---

### 3. feet_stumble_twist ✅
**修改**: 恢复为原始 TWIST-MAIN 实现

```python
# 最终版本 ✅ (与 TWIST-MAIN 完全一致)
def compute(self):
    contact_forces = self.contact_sensor.data.net_forces_w[:, self.contact_body_indices]

    xy_force_norm = contact_forces[..., :2].norm(dim=-1)
    z_force_abs = contact_forces[..., 2].abs()

    stumble_mask = xy_force_norm > 4.0 * z_force_abs
    stumble = stumble_mask.any(dim=-1).float()

    return stumble.unsqueeze(1)
```

**TWIST-MAIN 对应**:
```python
def _reward_feet_stumble(self):
    rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
         4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
    return rew.float()
```

**⚠️ 重要注意**: IsaacLab 的 ContactSensor 只报告法向力（Z 方向），XY 方向的力通常接近 0。因此此 reward 在训练中可能始终为 0。如需功能性实现，参考 `FEET_STUMBLE_BUG_FIX.md` 中的速度代理方案。

---

## 📊 完整对比表

| TWIST-MAIN Reward | HDMI Implementation | 权重 | 对齐状态 | 备注 |
|-------------------|---------------------|------|---------|------|
| **Tracking Rewards** |
| tracking_keybody_pos | tracking_keybody_pos_twist_aligned | 2.0 | ✅ 一致 | |
| tracking_joint_dof | tracking_joint_dof_twist_aligned | 0.6 | ✅ 一致 | |
| tracking_joint_vel | tracking_joint_vel_twist_aligned | 0.2 | ✅ 一致 | |
| tracking_root_pose | tracking_root_pose_twist_aligned | 0.6 | ✅ 一致 | |
| tracking_root_vel | tracking_root_vel_twist_aligned | 1.0 | ✅ 一致 | |
| **Regularization Rewards** |
| feet_slip | feet_slip_twist | -0.1 | ⚠️ 轻微差异 | 接触检测方式不同 |
| feet_contact_forces | feet_contact_forces_twist | -5e-4 | ✅ 一致 | |
| feet_stumble | feet_stumble_twist | -1.25 | ✅ 已对齐 | 功能性受限* |
| dof_pos_limits | joint_pos_limits_twist | -5.0 | ✅ 一致 | |
| dof_torque_limits | joint_torque_limits_twist | -1.0 | ✅ 一致 | |
| dof_vel | dof_vel_twist | -1e-4 | ✅ 一致 | |
| dof_acc | dof_acc_twist | -5e-8 | ✅ 一致 | |
| action_rate | action_rate_l2_twist | -0.01 | ✅ 已修复 | |
| feet_air_time | feet_air_time_twist | 5.0 | ✅ 已修复 | |
| ang_vel_xy | ang_vel_xy_twist | -0.01 | ✅ 一致 | |
| ankle_dof_acc | ankle_dof_acc_twist | -1e-7 | ✅ 一致 | |
| ankle_dof_vel | ankle_dof_vel_twist | -2e-4 | ✅ 一致 | |

\* **功能性受限**: IsaacLab ContactSensor 只报告法向力，XY 方向力接近 0

---

## 🎯 对齐状态总结

### ✅ 完全对齐 (11个)
1. ✅ action_rate_l2_twist **(已修复)**
2. ✅ feet_air_time_twist **(已修复)**
3. ✅ feet_stumble_twist **(已恢复原始实现)**
4. ✅ feet_contact_forces_twist
5. ✅ joint_pos_limits_twist
6. ✅ joint_torque_limits_twist
7. ✅ dof_vel_twist
8. ✅ dof_acc_twist
9. ✅ ang_vel_xy_twist
10. ✅ ankle_dof_acc_twist
11. ✅ ankle_dof_vel_twist

### ⚠️ 轻微差异 (1个)
1. ⚠️ feet_slip_twist
   - 接触检测: `current_contact_time > 0.02` vs `force > 5.0`
   - 算法逻辑一致，功能等价

---

## 📝 修改记录

### 2024-10-31

**修改 1: action_rate_l2_twist**
- 文件: `active_adaptation/envs/mdp/commands/twist/rewards_new.py:399-410`
- 从 `(action_diff ** 2).sum()` 改为 `torch.norm(action_diff)`
- 理由: TWIST-MAIN 使用 L2 范数而非平方和

**修改 2: feet_air_time_twist**
- 文件: `active_adaptation/envs/mdp/commands/twist/rewards_new.py:434-467`
- 接触检测: `current_contact_time > 0.02` → `force > 5.0`
- 移动阈值: `0.1 m/s` → `0.05 m/s`
- 理由: 完全对齐 TWIST-MAIN 实现

**修改 3: feet_stumble_twist**
- 文件: `active_adaptation/envs/mdp/commands/twist/rewards_new.py:286-314`
- 恢复为原始 TWIST-MAIN 实现（直接使用 XY/Z 力比较）
- 理由: 代码层面完全对齐（即使功能性受 IsaacLab 限制）

---

## ⚙️ 可选优化建议

### feet_slip_twist 接触检测

如果想完全对齐 TWIST-MAIN 的接触检测方式：

```python
# 当前 (使用 contact time)
in_contact = self.contact_sensor.data.current_contact_time[:, self.contact_body_indices] > 0.02

# 可改为 (使用 force，与 TWIST-MAIN 一致)
contact_forces_z = self.contact_sensor.data.net_forces_w[:, self.contact_body_indices, 2]
in_contact = contact_forces_z > 5.0
```

**影响**: 轻微，两种方法都能有效检测接触状态。

---

## 🔍 IsaacLab ContactSensor 限制说明

### 问题
IsaacLab 的 `ContactSensor.data.net_forces_w` 只报告**法向力（normal forces）**：
- Z 方向（垂直）: 有真实值 ✅
- XY 方向（水平）: 接近 0 ❌

### 实测对比
| 环境 | XY force | Z force |
|------|----------|---------|
| **Isaac Gym (TWIST-MAIN)** | 40-300+ | 正常 |
| **IsaacLab (HDMI)** | ~0.01 | 正常 |

### 受影响的 Reward
- ✅ **feet_slip_twist**: 不受影响（使用速度）
- ✅ **feet_contact_forces_twist**: 不受影响（只用 Z 方向）
- ⚠️ **feet_stumble_twist**: 受影响（需要 XY 方向力）

### feet_stumble 的选择

**选项 1: 使用原始实现**（当前方案）
- 代码层面完全对齐 TWIST-MAIN ✅
- 但功能性受限，训练中可能始终为 0 ❌

**选项 2: 使用速度代理**（替代方案，见 `FEET_STUMBLE_BUG_FIX.md`）
- 代码层面不对齐 TWIST-MAIN ❌
- 但功能性正常，能检测 stumbling ✅

**当前决定**: 使用选项 1（原始实现），保持代码一致性。

---

## 📚 相关文档

- `TWIST_REWARDS_COMPARISON.md`: 详细的逐项对比
- `FEET_STUMBLE_BUG_FIX.md`: feet_stumble 的 IsaacLab 限制和替代方案
- `TWIST_ALIGNMENT_CHANGES.md`: 历史修改记录

---

## ✅ 验证清单

训练前检查：
- [x] action_rate_l2_twist 使用 L2 范数（不是平方和）
- [x] feet_air_time_twist 使用 force > 5.0 检测接触
- [x] feet_air_time_twist 使用 0.05 m/s 移动阈值
- [x] feet_stumble_twist 使用原始 TWIST-MAIN 实现
- [x] 所有权重与 TWIST-MAIN 配置一致

训练中监控：
- [ ] feet_stumble_twist 是否始终为 0（预期，IsaacLab 限制）
- [ ] feet_slip_twist 是否正常工作
- [ ] feet_air_time_twist 是否只在移动时触发
- [ ] action_rate 惩罚是否合理（不过度抑制动作变化）
