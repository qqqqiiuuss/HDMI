# Reward函数实现详细对比分析

## TWIST-master vs HDMI-todesk

本文档详细对比两个项目中17个reward函数的具体实现，分析它们在算法层面的一致性。

---

## 📊 总览

| 分类 | TWIST-master | HDMI-todesk | 实现一致性 |
|-----|-------------|-------------|----------|
| **核心跟踪奖励** | 5个 | 5个 | ⚠️ **部分差异** |
| **正则化奖励** | 12个 | 12个 | ⚠️ **部分差异** |

---

## 1️⃣ 核心跟踪奖励 (Tracking Rewards)

### 1.1 关键点位置跟踪 (tracking_keybody_pos)

**TWIST-master 实现** (`humanoid_mimic.py:524-544`):
```python
def _reward_tracking_keybody_pos(self):
    # 获取关键点世界坐标
    key_body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3]
    # 转为相对根节点坐标
    key_body_pos = key_body_pos - self.root_states[:, 0:3].unsqueeze(1)

    if not self.global_obs:
        # 提取yaw角度
        base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        # 转为局部坐标系（只考虑yaw旋转）
        key_body_pos = convert_to_local_root_body_pos(base_yaw_quat, key_body_pos)

    # 参考动作同样处理
    tar_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :]
    tar_key_body_pos = tar_key_body_pos - self._ref_root_pos.unsqueeze(1)
    if not self.global_obs:
        _, _, ref_yaw = euler_from_quaternion(self._ref_root_rot)
        ref_yaw_quat = quat_from_euler_xyz(0*ref_yaw, 0*ref_yaw, ref_yaw)
        tar_key_body_pos = convert_to_local_root_body_pos(ref_yaw_quat, tar_key_body_pos)

    # 计算误差
    key_body_pos_diff = key_body_pos - tar_key_body_pos
    key_body_pos_err = torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)  # L2范数平方
    key_body_pos_err = torch.sum(key_body_pos_err, dim=-1)  # 所有关键点累加

    key_body_pos_scale = 10.0
    return torch.exp(-key_body_pos_scale * key_body_pos_err)
```

**HDMI-todesk 实现** (`hdmi/rewards.py:62-89`):
```python
class keypoint_pos_tracking_local_product(_tracking_keypoint):
    def compute(self):
        # 获取当前和参考位置
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]

        root_pos_asset = self.command_manager.robot_root_pos_w.clone()
        root_pos_motion = self.command_manager.ref_root_pos_w.clone()
        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w

        # 零化Z坐标，只保留XY平面
        root_pos_asset[..., 2] = 0.0
        root_pos_motion[..., 2] = 0.0
        # 只保留yaw旋转
        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)

        # 扩展维度以匹配body数量
        root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        # 转为局部坐标系
        body_pos_asset_relative = quat_apply_inverse(root_quat_asset, body_pos_asset - root_pos_asset)
        body_pos_motion_relative = quat_apply_inverse(root_quat_motion, body_pos_motion - root_pos_motion)

        # 计算误差
        diff = body_pos_motion_relative - body_pos_asset_relative
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)  # L2范数 - tolerance
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
```

**关键差异分析：**

| 维度 | TWIST-master | HDMI-todesk | 一致性 |
|-----|-------------|-------------|-------|
| **坐标转换** | 相对根节点，yaw局部坐标 | 相对根节点(XY平面)，yaw局部坐标 | ✅ **逻辑相同** |
| **误差计算** | `sum(diff²)` 然后 `sum(所有关键点)` | `norm(diff)` 然后 `mean(所有关键点)` | ❌ **不同** |
| **缩放因子** | 固定 `scale=10.0` | 可配置 `sigma=0.2` | ❌ **不同** |
| **容忍度** | 无 | `tolerance`容忍阈值 | ❌ **不同** |
| **最终公式** | `exp(-10.0 * sum(err²))` | `exp(-mean(norm(err)-tol) / 0.2)` | ❌ **显著不同** |

**⚠️ 结论：** 虽然都是指数衰减奖励，但：
1. TWIST使用L2范数的**平方和**，HDMI使用**平均L2范数**
2. TWIST的scale=10.0对应HDMI的sigma=0.2（但由于计算方式不同，实际效果不同）
3. HDMI增加了tolerance机制，小于容忍度的误差不惩罚

---

### 1.2 关节位置跟踪 (tracking_joint_dof)

**TWIST-master 实现** (`humanoid_mimic.py:435-440`):
```python
def _reward_tracking_joint_dof(self):
    dof_diff = self._ref_dof_pos - self.dof_pos
    dof_err = torch.sum(self._dof_err_w * dof_diff * dof_diff, dim=-1)  # 加权L2平方

    pos_scale = 0.15
    return torch.exp(-pos_scale * dof_err)
```

**HDMI-todesk 实现** (`hdmi/rewards.py:298-305`):
```python
class joint_pos_tracking_product(_tracking_joint):
    def compute(self):
        joint_pos_asset = self.command_manager.asset.data.joint_pos[:, self.joint_indices_asset]
        joint_pos_motion = self.command_manager.ref_joint_pos[:, self.joint_indices_motion]
        diff = joint_pos_motion - joint_pos_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)  # L1范数 - tolerance
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
```

**关键差异分析：**

| 维度 | TWIST-master | HDMI-todesk | 一致性 |
|-----|-------------|-------------|-------|
| **误差计算** | 加权L2平方 `sum(w * diff²)` | L1绝对值 `mean(\|diff\|)` | ❌ **完全不同** |
| **关节权重** | 使用 `dof_err_w` 加权 | 无权重 | ❌ **不同** |
| **缩放因子** | `scale=0.15` | `sigma=0.2` | ⚠️ **接近但不同** |
| **容忍度** | 无 | 有tolerance | ❌ **不同** |

**⚠️ 结论：** **实现差异较大**，TWIST使用加权L2范数，HDMI使用简单L1范数。

---

### 1.3 关节速度跟踪 (tracking_joint_vel)

**TWIST-master 实现** (`humanoid_mimic.py:448-453`):
```python
def _reward_tracking_joint_vel(self):
    vel_diff = self._ref_dof_vel - self.dof_vel
    vel_err = torch.sum(self._dof_err_w * vel_diff * vel_diff, dim=-1)  # 加权L2平方

    vel_scale = 0.01
    return torch.exp(-vel_scale * vel_err)
```

**HDMI-todesk 实现** (`hdmi/rewards.py:315-322`):
```python
class joint_vel_tracking_product(_tracking_joint):
    def compute(self):
        joint_vel_asset = self.command_manager.asset.data.joint_vel[:, self.joint_indices_asset]
        joint_vel_motion = self.command_manager.ref_joint_vel[:, self.joint_indices_motion]
        diff = joint_vel_motion - joint_vel_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
```

**关键差异分析：**

| 维度 | TWIST-master | HDMI-todesk |
|-----|-------------|-------------|
| **误差计算** | 加权L2平方 | L1绝对值 |
| **缩放因子** | `scale=0.01` | `sigma=0.5` |

**⚠️ 结论：** 与joint_pos类似，**实现方式不同**。

---

### 1.4 根节点姿态跟踪 (tracking_root_pose)

**TWIST-master 实现** (`humanoid_mimic.py:461-476`):
```python
def _reward_tracking_root_pose(self):
    if self.global_obs:
        root_pos_diff = self._ref_root_pos - self.root_states[:, 0:3]
    else:
        root_pos_diff = self._ref_root_pos[:, 2:3] - self.root_states[:, 2:3]  # 只比较Z坐标

    root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)  # 位置误差L2平方

    # 四元数角度差
    root_rot_err = torch_utils.quat_diff_angle(self.root_states[:, 3:7], self._ref_root_rot)
    root_rot_err *= root_rot_err  # 平方

    root_pose_scale = 5.0
    return torch.exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))  # 旋转权重0.1
```

**HDMI-todesk 实现** (`hdmi/rewards.py:186-208`):
```python
class keypoint_ori_tracking_local_product(_tracking_keypoint):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_quat_w[:, self.body_indices_asset]
        body_ori_motion = self.command_manager.ref_body_quat_w[:, self.body_indices_motion]

        root_quat_asset = self.command_manager.robot_root_quat_w
        root_quat_motion = self.command_manager.ref_root_quat_w

        # 只保留yaw旋转
        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)

        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        # 转为局部方向
        body_ori_asset_relative = quat_mul(quat_conjugate(root_quat_asset), body_ori_asset)
        body_ori_motion_relative = quat_mul(quat_conjugate(root_quat_motion), body_ori_motion)

        # 计算四元数差
        diff = quat_mul(quat_conjugate(body_ori_motion_relative), body_ori_asset_relative)
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)  # 转轴角再取范数
        error = (error - self.tolerance).clamp_min(0.0)
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
```

**关键差异分析：**

| 维度 | TWIST-master | HDMI-todesk |
|-----|-------------|-------------|
| **对象** | **只跟踪根节点姿态** | **跟踪所有关键点方向** |
| **位置处理** | 包含位置误差 | 无位置，只有方向 |
| **旋转误差** | 四元数角度差平方 | 轴角范数 |
| **权重** | 位置:旋转=1:0.1 | 无权重区分 |

**❌ 结论：** **完全不同**。TWIST跟踪根节点位姿，HDMI跟踪所有关键点的相对方向。

---

### 1.5 根节点速度跟踪 (tracking_root_vel)

**TWIST-master 实现** (`humanoid_mimic.py:490-507`):
```python
def _reward_tracking_root_vel(self):
    if self.global_obs:
        root_vel_diff = self._ref_root_vel - self.root_states[:, 7:10]
        root_ang_vel_diff = self._ref_root_ang_vel - self.root_states[:, 10:13]
    else:
        # 转为局部坐标系
        local_ref_root_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_vel)
        root_vel_diff = local_ref_root_vel - self.base_lin_vel
        local_ref_root_ang_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_ang_vel)
        root_ang_vel_diff = local_ref_root_ang_vel - self.base_ang_vel

    root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)  # L2平方
    root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)
    root_vel_scale = 1.0

    return torch.exp(-root_vel_scale * (root_vel_err + 0.5 * root_ang_vel_err))
```

**HDMI-todesk 实现** (`hdmi/rewards.py:245-253`):
```python
class keypoint_lin_vel_tracking_product(_tracking_keypoint):
    def compute(self):
        body_lin_vel_asset = self.command_manager.asset.data.body_com_lin_vel_w[:, self.body_indices_asset]
        body_lin_vel_motion = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion]
        diff = body_lin_vel_motion - body_lin_vel_asset
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
```

**关键差异分析：**

| 维度 | TWIST-master | HDMI-todesk |
|-----|-------------|-------------|
| **对象** | **只跟踪根节点速度** | **跟踪所有关键点线速度** |
| **角速度** | 包含角速度 | **没有角速度** |
| **权重** | 线速度:角速度=1:0.5 | 无 |

**❌ 结论：** **完全不同**。TWIST跟踪根节点线速度+角速度，HDMI跟踪所有关键点线速度。

---

## 2️⃣ 正则化奖励 (Regularization Rewards)

### 2.1 脚部滑动 (feet_slip)

**TWIST-master 实现** (`humanoid_mimic.py:610-615`):
```python
def _reward_feet_slip(self):
    contact = self.contact_forces[:, self.feet_indices, 2] > 5.  # Z方向力>5N
    foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 7:9], dim=2)  # XY速度
    rew = torch.sqrt(foot_speed_norm)  # 平方根
    rew *= contact  # 只在接触时惩罚
    return torch.sum(rew, dim=1)
```

**HDMI-todesk 实现** (`twist/rewards_new.py:224-252`):
```python
class feet_slip_twist(TrackReward):
    def compute(self):
        foot_velocities = self.command_manager.asset.data.body_lin_vel_w[:, self.asset_body_indices]
        in_contact = self.contact_sensor.data.current_contact_time[:, self.contact_body_indices] > 0.02

        xy_velocity = foot_velocities[..., :2].norm(dim=-1)  # XY速度范数
        slip = (in_contact.float() * xy_velocity).sum(dim=-1)  # 直接乘速度

        return slip.unsqueeze(1)
```

**关键差异分析：**

| 维度 | TWIST-master | HDMI-todesk |
|-----|-------------|-------------|
| **接触判断** | Z方向力 > 5N | 接触时间 > 0.02s |
| **速度处理** | `sqrt(norm(v_xy))` | `norm(v_xy)` |
| **最终值** | `sum(sqrt(v))` | `sum(v)` |

**⚠️ 结论：** **实现有差异**。TWIST使用平方根，HDMI直接用速度，惩罚强度曲线不同。

---

### 2.2 脚部接触力 (feet_contact_forces)

**TWIST-master 实现** (`humanoid_mimic.py:597-601`):
```python
def _reward_feet_contact_forces(self):
    rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)  # Z方向力
    rew[rew < self.cfg.rewards.max_contact_force] = 0  # 小于阈值清零
    rew[rew > self.cfg.rewards.max_contact_force] -= self.cfg.rewards.max_contact_force  # 大于阈值减去阈值
    return rew
```

**HDMI-todesk 实现** (`twist/rewards_new.py:255-275`):
```python
class feet_contact_forces_twist(TrackReward):
    def compute(self):
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.body_indices]
        contact_force_magnitude = contact_forces.norm(dim=-1)  # 三维力范数
        excessive_force = (contact_force_magnitude - self.max_contact_force).clamp_min(0.0)
        return excessive_force.sum(dim=-1).unsqueeze(1)
```

**关键差异分析：**

| 维度 | TWIST-master | HDMI-todesk |
|-----|-------------|-------------|
| **力类型** | 只Z方向力 | **三维力向量范数** |
| **处理方式** | `max(0, \|F_z\| - threshold)` | `max(0, \|\|F\|\| - threshold)` |

**⚠️ 结论：** **部分不同**。HDMI使用三维力范数，TWIST只用Z方向。

---

### 2.3 脚部绊倒 (feet_stumble)

**TWIST-master 实现** (`humanoid_mimic.py:592-595`):
```python
def _reward_feet_stumble(self):
    # XY方向力 > 4 * |Z方向力|
    rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
         4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
    return rew.float()
```

**HDMI-todesk 实现** (`twist/rewards_new.py:278-311`):
```python
class feet_stumble_twist(TrackReward):
    def compute(self):
        foot_velocities = self.command_manager.asset.data.body_lin_vel_w[:, self.asset_body_indices]
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.contact_body_indices]

        contact_mask = (contact_forces[..., 2] > self.threshold).float()  # Z方向力>threshold
        downward_velocity = (-foot_velocities[..., 2]).clamp_min(0.0)  # 向下速度
        stumble = (downward_velocity * contact_mask).sum(dim=-1)

        return stumble.unsqueeze(1)
```

**关键差异分析：**

| 维度 | TWIST-master | HDMI-todesk |
|-----|-------------|-------------|
| **判断依据** | XY力 > 4 * Z力 | **接触时向下速度** |
| **物理意义** | 侧向接触力过大 | **接触时还在下沉** |

**❌ 结论：** **完全不同的实现逻辑**！TWIST检查侧向力，HDMI检查下降速度。

---

### 2.4 关节位置限制 (dof_pos_limits)

**TWIST-master 实现** (`humanoid_mimic.py:583-586`):
```python
def _reward_dof_pos_limits(self):
    out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # 下限违反
    out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)  # 上限违反
    return torch.sum(out_of_limits, dim=1)
```

**HDMI-todesk 实现** (`twist/rewards_new.py:314-333`):
```python
class joint_pos_limits_twist(TrackReward):
    def compute(self):
        joint_pos = self.command_manager.asset.data.joint_pos
        joint_limits = self.command_manager.asset.data.soft_joint_pos_limits

        soft_limits = joint_limits * self.soft_factor  # 软限制（0.9倍）
        lower_violation = (soft_limits[:, :, 0] - joint_pos).clamp_min(0.0)
        upper_violation = (joint_pos - soft_limits[:, :, 1]).clamp_min(0.0)

        violation = (lower_violation + upper_violation).sum(dim=-1)
        return violation.unsqueeze(1)
```

**关键差异分析：**

| 维度 | TWIST-master | HDMI-todesk |
|-----|-------------|-------------|
| **限制类型** | 硬限制 | **软限制（0.9倍）** |
| **计算方式** | 相同 | 相同 |

**⚠️ 结论：** **逻辑相同**，但HDMI增加了soft_factor机制，提前触发惩罚。

---

### 2.5 关节扭矩限制 (dof_torque_limits)

**TWIST-master 实现** (`humanoid_mimic.py:588-590`):
```python
def _reward_dof_torque_limits(self):
    out_of_limits = torch.sum((torch.abs(self.torques) / self.torque_limits -
                                self.cfg.rewards.soft_torque_limit).clip(min=0), dim=1)
    return out_of_limits
```

**HDMI-todesk 实现** (`twist/rewards_new.py:336-352`):
```python
class joint_torque_limits_twist(TrackReward):
    def compute(self):
        joint_torques = self.command_manager.asset.data.applied_torque
        torque_limits = self.command_manager.asset.data.joint_effort_limits

        soft_limits = torque_limits * self.soft_factor  # 0.95倍
        violation = (joint_torques.abs() - soft_limits).clamp_min(0.0)

        return violation.sum(dim=-1).unsqueeze(1)
```

**✅ 结论：** **完全一致**。都使用相对扭矩比例，超过soft_factor阈值开始惩罚。

---

### 2.6 关节速度 (dof_vel)

**TWIST-master**:
```python
def _reward_dof_vel(self):
    return torch.sum(torch.square(self.dof_vel), dim=1)
```

**HDMI-todesk**:
```python
class dof_vel_twist(TrackReward):
    def compute(self):
        joint_vel = self.command_manager.asset.data.joint_vel
        return (joint_vel ** 2).sum(dim=-1).unsqueeze(1)
```

**✅ 结论：** **完全一致**。L2范数平方。

---

### 2.7 关节加速度 (dof_acc)

**TWIST-master**:
```python
def _reward_dof_acc(self):
    return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
```

**HDMI-todesk**:
```python
class dof_acc_twist(TrackReward):
    def compute(self):
        joint_vel = self.command_manager.asset.data.joint_vel
        if self.last_joint_vel is None:
            self.last_joint_vel = joint_vel.clone()
            return torch.zeros(self.num_envs, 1, device=self.device)

        joint_acc = (joint_vel - self.last_joint_vel) / self.env.step_dt
        self.last_joint_vel = joint_vel.clone()
        return (joint_acc ** 2).sum(dim=-1).unsqueeze(1)
```

**✅ 结论：** **完全一致**。有限差分近似加速度，L2范数平方。

---

### 2.8 动作变化率 (action_rate)

**TWIST-master**:
```python
def _reward_action_rate(self):
    return torch.norm(self.last_actions - self.actions, dim=1)  # L2范数
```

**HDMI-todesk**:
```python
class action_rate_l2_twist(TrackReward):
    def compute(self):
        action_buf = self.env.action_manager.action_buf
        action_diff = action_buf[:, :, 0] - action_buf[:, :, 1]
        return (action_diff ** 2).sum(dim=-1).unsqueeze(1)  # L2范数平方
```

**⚠️ 结论：** **稍有不同**。TWIST用L2范数，HDMI用L2范数的平方（差异很小）。

---

### 2.9 脚部空中时间 (feet_air_time)

**TWIST-master** (`humanoid_mimic.py:647-659`):
```python
def _reward_feet_air_time(self):
    contact = self.contact_forces[:, self.feet_indices, 2] > 5.
    self.contact_filt = torch.logical_or(contact, self.last_contacts)
    self.last_contacts = contact
    first_contact = (self.feet_air_time > 0.) * self.contact_filt
    self.feet_air_time += self.dt
    tgt_air_time = self.cfg.rewards.feet_air_time_target  # 0.5s
    air_time = (self.feet_air_time - tgt_air_time) * first_contact
    air_time = air_time.clamp(max=0.)  # 只奖励>=target的情况
    self.feet_air_time *= ~self.contact_filt
    rew_airtime = air_time.sum(dim=1)
    rew_airtime *= torch.norm(self._ref_root_vel[:, :2], dim=1) > 0.05  # 只在移动时奖励
    return rew_airtime
```

**HDMI-todesk** (`twist/rewards_new.py:399-436`):
```python
class feet_air_time_twist(TrackReward):
    def compute(self):
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_indices] > 0.02

        self.air_time += self.env.step_dt
        self.air_time[in_contact] = 0.0

        landing = in_contact & (~self.last_contact)
        reward = torch.zeros(self.num_envs, device=self.device)

        for i in range(len(self.body_indices)):
            landing_mask = landing[:, i]
            air_time_error = torch.abs(self.air_time[landing_mask, i] - self.target_air_time)
            reward[landing_mask] += torch.exp(-air_time_error / 0.25)  # 指数奖励

        self.last_contact = in_contact.clone()
        return reward.unsqueeze(1)
```

**关键差异分析：**

| 维度 | TWIST-master | HDMI-todesk |
|-----|-------------|-------------|
| **奖励时机** | 首次接触 | 着陆瞬间 |
| **奖励函数** | `clamp(t - target, max=0)` 线性 | `exp(-\|t - target\| / 0.25)` 指数 |
| **速度门控** | 只在移动时奖励 | **无速度门控** |

**⚠️ 结论：** **实现差异较大**。TWIST线性奖励且门控，HDMI指数奖励无门控。

---

### 2.10 XY角速度 (ang_vel_xy)

**TWIST-master**:
```python
def _reward_ang_vel_xy(self):
    return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
```

**HDMI-todesk**:
```python
class ang_vel_xy_twist(TrackReward):
    def compute(self):
        ang_vel = self.command_manager.asset.data.root_ang_vel_b
        return (ang_vel[..., :2] ** 2).sum(dim=-1).unsqueeze(1)
```

**✅ 结论：** **完全一致**。

---

### 2.11 踝关节加速度 (ankle_dof_acc)

**TWIST-master** (`g1_mimic_distill.py:354-356`):
```python
def _reward_ankle_dof_acc(self):
    ankle_dof_idx = [4, 5, 10, 11]  # 硬编码索引
    return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, ankle_dof_idx], dim=1)
```

**HDMI-todesk** (`twist/rewards_new.py:449-470`):
```python
class ankle_dof_acc_twist(TrackReward):
    def __init__(self, joint_names: List[str] | str = ".*ankle.*", **kwargs):
        super().__init__(**kwargs)
        joint_indices, self.joint_names = resolve_matching_names(joint_names,
                                                                  self.command_manager.asset.joint_names)
        self.joint_indices = torch.tensor(joint_indices, device=self.device, dtype=torch.long)
        self.last_joint_vel = None

    def compute(self):
        joint_vel = self.command_manager.asset.data.joint_vel[:, self.joint_indices]
        if self.last_joint_vel is None:
            self.last_joint_vel = joint_vel.clone()
            return torch.zeros(self.num_envs, 1, device=self.device)

        joint_acc = (joint_vel - self.last_joint_vel) / self.env.step_dt
        self.last_joint_vel = joint_vel.clone()
        return (joint_acc ** 2).sum(dim=-1).unsqueeze(1)
```

**关键差异分析：**

| 维度 | TWIST-master | HDMI-todesk |
|-----|-------------|-------------|
| **关节选择** | 硬编码索引 [4,5,10,11] | **正则表达式匹配 ".*ankle.*"** |
| **计算方式** | 相同 | 相同 |

**✅ 结论：** **逻辑一致**，但HDMI更灵活（支持不同机器人配置）。

---

### 2.12 踝关节速度 (ankle_dof_vel)

**TWIST-master**:
```python
def _reward_ankle_dof_vel(self):
    ankle_dof_idx = [4, 5, 10, 11]
    return torch.sum(torch.square(self.dof_vel[:, ankle_dof_idx]), dim=1)
```

**HDMI-todesk**:
```python
class ankle_dof_vel_twist(TrackReward):
    def __init__(self, joint_names: List[str] | str = ".*ankle.*", **kwargs):
        super().__init__(**kwargs)
        joint_indices, self.joint_names = resolve_matching_names(joint_names,
                                                                  self.command_manager.asset.joint_names)
        self.joint_indices = torch.tensor(joint_indices, device=self.device, dtype=torch.long)

    def compute(self):
        joint_vel = self.command_manager.asset.data.joint_vel[:, self.joint_indices]
        return (joint_vel ** 2).sum(dim=-1).unsqueeze(1)
```

**✅ 结论：** **逻辑一致**，HDMI更灵活。

---

## 📊 总结表格

| Reward函数 | 权重一致？ | 实现一致性 | 主要差异 |
|-----------|----------|----------|---------|
| **tracking_keybody_pos** | ✅ 2.0 | ⚠️ 部分不同 | 误差聚合方式不同（sum vs mean），sigma含义不同 |
| **tracking_joint_dof** | ✅ 0.6 | ❌ 显著不同 | L2加权 vs L1无权重，有无tolerance |
| **tracking_joint_vel** | ✅ 0.2 | ❌ 显著不同 | L2加权 vs L1无权重 |
| **tracking_root_pose** | ✅ 0.6 | ❌ 完全不同 | 根节点位姿 vs 所有关键点方向 |
| **tracking_root_vel** | ✅ 1.0 | ❌ 完全不同 | 根节点速度+角速度 vs 所有关键点线速度 |
| **feet_slip** | ❌ -0.1 vs -0.5 | ⚠️ 部分不同 | sqrt(v) vs v |
| **feet_contact_forces** | ✅ -5e-4 | ⚠️ 部分不同 | Z力 vs 三维力范数 |
| **feet_stumble** | ✅ -1.25 | ❌ 完全不同 | 侧向力 vs 向下速度 |
| **dof_pos_limits** | ✅ -5.0 | ✅ 基本一致 | HDMI有soft_factor |
| **dof_torque_limits** | ✅ -1.0 | ✅ 完全一致 | - |
| **dof_vel** | ✅ -1e-4 | ✅ 完全一致 | - |
| **dof_acc** | ✅ -5e-8 | ✅ 完全一致 | - |
| **action_rate** | ✅ -0.01 | ✅ 基本一致 | L2 vs L2² (差异极小) |
| **feet_air_time** | ✅ 5.0 | ⚠️ 显著不同 | 线性奖励+门控 vs 指数奖励无门控 |
| **ang_vel_xy** | ✅ -0.01 | ✅ 完全一致 | - |
| **ankle_dof_acc** | ✅ -1e-7 | ✅ 逻辑一致 | 硬编码索引 vs 正则匹配 |
| **ankle_dof_vel** | ✅ -2e-4 | ✅ 逻辑一致 | 硬编码索引 vs 正则匹配 |

---

## 🎯 最终结论

### ✅ 完全一致的reward (6/17)
1. dof_torque_limits
2. dof_vel
3. dof_acc
4. ang_vel_xy
5. ankle_dof_acc (逻辑一致)
6. ankle_dof_vel (逻辑一致)

### ⚠️ 部分差异但可能等效的reward (4/17)
1. keypoint_pos_tracking (误差聚合方式不同，但都是指数衰减)
2. dof_pos_limits (增加了soft_factor)
3. action_rate (L2 vs L2²，差异极小)
4. feet_slip (sqrt vs 原值，曲线不同但趋势相同)

### ❌ 实现显著不同的reward (7/17)
1. **tracking_joint_dof** - L2加权 vs L1无权重
2. **tracking_joint_vel** - L2加权 vs L1无权重
3. **tracking_root_pose** - 对象完全不同（根节点 vs 所有关键点）
4. **tracking_root_vel** - 对象完全不同（根节点 vs 所有关键点）
5. **feet_stumble** - 判断逻辑完全不同（侧向力 vs 下降速度）
6. **feet_contact_forces** - Z力 vs 三维力范数
7. **feet_air_time** - 线性+门控 vs 指数无门控

---

## 🔍 关键发现

### 1. **核心跟踪奖励存在本质差异**
TWIST的`tracking_root_pose`和`tracking_root_vel`只跟踪**根节点**，而HDMI跟踪**所有关键点**。这是架构级的差异，可能导致训练行为显著不同。

### 2. **误差聚合方式不同**
- TWIST倾向使用**加权L2平方和**（对大误差更敏感）
- HDMI使用**平均L1/L2范数**（对所有关键点均等对待）

### 3. **HDMI增加了更多灵活性**
- tolerance机制：小误差不惩罚
- 正则表达式匹配关节：适配不同机器人
- soft_factor：提前触发限制惩罚

### 4. **权重配置基本对齐，但实现公式不同**
虽然配置文件中的权重相同，但由于底层计算公式不同（L1 vs L2，sum vs mean等），**实际训练时的梯度大小和方向可能不同**。

---

## ⚠️ 训练影响评估

### 高风险差异项
这些差异可能导致训练结果显著不同：
1. **tracking_root_pose/vel** - 跟踪对象完全不同
2. **feet_stumble** - 判断逻辑完全不同
3. **tracking_joint_dof/vel** - 加权方式和范数类型不同

### 中等风险差异项
这些差异可能影响收敛速度和最终性能：
1. **keypoint_pos_tracking** - 误差聚合方式不同
2. **feet_air_time** - 奖励函数形状不同
3. **feet_slip** - 惩罚强度曲线不同

### 低风险差异项
这些差异影响较小：
1. **action_rate** - L2 vs L2² 差异极小
2. **ankle相关** - 只是索引选择方式不同

---

## 建议

如果希望**完全复现TWIST-master的训练效果**，需要：

1. **修改核心跟踪奖励**：
   - 修改`tracking_root_pose`为只跟踪根节点位姿
   - 修改`tracking_root_vel`为只跟踪根节点线速度+角速度
   - 添加关节加权系数`dof_err_w`

2. **调整正则化奖励**：
   - `feet_slip`改用sqrt(v)
   - `feet_stumble`改为检查侧向力
   - `feet_air_time`添加速度门控

3. **移除HDMI特有机制**：
   - 移除tolerance机制
   - 或调整sigma使其与TWIST的scale对应

是否需要我帮你修改代码以完全对齐TWIST-master的实现？
