# feet_stumble_twist Bug 修复说明

## 问题根源

### IsaacLab ContactSensor 的限制

IsaacLab 的 `ContactSensor` **只报告法向力（normal forces）**，不报告切向力。

从 IsaacLab 源码（`contact_sensor.py:35`）：
```python
"""
The contact sensor reports the **normal contact forces** on a rigid body in the world frame.
"""
```

**实测结果**：
- `contact_forces[..., 2]` (Z方向): **有值** ✅ — 垂直地面的法向力
- `contact_forces[..., :2]` (XY方向): **全是0** ❌ — 没有切向力数据

### TWIST-master 的实现

TWIST-master 使用 Isaac Gym 的 `acquire_net_contact_force_tensor()`，它返回**完整的3D接触力**（包括法向和切向）：

```python
# TWIST-master (Isaac Gym)
net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

def _reward_feet_stumble(self):
    # XY 和 Z 方向都有真实的力数据
    rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
         4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
    return rew.float()
```

### 为什么 HDMI 实现一直返回 0

```python
# HDMI 原实现
contact_forces = self.contact_sensor.data.net_forces_w[:, self.contact_body_indices]
xy_force_norm = contact_forces[..., :2].norm(dim=-1)  # ← 总是 0
z_force_abs = contact_forces[..., 2].abs()            # ← 有值

stumble_mask = xy_force_norm > 4.0 * z_force_abs      # ← 0 > X*Z，永远 False
```

因为 XY 方向的力总是 0，`stumble_mask` 永远不会触发。

## 解决方案

### 方案 1: 使用速度作为代理（已实现）

**原理**：脚在接触地面时横向移动过快 = stumbling

```python
# 检测接触
in_contact = contact_forces_z.abs() > 5.0

# 获取脚的 XY 速度
foot_velocities = asset.data.body_lin_vel_w[:, body_indices]
xy_velocity_norm = foot_velocities[..., :2].norm(dim=-1)

# 脚在接触时横向速度过大 = stumble
stumble_mask = in_contact & (xy_velocity_norm > 0.5)
```

**优点**：
- 直接可用，不需要额外数据
- 物理上合理：stumbling 时脚会快速横向滑动
- 与 TWIST 的意图一致

**缺点**：
- 不是精确的力比较
- 阈值（0.5 m/s）需要调优

### 方案 2: 从 PhysX API 直接获取完整接触力

IsaacSim 的底层 PhysX 应该有完整的接触力数据，但 IsaacLab 的 ContactSensor 封装只暴露了法向力。

**可能的途径**：
1. 直接访问 PhysX contact report API
2. 使用 `RigidContact` 类（IsaacSim core）
3. 通过 `asset.root_physx_view` 获取

**示例代码（需要测试）**：
```python
# 尝试从 PhysX view 获取完整力
from omni.isaac.core.prims import RigidContactView

# 或者
contact_report = self.contact_physx_view.get_contact_force_data()
```

**需要研究 IsaacLab 的内部实现**，可能需要修改底层代码。

### 方案 3: 完全禁用这个 Reward

如果这个 reward 对训练效果影响不大，可以暂时禁用：

```yaml
# cfg/task/G1/twist/0927_twist_teacher_new.yaml
feet_stumble_twist:
  weight: -1.25
  enabled: false  # ← 禁用
```

然后依赖其他 feet rewards (`feet_slip`, `feet_contact_forces`) 来约束步态。

## 推荐做法

### 短期（立即可用）

使用**方案 1（速度代理）**，已经在 `rewards_new.py` 中实现。

**需要调优的参数**：
- `in_contact` 阈值：目前是 5.0 N
- `xy_velocity` 阈值：目前是 0.5 m/s

**调优方法**：
1. 在训练日志中监控 stumble 触发频率
2. 与 feet_slip reward 对比，确保不冲突
3. 根据实际表现调整阈值

### 中期（研究方案）

研究如何从 IsaacSim/PhysX 获取完整接触力：

1. 查看 IsaacLab 的 `RigidContactView` 实现
2. 检查是否可以通过配置让 ContactSensor 报告切向力
3. 联系 IsaacLab 开发者确认是否有 API

### 长期（框架改进）

如果完整接触力是必需的：

1. 向 IsaacLab 提交 feature request
2. 或者自己扩展 ContactSensor 类
3. 贡献回 IsaacLab 社区

## 验证修复

### 测试代码

```python
# 在训练中添加调试输出
if step % 100 == 0:
    stumble_val = env.reward_manager.rewards['feet_stumble_twist'].mean().item()
    print(f"Step {step}, feet_stumble_twist: {stumble_val}")

    # 检查是否有触发
    if abs(stumble_val) > 1e-6:
        print(f"  ✓ Stumble detected!")
```

### 期望结果

修复后，`feet_stumble_twist` 应该：
- **不再一直是 0**
- **偶尔出现负值**（当检测到 stumble 时）
- 与 TWIST-master 的行为类似（偶尔波动）

## 其他受影响的 Rewards

以下 rewards 也使用 ContactSensor，但它们只需要法向力，**不受影响**：

✅ `feet_slip_twist` - 使用 `current_contact_time` 和速度
✅ `feet_contact_forces_twist` - 只检查 Z 方向力
✅ `feet_air_time_twist` - 使用接触状态，不需要XY力

## 参考资料

- IsaacLab ContactSensor 源码: `IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py`
- IsaacSim PhysX ContactReport API: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_contact_report_a_p_i.html
- TWIST-master 实现: `/home/ubuntu/DATA2/workspace/xmh/TWIST-master/legged_gym/legged_gym/envs/base/humanoid_mimic.py:592`

## 总结

**问题**：IsaacLab 的 ContactSensor 只报告法向力，导致原始的 stumble 检测（基于 XY/Z 力比较）失效。

**修复**：使用脚的 XY 速度作为代理，检测"脚在接触时快速横向移动"。

**状态**：✅ 已修复并提交到 `rewards_new.py`
