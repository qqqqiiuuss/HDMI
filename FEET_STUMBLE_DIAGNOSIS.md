# feet_stumble_twist Reward 一直为 0 的诊断和修复指南

## 问题描述

在使用 HDMI 框架训练时，`feet_stumble_twist` reward 一直是 0，但在 TWIST-master 中这个值会在 0 附近波动。

## 根本原因分析

### TWIST-master vs HDMI 的实现差异

#### 1. TWIST-master (Isaac Gym)
```python
# 获取所有 body 的 contact forces
net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

# Reward 计算
def _reward_feet_stumble(self):
    rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
         4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
    return rew.float()
```

**特点**:
- `self.contact_forces` 包含**所有 body** 的接触力
- Shape: `[num_envs, num_all_bodies, 3]`
- 通过 `self.feet_indices` 索引获取脚部的力

#### 2. HDMI (IsaacLab)
```python
# 只获取特定 body 的 contact forces
self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
self.contact_body_indices, self.body_names = self.contact_sensor.find_bodies(body_names)

# Reward 计算
def compute(self):
    contact_forces = self.contact_sensor.data.net_forces_w[:, self.contact_body_indices]
    xy_force_norm = contact_forces[..., :2].norm(dim=-1)
    z_force_abs = contact_forces[..., 2].abs()
    stumble_mask = xy_force_norm > 4.0 * z_force_abs
    stumble = stumble_mask.any(dim=-1).float()
    return stumble.unsqueeze(1)
```

**特点**:
- `ContactSensor` 只监控配置的特定 bodies
- Shape: `[num_envs, num_monitored_bodies, 3]`
- 需要通过 `find_bodies()` 来匹配 body names

### 可能的问题点

| 问题 | 可能性 | 影响 |
|------|--------|------|
| 1. Contact forces 数据一直是 0 | ⭐⭐⭐⭐⭐ | 最可能 - sensor 没有正确更新或配置错误 |
| 2. Body names 没有正确匹配 | ⭐⭐⭐⭐ | 可能 - `find_bodies()` 返回空列表 |
| 3. Sensor 配置的 prim_path 不正确 | ⭐⭐⭐ | 可能 - 没有监控到脚部 |
| 4. 阈值设置过高 (4.0) | ⭐⭐ | 不太可能 - TWIST 用的也是 4.0 |
| 5. IsaacLab vs Isaac Gym 的物理差异 | ⭐ | 不太可能 - 但需排查 |

## 诊断步骤

### 步骤 1: 使用诊断脚本

运行提供的 `debug_feet_stumble.py`:

```python
from debug_feet_stumble import diagnose_feet_stumble_reward

# 创建环境后
env = ... # 你的环境
diagnose_feet_stumble_reward(env, step_count=100)
```

**检查输出**:
- ✅ Contact sensor 是否找到
- ✅ Body names 是否正确匹配 (应该找到 `left_ankle_roll_link`, `right_ankle_roll_link`)
- ✅ Contact forces 是否全是 0
- ✅ XY/Z 力比率的最大值

### 步骤 2: 使用调试版本 Reward

在配置文件中使用 `feet_stumble_debug_version.py`:

```yaml
# cfg/task/G1/twist/0927_twist_teacher_new.yaml
reward:
  regularization:
    feet_stumble_twist:
      _target_: feet_stumble_debug_version.feet_stumble_twist_debug
      weight: -1.25
      enabled: true
      threshold: 1.0
      body_names: ".*ankle_roll_link"
```

运行训练并观察每 100 步的详细输出。

### 步骤 3: 检查 Contact Sensor 配置

查看 `locomotion.py` 中的配置:

```python
# active_adaptation/envs/locomotion.py:164-168
scene_cfg.contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*(ankle_roll|wrist_.*)_link",
    history_length=3,
    track_air_time=True
)
```

**验证**:
1. Prim path 是否正确
2. 机器人的 ankle 链接名称是否匹配

```bash
# 在 Isaac Sim 中检查实际的 prim 路径
python scripts/play.py task=... --verbose
# 查看输出中的 body names
```

## 修复方案

### 方案 1: 修复 Contact Sensor 配置 (最可能)

如果发现 contact forces 一直是 0，可能是 sensor 配置问题。

**检查点**:
```python
# 在环境初始化后添加检查
print("Contact sensor bodies:", env.scene["contact_forces"].body_names)
print("Asset bodies:", env.scene["robot"].body_names)
```

**可能的修复**:
```python
# locomotion.py
scene_cfg.contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",  # 监控所有 body，方便调试
    history_length=3,
    track_air_time=True,
    debug_vis=True  # 启用可视化
)
```

### 方案 2: 使用 Asset 的 Body Velocities 代替 Contact Forces

如果 ContactSensor 确实有问题，可以修改实现使用 asset 的数据:

```python
class feet_stumble_twist_alternative(TrackReward):
    """
    备选方案：使用 asset 的 contact forces
    """
    def __init__(self, body_names: List[str] | str = ".*ankle_roll_link", **kwargs):
        super().__init__(**kwargs)

        # 从 asset 获取 body indices
        self.asset = self.env.scene["robot"]
        body_indices, self.body_names = resolve_matching_names(
            body_names, self.asset.body_names
        )
        self.body_indices = torch.tensor(body_indices, device=self.device, dtype=torch.long)

        # 仍然尝试从 contact sensor 获取
        from isaaclab.sensors import ContactSensor
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

    def compute(self):
        # 方法1: 尝试从 contact sensor 获取
        try:
            contact_forces = self.contact_sensor.data.net_forces_w
            # 需要将 asset body indices 映射到 sensor indices
            # ... (复杂，需要实现映射)
        except:
            pass

        # 方法2: 从 asset 的 contact forces 获取 (IsaacLab 可能提供)
        # 检查 self.asset 是否有 contact forces 属性
        if hasattr(self.asset, 'root_physx_view'):
            # 尝试从 PhysX view 获取
            pass

        # 实现 stumble 检测...
```

### 方案 3: 修改 Body Names 匹配模式

如果 `find_bodies()` 没有找到匹配的 bodies:

```python
# 尝试不同的匹配模式
body_patterns = [
    ".*ankle_roll.*",      # 更宽松的匹配
    ".*ankle.*",           # 匹配所有 ankle
    "left_ankle_roll_link|right_ankle_roll_link",  # 精确匹配
]

for pattern in body_patterns:
    indices, names = self.contact_sensor.find_bodies(pattern)
    if len(indices) > 0:
        print(f"✓ Pattern '{pattern}' matched: {names}")
        break
```

### 方案 4: 降低阈值 (临时方案)

如果确认 contact forces 有值但比率总是小于 4.0:

```yaml
# 临时降低阈值来测试
feet_stumble_twist:
  weight: -1.25
  enabled: true
  threshold: 2.0  # 降低到 2.0 测试
  body_names: ".*ankle_roll_link"
```

修改代码:
```python
def compute(self):
    # ...
    stumble_mask = xy_force_norm > 2.0 * z_force_abs  # 使用 self.threshold
    # ...
```

## 完整的修复代码示例

创建文件 `active_adaptation/envs/mdp/commands/twist/rewards_fixed.py`:

```python
import torch
from typing import List
from active_adaptation.envs.mdp.commands.twist.rewards_new import TrackReward
from active_adaptation.utils.string_utils import resolve_matching_names

class feet_stumble_twist_fixed(TrackReward):
    """
    修复版本的 feet_stumble_twist
    增加了多种获取 contact forces 的方法
    """
    def __init__(self, threshold: float = 4.0, body_names: List[str] | str = ".*ankle_roll_link", **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

        # 方法 1: 尝试从 contact sensor 获取
        from isaaclab.sensors import ContactSensor
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        # 获取 sensor body indices
        sensor_indices, sensor_names = self.contact_sensor.find_bodies(body_names)
        self.sensor_body_indices = torch.tensor(sensor_indices, device=self.device, dtype=torch.long)

        # 获取 asset body indices (备用方案)
        asset_indices, asset_names = resolve_matching_names(
            body_names, self.command_manager.asset.body_names
        )
        self.asset_body_indices = torch.tensor(asset_indices, device=self.device, dtype=torch.long)

        # 打印诊断信息
        print(f"\n[feet_stumble_twist_fixed] 初始化:")
        print(f"  Sensor bodies: {sensor_names}")
        print(f"  Asset bodies: {asset_names}")
        print(f"  Threshold: {threshold}")

        # 选择使用的方法
        if len(sensor_indices) > 0:
            self.use_sensor = True
            self.body_names = sensor_names
            print(f"  ✓ 使用 ContactSensor 方法")
        elif len(asset_indices) > 0:
            self.use_sensor = False
            self.body_names = asset_names
            print(f"  ⚠️  ContactSensor 未找到匹配bodies，使用 Asset 方法")
        else:
            raise ValueError(f"无法找到匹配 '{body_names}' 的 bodies!")

    def compute(self):
        if self.use_sensor:
            # 从 contact sensor 获取
            contact_forces = self.contact_sensor.data.net_forces_w[:, self.sensor_body_indices]
        else:
            # 备用：尝试从其他来源获取
            # 这需要根据 IsaacLab 的具体 API 实现
            raise NotImplementedError("Asset-based contact forces not implemented yet")

        # 计算 stumble
        xy_force_norm = contact_forces[..., :2].norm(dim=-1)
        z_force_abs = contact_forces[..., 2].abs()

        stumble_mask = xy_force_norm > self.threshold * z_force_abs
        stumble = stumble_mask.any(dim=-1).float()

        return stumble.unsqueeze(1)
```

## 验证修复

修复后，运行以下检查:

```python
# 1. 打印 reward 值
print(f"feet_stumble_twist reward: {env.reward_manager.rewards['feet_stumble_twist'].mean()}")

# 2. 检查统计
# 在训练 1000 步后
stumble_values = []
for _ in range(1000):
    env.step(actions)
    stumble_values.append(env.reward_manager.rewards['feet_stumble_twist'].mean().item())

print(f"Stumble reward 统计:")
print(f"  - 平均值: {np.mean(stumble_values):.6f}")
print(f"  - 最大值: {np.max(stumble_values):.6f}")
print(f"  - 非零次数: {np.count_nonzero(stumble_values)} / 1000")
```

**期望结果**:
- 非零次数 > 0 (表示 reward 有在工作)
- 平均值应该是一个小的负值 (因为 weight = -1.25)
- 与 TWIST-master 的行为类似

## 总结

最可能的问题是 **Contact Sensor 的数据没有正确更新**或**配置不正确**。

**推荐的调试流程**:
1. ✅ 运行 `debug_feet_stumble.py` 诊断脚本
2. ✅ 使用调试版本的 reward 查看详细输出
3. ✅ 检查 contact forces 是否为 0
4. ✅ 根据诊断结果选择对应的修复方案

**快速测试**:
```bash
# 运行训练并grep查看调试输出
python scripts/train.py task=G1/twist/0927_twist_teacher_new | grep -A 10 "feet_stumble"
```
