# 🐛 参考Motion朝向反转问题修复

## 问题描述

在play模式下，参考motion显示向前走，但机器人实际会后退。

## 根本原因

**Observation和Reward的坐标系转换不一致**：

### Reward计算（正确）
```python
# rewards.py line 82-83
# 各自转换到自己的局部坐标系
body_pos_asset_relative = quat_apply_inverse(root_quat_asset, body_pos_asset - root_pos_asset)
body_pos_motion_relative = quat_apply_inverse(root_quat_motion, body_pos_motion - root_pos_motion)
```

### Observation转换（有问题）
```python
# observations.py line 1245
# 参考motion被转换到机器人坐标系
ref_pos_b = quat_apply_inverse(robot_root_quat_w, ref_pos_w - robot_root_pos_w)
```

**结果**：
- 如果机器人朝向和参考motion朝向相反（如180°差异）
- Observation中的参考位置会反向
- Policy学到的是"跟随反向的参考"
- 导致向前走变成后退

---

## 修复方案

### 🎯 方案1：修改配置文件（最简单，推荐）

**文件**：`cfg/task/G1/twist/0927_twist_teacher_new.yaml`

**修改**：
```yaml
# 修改前（line 119, 140）
ref_motion_windowed:
  coordinate_frame: robot_root    # ❌ 会导致朝向问题

# 修改后
ref_motion_windowed:
  coordinate_frame: world         # ✅ 使用世界坐标系
```

**优点**：
- 最简单，只改配置
- 避免所有坐标转换问题
- 对训练没有影响（reward计算不受影响）

**缺点**：
- Observation变大（包含世界坐标信息）
- 可能需要重新训练

---

### 🔧 方案2：修改Observation代码（彻底修复）

**文件**：`active_adaptation/envs/mdp/commands/twist/observations.py`

**修改**：line 1237-1272的坐标转换逻辑

**原代码**：
```python
# 使用机器人root转换参考motion
ref_pos_b = quat_apply_inverse(robot_root_quat_w, ref_pos_w - robot_root_pos_w)
```

**修复后**：
```python
# 使用参考motion自己的root转换（与reward一致）
ref_root_quat_yaw = yaw_quat(ref_root_quat_w)
ref_pos_local = quat_apply_inverse(ref_root_quat_yaw, ref_pos_w - ref_root_pos_w)
```

**优点**：
- 与reward计算完全一致
- 彻底解决问题

**缺点**：
- 需要修改代码
- 需要重新训练已有的模型

---

### 🚀 方案3：Play时强制对齐朝向（临时方案）

如果只是为了play/可视化，可以在`command.py`的`reset`中添加：

```python
def reset(self, env_ids):
    # ... 现有代码 ...

    # 强制对齐参考motion和机器人的yaw角度
    robot_yaw = get_yaw_from_quat(self.asset.data.root_quat_w[env_ids])
    ref_yaw = get_yaw_from_quat(self.ref_root_quat_w[env_ids])
    yaw_diff = ref_yaw - robot_yaw

    # 旋转参考motion以匹配机器人朝向
    self.ref_root_quat_w[env_ids] = quat_from_yaw(robot_yaw)
    # 旋转所有body位置...
```

**优点**：
- 不影响训练
- 只改善可视化效果

**缺点**：
- 治标不治本
- 不解决根本问题

---

## 推荐流程

### 1. 立即修复（Play/可视化）

修改配置文件：
```bash
nano cfg/task/G1/twist/0927_twist_teacher_new.yaml

# 找到 coordinate_frame: robot_root (line 119, 140)
# 改为：
coordinate_frame: world
```

重新play测试：
```bash
python scripts/play.py task=G1/twist/0927_twist_teacher_new checkpoint_path=xxx
```

---

### 2. 长期修复（新训练）

如果需要开始新的训练：

1. 应用方案2的代码修复（observations.py）
2. 或者使用方案1的配置修复
3. 从头开始训练新模型

---

## 验证方法

### ✅ 修复成功的标志

Play时应该看到：
1. 参考motion的skeleton向前走
2. 机器人也向前走（跟随参考motion）
3. 两者朝向一致

### ❌ 问题依旧的标志

1. 参考motion向前，机器人后退
2. 两者镜像运动
3. 朝向相反

---

## 技术细节

### 为什么Reward计算是正确的？

Reward使用**相对位置**比较：
```python
# 机器人的身体相对于机器人root的位置
body_pos_asset_relative = quat_apply_inverse(root_quat_asset, ...)

# 参考motion的身体相对于参考root的位置
body_pos_motion_relative = quat_apply_inverse(root_quat_motion, ...)

# 比较这两个相对位置
diff = body_pos_motion_relative - body_pos_asset_relative
```

即使两个root朝向不同，只要相对位置一致，diff就是0，reward就是最大值。

### 为什么Observation会有问题？

Observation直接将参考motion转换到机器人坐标系：
```python
ref_pos_b = quat_apply_inverse(robot_root_quat_w, ref_pos_w - robot_root_pos_w)
```

如果两个root朝向相反：
- 参考motion说："前方1米"
- 转换后变成："后方1米"（在机器人坐标系下）
- Policy学到："跟随这个后方1米的目标"
- 结果：机器人后退

---

## 相关文件

- `observations.py` line 1237-1272: 坐标转换代码
- `rewards.py` line 62-89: 正确的相对位置计算
- `command.py`: Motion初始化和reset逻辑
- `0927_twist_teacher_new.yaml` line 119, 140: coordinate_frame配置

---

## 总结

**最快修复**：改配置文件，设置`coordinate_frame: world`

**最好修复**：修改observation代码，使用参考motion自己的root转换

两种方案都需要重新训练才能完全修复问题！
