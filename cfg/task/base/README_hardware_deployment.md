# HDMI 硬件部署配置说明

## 概述

本文档说明了如何将 HDMI (Humanoid Dynamic Motion Imitation) 策略从仿真环境迁移到真实硬件。主要修改包括移除仿真器特权信息，使策略仅依赖于真实硬件可获得的观测信息。

## 主要修改内容

### 1. 移除轨迹参考观测 (ref)

**原始配置：**
```yaml
observation:
  command:
    ref_body_pos_future_local: {}
    ref_joint_pos_future: {}
    ref_motion_phase: {}
```

**硬件配置：**
```yaml
observation:
  command:
    pass: {}  # 移除所有参考轨迹观测
```

**说明：** 参考轨迹信息在硬件上不可直接获得，但参考信息仍通过奖励函数间接使用。

### 2. 移除根部线速度观测

**原始配置：**
```yaml
observation:
  priv:
    root_linvel_b: {}  # 根部线速度 (身体坐标系)
```

**硬件配置：**
完全移除 `priv` 观测部分，包括 `root_linvel_b`。

**说明：** 根部线速度在硬件上难以精确测量，策略需要从其他可用观测（如关节状态、IMU数据）推断运动状态。

### 3. 移除特权物理信息

**原始配置：**
```yaml
randomization:
  perturb_body_mass:
    .*: [0.9, 1.1]
  perturb_body_materials:
    static_friction_range: [0.3, 1.6]
    dynamic_friction_range: [0.3, 1.2]
    restitution_range: [0.0, 0.2]
  perturb_body_com:
    com_range: [-0.02, 0.02]
```

**硬件配置：**
```yaml
randomization:
  # 移除所有物理属性随机化
  random_joint_offset:
    .*: [-0.005, 0.005]  # 仅保留关节偏移，模拟硬件误差
```

**说明：** 物体质量、摩擦系数、质心等信息在硬件上是固定的，策略需要适应实际的物理参数。

### 4. 添加时间编码观测

**硬件配置新增：**
```yaml
observation:
  temporal:
    episode_time_normalized: {noise_std: 0.0}  # t/T
    episode_time_sin: {noise_std: 0.0}         # sin(2πt/T)
    episode_time_cos: {noise_std: 0.0}         # cos(2πt/T)
```

**说明：** 时间编码帮助策略理解任务进度和时序依赖关系，其中 T 是回合总长度。

### 5. 转换参考轨迹奖励

**原始配置：**
```yaml
reward:
  tracking:
    tracking_root_pos(keypoint_pos_tracking_product):
      body_names: ["pelvis"]
      weight: 0.5
      sigma: 0.5
```

**硬件配置：**
```yaml
reward:
  tracking:
    tracking_root_pos_relative(keypoint_pos_tracking_relative_product):
      body_names: ["pelvis"]
      weight: 0.5
      sigma: 0.5
  object_tracking:
    object_pos_tracking_relative: {weight: 1.0, sigma: 0.5}
    object_ori_tracking_relative: {weight: 1.0, sigma: 0.5}
```

**说明：** 使用相对于物体/目标的位置而非绝对位置，避免依赖全局坐标系信息。

## 关键设计原则

### 1. 仅使用硬件可获得的观测
- **保留：** 关节位置/速度、IMU数据（角速度、重力方向）、相对物体位置
- **移除：** 绝对位置、线速度、物理参数、参考轨迹

### 2. 相对坐标系转换
- 所有空间信息使用相对坐标系（身体坐标系或局部坐标系）
- 避免依赖全局坐标系或仿真器坐标系

### 3. 时间感知
- 添加时间编码使策略具备时序意识
- 帮助策略理解任务阶段和预期行为

### 4. 鲁棒性设计
- 减少随机化以匹配硬件的确定性特征
- 保留必要的噪声建模硬件传感器误差

## 使用方法

1. **训练阶段：** 使用原始 `hdmi-base.yaml` 进行 Stage 1 和 Stage 2 训练
2. **重训练阶段：** 使用 `hdmi-hardware.yaml` 对策略进行微调，适应硬件约束
3. **部署阶段：** 直接使用硬件配置在真实机器人上运行

## 注意事项

1. **观测维度变化：** 硬件配置的观测空间与仿真配置不同，需要重新训练或适配网络结构
2. **奖励函数调整：** 相对位置跟踪可能需要调整权重和参数
3. **传感器校准：** 确保硬件传感器（IMU、编码器等）正确校准
4. **安全机制：** 在硬件部署前充分测试，确保安全性

## 扩展建议

1. **自适应机制：** 可以添加在线适应模块来处理硬件特性差异
2. **传感器融合：** 利用多传感器信息提高状态估计精度
3. **渐进部署：** 从简单任务开始，逐步增加复杂度


