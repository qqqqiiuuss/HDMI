# GMR PKL到HDMI NPZ格式转换指南

## 概述

本文档解释了如何将GMR (General Motion Retargeting) 项目中的PKL格式运动数据转换为HDMI项目使用的NPZ格式，以便在LocoMujoco环境中进行可视化。

## 文件格式对比

### GMR PKL格式

GMR的`vis_robot_motion.py`脚本使用的PKL文件包含以下数据：

```python
motion_data = {
    "fps": 50.0,                    # 帧率
    "root_pos": [T, 3],            # 根部位置 (世界坐标系)
    "root_rot": [T, 4],            # 根部旋转四元数 (xyzw格式)
    "dof_pos": [T, N_joints],      # 关节位置
    "local_body_pos": [T, N_bodies, 3],  # 局部身体位置
    "link_body_list": [...],       # 身体链接列表
}
```

### HDMI NPZ格式

HDMI项目使用的NPZ格式包含：

```python
# motion.npz文件
{
    "body_pos_w": [T, N_bodies, 3],     # 世界坐标系下的身体位置
    "body_quat_w": [T, N_bodies, 4],    # 世界坐标系下的身体四元数 (wxyz格式)
    "joint_pos": [T, N_joints],         # 关节位置
    "body_lin_vel_w": [T, N_bodies, 3], # 身体线速度
    "body_ang_vel_w": [T, N_bodies, 3], # 身体角速度
    "joint_vel": [T, N_joints],         # 关节速度
    "object_contact": [T, 1]            # 物体接触信息 (可选)
}

# meta.json文件
{
    "body_names": ["pelvis", "left_hip_pitch_link", ...],  # 身体名称列表
    "joint_names": ["left_hip_pitch_joint", ...],          # 关节名称列表
    "fps": 50.0                                            # 帧率
}
```

## 使用转换脚本

### 1. 基本用法

```bash
python convert_pkl_to_npz.py \
    --pkl_path path/to/robot_motion.pkl \
    --output_dir path/to/output/directory \
    --robot_name unitree_g1
```

### 2. 参数说明

- `--pkl_path`: 输入的PKL文件路径
- `--output_dir`: 输出目录，将生成`motion.npz`和`meta.json`文件
- `--robot_name`: 机器人名称，用于生成正确的关节和身体名称 (默认: unitree_g1)

### 3. 示例

```bash
# 转换G1机器人的运动数据
python convert_pkl_to_npz.py \
    --pkl_path GMR/output/robot_motion.pkl \
    --output_dir data/motion/g1/converted_motion \
    --robot_name unitree_g1
```

## 在LocoMujoco中可视化

转换完成后，可以使用HDMI项目的可视化工具：

### 1. 使用motion_data_publisher.py

```bash
cd scripts/vis
python motion_data_publisher.py --data_file ../../data/motion/g1/converted_motion/motion.npz
```

### 2. 使用MotionDataset加载

```python
from active_adaptation.utils.motion import MotionDataset

# 加载转换后的数据
dataset = MotionDataset.create_from_path("data/motion/g1/converted_motion", target_fps=50)
motion_data = dataset.data

print(f"运动数据形状: {motion_data.shape}")
print(f"关节名称: {dataset.joint_names}")
print(f"身体名称: {dataset.body_names}")
```

## 转换过程说明

转换脚本执行以下步骤：

1. **加载PKL数据**: 读取GMR格式的运动数据
2. **坐标系转换**: 将局部身体位置转换到世界坐标系
3. **四元数格式转换**: 从xyzw格式转换为wxyz格式
4. **速度计算**: 使用数值微分计算线速度和角速度
5. **元数据生成**: 创建包含关节和身体名称的meta.json文件

## 注意事项

1. **四元数格式**: GMR使用xyzw格式，HDMI使用wxyz格式
2. **坐标系**: 转换过程假设local_body_pos是相对于根部的位置
3. **速度计算**: 使用中心差分法计算速度，可能引入轻微的数值误差
4. **身体旋转**: 简化假设所有身体部位具有相同的根部旋转

## 故障排除

### 常见问题

1. **维度不匹配**: 检查PKL文件中的数据形状是否符合预期
2. **关节名称错误**: 根据实际机器人模型调整joint_names和body_names
3. **速度异常**: 检查原始数据的时间步长和帧率设置

### 调试技巧

```python
# 检查转换后的数据
import numpy as np
data = np.load("output/motion.npz")
print("数据键:", list(data.keys()))
for key in data.keys():
    print(f"{key}: {data[key].shape}")
```

## 扩展支持

要支持其他机器人类型，需要：

1. 在转换脚本中添加对应的关节和身体名称
2. 根据机器人的运动学结构调整坐标转换逻辑
3. 验证转换结果的正确性

## 相关文件

- `convert_pkl_to_npz.py`: 主要转换脚本
- `GMR/scripts/vis_robot_motion.py`: GMR可视化脚本
- `scripts/vis/motion_data_publisher.py`: HDMI可视化脚本
- `active_adaptation/utils/motion.py`: HDMI运动数据处理模块







