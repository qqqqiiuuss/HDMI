# TWIST to HDMI 转换和训练完整指南

本指南详细说明如何将 TWIST 的运动数据转换为 HDMI 格式，并使用转换后的数据在 HDMI 框架中训练运动跟踪策略。

## 目录
1. [数据转换](#1-数据转换)
2. [验证转换结果](#2-验证转换结果)
3. [配置任务](#3-配置任务)
4. [训练策略](#4-训练策略)
5. [评估策略](#5-评估策略)
6. [常见问题](#6-常见问题)

---

## 1. 数据转换

### 1.1 准备工作

确保你已经安装了必要的依赖：
```bash
conda activate hdmi
pip install poselib  # TWIST 的 poselib 依赖
```

### 1.2 使用转换脚本

有两种转换方式：

**方式 A：从 YAML 配置批量转换（推荐）**

这是最简单的方式，可以一次性转换 TWIST 数据集中的所有运动：

```bash
# 使用提供的示例脚本
bash convert_twist_example.sh

# 或者手动运行转换命令
python convert_twist_to_hdmi.py \
    --yaml_config /home/ubuntu/DATA2/workspace/xmh/TWIST-master/legged_gym/motion_data_configs/twist_dataset.yaml \
    --output data/motion/g1/twist_converted \
    --target_fps 50
```

**方式 B：单个文件转换**

如果你只想转换特定的运动文件：

```bash
python convert_twist_to_hdmi.py \
    --input /path/to/twist/motion/mocap_0.npy \
    --output data/motion/g1/twist_converted/mocap_0 \
    --target_fps 50
```

### 1.3 转换参数说明

- `--yaml_config`: TWIST 的 YAML 配置文件路径（批量转换）
- `--input`: 单个 TWIST motion .npy 文件路径（单文件转换）
- `--output`: 输出目录（HDMI 格式）
- `--target_fps`: 目标帧率（默认 50 FPS，HDMI 标准）
- `--robot_name`: 机器人名称（默认 "unitree_g1"）

### 1.4 转换过程说明

转换脚本会执行以下操作：

1. **加载 TWIST 数据**: 从 .npy 文件加载 `SkeletonMotion` 对象
2. **提取关键数据**:
   - 全局身体位置 (`global_translation`)
   - 全局身体旋转 (`global_rotation`)
   - 局部关节旋转 (`local_rotation`)
   - 线速度 (`linear_velocity`)
   - 关节速度 (`dof_vels`)

3. **坐标转换和计算**:
   - 将 TWIST 的全局旋转转换为 HDMI 的世界坐标系四元数
   - 从四元数计算角速度
   - 从局部旋转转换为关节角度 (DOF positions)

4. **FPS 重采样**: 如果 TWIST 数据的 FPS 与目标 FPS 不同，使用 scipy 插值进行重采样

5. **保存为 HDMI 格式**:
   - `motion.npz`: 包含所有运动数据
   - `meta.json`: 包含元数据（身体名称、关节名称、FPS）

---

## 2. 验证转换结果

### 2.1 检查输出目录结构

转换后的数据应该有以下结构：

```
data/motion/g1/twist_converted/
├── mocap_0/
│   ├── motion.npz
│   └── meta.json
├── mocap_1/
│   ├── motion.npz
│   └── meta.json
├── ...
└── mocap_N/
    ├── motion.npz
    └── meta.json
```

### 2.2 验证数据完整性

使用 Python 脚本验证数据：

```python
import numpy as np
import json

# 加载转换后的数据
motion_data = np.load("data/motion/g1/twist_converted/mocap_0/motion.npz")
with open("data/motion/g1/twist_converted/mocap_0/meta.json", "r") as f:
    meta = json.load(f)

# 检查数据形状
print("数据检查:")
print(f"帧数: {motion_data['body_pos_w'].shape[0]}")
print(f"身体数量: {motion_data['body_pos_w'].shape[1]}")
print(f"关节数量: {motion_data['joint_pos'].shape[1]}")
print(f"FPS: {meta['fps']}")

# 检查必需的字段
required_keys = [
    'body_pos_w', 'body_quat_w', 'joint_pos',
    'body_lin_vel_w', 'body_ang_vel_w', 'joint_vel'
]
for key in required_keys:
    assert key in motion_data, f"缺少必需字段: {key}"
    print(f"✓ {key}: {motion_data[key].shape}")

print("\n元数据:")
print(f"身体名称: {len(meta['body_names'])} 个")
print(f"关节名称: {len(meta['joint_names'])} 个")
```

### 2.3 检查转换报告

转换脚本会在最后输出转换摘要：

```
转换摘要:
成功转换: 150 个
失败: 0 个
总耗时: 45.2 秒
```

如果有失败的转换，请检查错误日志并确保：
- TWIST 数据文件路径正确
- 文件格式正确（.npy 或 .pkl）
- 文件包含有效的 `SkeletonMotion` 对象

---

## 3. 配置任务

### 3.1 使用模板配置文件

我已经为你创建了一个模板配置文件：`cfg/task/G1/hdmi/twist_motion_tracking.yaml`

### 3.2 根据你的任务类型修改配置

#### 情况 A：纯运动跟踪（无物体交互）

如果你的 TWIST 数据是纯运动跟踪（例如步态、舞蹈等），**不需要操作物体**：

1. **修改数据路径**（第 48 行）:
   ```yaml
   command:
     data_path: data/motion/g1/twist_converted  # 修改为你的转换数据路径
   ```

2. **保持命令类型为 `RobotTracking`**（第 46 行）:
   ```yaml
   command:
     _target_: active_adaptation.envs.mdp.commands.hdmi.command.RobotTracking
   ```

3. **删除物体相关的配置**:
   - 删除 `object_asset_name`, `object_body_name`
   - 删除 `contact_target_pos_offset`, `contact_eef_pos_offset`
   - 删除 observation 中的 `object` 和 `depth` 部分
   - 删除 reward 中的 `object_tracking` 组
   - 删除 termination 中的物体相关条件
   - 删除 randomization 中的物体相关设置

4. **使用简化的机器人类型**（第 25 行）:
   ```yaml
   robot:
     robot_type: g1_29dof_rubberhand
   ```

#### 情况 B：运动跟踪 + 物体交互

如果你的 TWIST 数据包含物体交互（例如推箱子、开门等）：

1. **修改数据路径**（同上）

2. **使用 `RobotObjectTracking` 命令类型**:
   ```yaml
   command:
     _target_: active_adaptation.envs.mdp.commands.hdmi.command.RobotObjectTracking
   ```

3. **配置物体信息**:
   ```yaml
   command:
     object_asset_name: "suitcase"  # 修改为你的物体类型
     object_body_name: "suitcase"   # 修改为你的物体名称

     # 根据你的任务调整接触位置偏移
     contact_target_pos_offset:
       - [-0.1, 0.18, 0.25]  # 左手接触点
       - [-0.1,-0.18, 0.25]  # 右手接触点
   ```

4. **使用完整的机器人类型**:
   ```yaml
   robot:
     robot_type: g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule
   ```

5. **保留所有物体相关的观察、奖励、终止条件配置**

### 3.3 调整其他参数

根据你的运动数据特点，可能需要调整：

- **回合长度** (`max_episode_length`): 根据你的运动序列长度设置
- **奖励权重**: 在 `reward` 部分调整各项奖励的 `weight`
- **终止条件阈值**: 在 `termination` 部分调整 `threshold`
- **随机化范围**: 在 `randomization` 部分调整域随机化参数

---

## 4. 训练策略

### 4.1 阶段一：训练 Teacher 策略

Teacher 策略使用特权信息（privileged information）进行训练：

```bash
python scripts/train.py \
    algo=ppo_roa_train \
    task=G1/hdmi/twist_motion_tracking \
    wandb.project=hdmi_twist \
    wandb.name=twist_teacher_001
```

**重要参数说明**:
- `algo=ppo_roa_train`: 使用 PPO-ROA 训练算法（Teacher-Student 架构）
- `task=G1/hdmi/twist_motion_tracking`: 你的任务配置文件名
- `wandb.project`: WandB 项目名称
- `wandb.name`: WandB 运行名称（用于跟踪实验）

**可选参数**:
```bash
python scripts/train.py \
    algo=ppo_roa_train \
    task=G1/hdmi/twist_motion_tracking \
    total_frames=200_000_000 \        # 总训练帧数（默认值）
    num_envs=4096 \                    # 并行环境数（默认 4096）
    algo.lr=3e-4 \                     # 学习率
    wandb.mode=online                  # WandB 模式（online/offline/disabled）
```

### 4.2 监控训练进度

训练过程中，可以通过以下方式监控：

1. **WandB Dashboard**:
   - 访问 https://wandb.ai/<你的用户名>/hdmi_twist
   - 查看奖励曲线、成功率等指标

2. **关键指标**:
   - `reward/total`: 总奖励
   - `reward/tracking/*`: 各项跟踪奖励
   - `episode_length_mean`: 平均回合长度
   - `metrics/*`: 跟踪误差指标

3. **终端输出**:
   ```
   Epoch 100/1000 | Reward: 45.2 | Episode Length: 450 | FPS: 12000
   ```

### 4.3 阶段二：Fine-tune Student 策略

当 Teacher 策略训练良好后（通常 reward 收敛），fine-tune Student 策略用于部署：

```bash
python scripts/train.py \
    algo=ppo_roa_finetune \
    task=G1/hdmi/twist_motion_tracking \
    checkpoint_path=run:hdmi_twist/<teacher_run_id> \
    wandb.project=hdmi_twist \
    wandb.name=twist_student_001
```

**重要参数说明**:
- `algo=ppo_roa_finetune`: 使用 Student fine-tuning 算法
- `checkpoint_path=run:<project>/<run_id>`: Teacher 策略的 WandB 路径
  - 从 WandB 找到你的 Teacher 运行 ID（例如 `abc123xyz`）
  - 路径格式: `run:hdmi_twist/abc123xyz`

**如何找到 Teacher Run ID**:
1. 登录 WandB: https://wandb.ai
2. 进入项目: `hdmi_twist`
3. 找到 Teacher 训练运行
4. 在 URL 中复制运行 ID（最后一段）

### 4.4 使用普通 PPO 训练（可选）

如果你不需要 Teacher-Student 架构，可以直接使用普通 PPO：

```bash
python scripts/train.py \
    algo=ppo \
    task=G1/hdmi/twist_motion_tracking \
    wandb.project=hdmi_twist \
    wandb.name=twist_ppo_001
```

**注意**: 普通 PPO 不使用特权信息，可能训练效果不如 PPO-ROA，但训练更简单。

---

## 5. 评估策略

### 5.1 评估训练好的策略

```bash
python scripts/play.py \
    algo=ppo_roa_finetune \
    task=G1/hdmi/twist_motion_tracking \
    checkpoint_path=run:hdmi_twist/<student_run_id> \
    num_envs=16
```

**参数说明**:
- `checkpoint_path`: Student 策略的 WandB 路径
- `num_envs`: 评估环境数（建议设置较小值以便观察）

### 5.2 可视化评估

评估时会自动打开 IsaacLab 可视化窗口，你可以看到：
- 机器人执行运动跟踪
- 参考运动的可视化（如果启用）
- 跟踪误差可视化

### 5.3 查看评估指标

评估结束后会输出详细指标：

```
评估结果:
平均奖励: 48.5
平均回合长度: 485
成功率: 92.3%

跟踪误差:
- 身体位置误差: 0.05m
- 身体朝向误差: 0.12 rad
- 关节位置误差: 0.08 rad
```

---

## 6. 常见问题

### Q1: 转换时出错: "No module named 'poselib'"

**解决方案**:
```bash
conda activate hdmi
pip install poselib
```

### Q2: 训练时找不到运动数据

**可能原因**:
- 数据路径配置错误
- NPZ 文件命名不正确（必须是 `motion.npz`）
- `meta.json` 文件缺失

**检查方法**:
```bash
# 检查数据目录结构
ls -R data/motion/g1/twist_converted/

# 应该看到：
# mocap_0/motion.npz
# mocap_0/meta.json
# mocap_1/motion.npz
# mocap_1/meta.json
# ...
```

### Q3: 训练时 reward 不收敛

**可能原因和解决方案**:

1. **运动数据质量问题**:
   - 检查转换后的数据是否有 NaN 或异常值
   - 验证 FPS 是否正确（应为 50）
   - 检查关节名称和身体名称是否匹配

2. **奖励权重不合适**:
   - 尝试调整 `reward` 配置中的 `weight` 和 `sigma` 参数
   - 从较大的 `sigma` 开始，逐步减小

3. **学习率太高或太低**:
   ```bash
   python scripts/train.py ... algo.lr=1e-4  # 尝试更小的学习率
   ```

4. **环境数量不足**:
   ```bash
   python scripts/train.py ... num_envs=8192  # 增加并行环境数
   ```

### Q4: 转换后的数据帧数不对

**原因**: TWIST 数据的 FPS 与目标 FPS 不同，脚本会自动重采样。

**验证**:
```python
import numpy as np
import json

data = np.load("data/motion/g1/twist_converted/mocap_0/motion.npz")
meta = json.load(open("data/motion/g1/twist_converted/mocap_0/meta.json"))

original_frames = len(data['body_pos_w'])
fps = meta['fps']
duration = original_frames / fps

print(f"帧数: {original_frames}, FPS: {fps}, 时长: {duration:.2f}秒")
```

### Q5: 关节名称或身体名称不匹配

**问题**: TWIST 数据的关节/身体名称与 HDMI 任务配置中的名称不一致。

**解决方案**:

1. 查看转换后的 `meta.json` 文件:
   ```bash
   cat data/motion/g1/twist_converted/mocap_0/meta.json
   ```

2. 修改任务配置文件中的名称:
   ```yaml
   command:
     tracking_keypoint_names: [
       "pelvis",
       "left_hip_link",
       # ... 根据 meta.json 中的实际名称修改
     ]
   ```

### Q6: Teacher 训练完成，如何找到 checkpoint 路径？

**步骤**:

1. 登录 WandB: https://wandb.ai
2. 进入你的项目（例如 `hdmi_twist`）
3. 找到 Teacher 训练的运行
4. 复制 URL 中的运行 ID（最后一段）
5. 使用格式: `run:<project>/<run_id>`

**示例**:
- WandB URL: `https://wandb.ai/username/hdmi_twist/runs/abc123xyz`
- Checkpoint 路径: `run:hdmi_twist/abc123xyz`

### Q7: 训练时显存不足

**解决方案**:

1. **减少并行环境数**:
   ```bash
   python scripts/train.py ... num_envs=2048  # 从 4096 减少到 2048
   ```

2. **减少历史步数**:
   修改任务配置中的 `history_steps`:
   ```yaml
   observation:
     policy:
       joint_pos_history: {history_steps: [0, 1, 2, 3, 4], ...}  # 减少步数
   ```

3. **使用更小的模型**:
   修改算法配置（需要修改代码中的 `ppo_roa.py`）

### Q8: 如何批量训练多个运动序列？

**方法 1: 转换所有数据到一个目录**

```bash
# 转换所有 TWIST 数据
python convert_twist_to_hdmi.py \
    --yaml_config /path/to/twist_dataset.yaml \
    --output data/motion/g1/twist_all \
    --target_fps 50

# 训练时使用整个目录
# 在任务配置中设置: data_path: data/motion/g1/twist_all
```

**方法 2: 使用正则表达式选择特定序列**

在任务配置中使用正则表达式：
```yaml
command:
  data_path: data/motion/g1/twist_converted/(mocap_0|mocap_1|mocap_5)
```

**方法 3: 分别训练每个序列**

创建多个任务配置文件，分别训练：
```bash
# 训练 mocap_0
python scripts/train.py task=G1/hdmi/twist_mocap_0 ...

# 训练 mocap_1
python scripts/train.py task=G1/hdmi/twist_mocap_1 ...
```

---

## 完整工作流程总结

1. **转换数据**:
   ```bash
   bash convert_twist_example.sh
   ```

2. **验证数据**:
   ```bash
   python -c "
   import numpy as np
   import json
   data = np.load('data/motion/g1/twist_converted/mocap_0/motion.npz')
   meta = json.load(open('data/motion/g1/twist_converted/mocap_0/meta.json'))
   print('Frames:', data['body_pos_w'].shape[0], 'FPS:', meta['fps'])
   "
   ```

3. **配置任务**:
   - 修改 `cfg/task/G1/hdmi/twist_motion_tracking.yaml`
   - 设置正确的 `data_path`
   - 根据任务类型调整配置

4. **训练 Teacher**:
   ```bash
   python scripts/train.py \
       algo=ppo_roa_train \
       task=G1/hdmi/twist_motion_tracking \
       wandb.project=hdmi_twist \
       wandb.name=teacher_001
   ```

5. **Fine-tune Student**:
   ```bash
   # 从 WandB 获取 teacher_run_id
   python scripts/train.py \
       algo=ppo_roa_finetune \
       task=G1/hdmi/twist_motion_tracking \
       checkpoint_path=run:hdmi_twist/<teacher_run_id> \
       wandb.name=student_001
   ```

6. **评估**:
   ```bash
   python scripts/play.py \
       algo=ppo_roa_finetune \
       task=G1/hdmi/twist_motion_tracking \
       checkpoint_path=run:hdmi_twist/<student_run_id> \
       num_envs=16
   ```

---

## 更多资源

- HDMI 项目文档: `CLAUDE.md`
- PPO-ROA 架构分析: `PPO_ROA_Architecture_Analysis.md`
- 坐标系说明: `coordinate_system_explanation.md`
- WandB 文档: https://docs.wandb.ai

如果遇到其他问题，请查看 HDMI 项目的 README 或提交 Issue。
