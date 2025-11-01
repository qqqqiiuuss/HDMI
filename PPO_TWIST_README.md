# PPO_TWIST - TWIST-Aligned PPO for HDMI

## 概述

`ppo_twist.py` 是一个完全对齐 TWIST-MASTER teacher 配置的 PPO 实现，专为 HDMI 框架设计。

**关键特点**:
- ✅ 完全对齐 TWIST-MASTER 的训练超参数
- ✅ 不使用 Motion Encoder（避免信息损失）
- ✅ 直接 MLP 处理完整观察空间
- ✅ 支持 HDMI 的观察格式（包含历史）

## 为什么需要这个实现？

### 问题：ppo_motion_encoder 效果差

`ppo_motion_encoder` 在训练时出现以下问题：
1. **Penalty 值过大** - 跟踪精度下降
2. **效果不如普通 PPO** - 整体性能低于预期
3. **与 TWIST-MASTER 不一致** - 难以复现论文结果

### 根本原因分析

#### 1. Motion Encoder 的信息损失

```
Motion部分: 672 dims (21×32)
  ↓
Conv1D Encoder
  ↓
Motion Latent: 128 dims  ← 压缩率 5.25x，损失 81% 维度
```

**问题**:
- 细粒度的运动细节丢失
- 关节协调关系被破坏
- 高频运动信息无法编码

#### 2. 采样策略不匹配

**TWIST 原版**:
```python
tar_obs_steps = [1, 5, 10, 15, 20, ...]  # 稀疏采样，间隔5帧
覆盖时长: 1.9秒
信息密度: 高（每帧都是关键姿态）
```

**HDMI ppo_motion_encoder**:
```python
future_steps = [1, 2, 3, ..., 10]  # 密集采样，间隔1帧
覆盖时长: 0.42秒
信息密度: 低（相邻帧高度冗余）
```

**Conv1D 感受野分析**:
```
TWIST: 感受野 14帧 × 5步间隔 = 70步 = 1.4秒运动信息
HDMI:  感受野 14帧 × 1步间隔 = 14步 = 0.28秒运动信息
差距: 5倍
```

**结论**: Conv1D 设计用于**稀疏关键帧**，不适合**密集连续帧**。

#### 3. 当前帧提取错误

**TWIST 原版**:
```python
# 提取未来第1帧（t+1时刻的目标）
obs[:, :num_single_motion_observations]
```

**HDMI ppo_motion_encoder**:
```python
# 提取中间帧（t时刻的当前状态）
current_frame_idx = motion_tsteps // 2  # 第11帧
```

**问题**: 网络需要"下一步该去哪"，不是"现在在哪"。

## PPO_TWIST 的解决方案

### 核心设计理念

**放弃 Motion Encoder，让网络自己学习**

```python
# 普通 PPO (效果好)
Input: [proprio_history(42240) + ref_motion(672)] = 42912 dims
  ↓
MLP [512, 512, 256, 128]  ← 网络自由学习如何处理
  ↓
Action

# PPO_TWIST (与普通PPO一致，但用TWIST超参数)
Input: 全部观察 (42912 dims)
  ↓
MLP [512, 512, 256, 128]  ← TWIST架构
  ↓
Action
```

**优势**:
- ✅ 无信息瓶颈
- ✅ 网络自主学习最优特征
- ✅ 适配任意采样策略
- ✅ 与 TWIST 超参数对齐

## 配置对照表

| 参数 | TWIST-MASTER | ppo_twist | ppo_motion_encoder | 说明 |
|------|--------------|-----------|-------------------|------|
| `train_every` | 24 | ✅ 24 | 24 | 每次更新的环境步数 |
| `ppo_epochs` | 5 | ✅ 5 | 5 | PPO 迭代轮数 |
| `num_minibatches` | 4 | ✅ 4 | 4 | Minibatch 数量 |
| `lr` | 2e-4 | ✅ 2e-4 | 2e-4 | 学习率 |
| `entropy_coef` | 0.01 | ✅ 0.01 | 0.01 | 熵系数 |
| `desired_kl` | 0.008 | ✅ 0.008 | 0.008 | 自适应学习率 |
| `actor_hidden_dims` | [512, 512, 256, 128] | ✅ [512, 512, 256, 128] | [512, 512, 256, 128] | Actor 网络结构 |
| `critic_hidden_dims` | [512, 512, 256, 128] | ✅ [512, 512, 256, 128] | [512, 512, 256, 128] | Critic 网络结构 |
| `layer_norm` | True | ✅ True | True | Layer Normalization |
| `activation` | 'silu' | ✅ 'silu' | 'elu' | 激活函数 |
| `init_noise_std` | 1.0 | ✅ 1.0 | 1.5 | 初始噪声 |
| `gamma` | 0.99 | ✅ 0.99 | 0.99 | 折扣因子 |
| `lmbda` | 0.95 | ✅ 0.95 | 0.95 | GAE lambda |
| **Motion Encoder** | ❌ 无 | ✅ 无 | ⚠️ 有（问题源头） | - |

## 使用方法

### 训练命令

```bash
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096 \
    suffix=test_ppo_twist
```

### 对比实验

```bash
# 方案1: ppo_twist (推荐)
python scripts/train.py algo=ppo_twist task=G1/twist/0927_twist_teacher_new suffix=ppo_twist

# 方案2: ppo_motion_encoder (问题版本)
python scripts/train.py algo=ppo_motion_encoder task=G1/twist/0927_twist_teacher_new suffix=ppo_encoder

# 方案3: 普通 ppo (HDMI默认)
python scripts/train.py algo=ppo task=G1/twist/0927_twist_teacher_new suffix=ppo_default
```

## 预期效果

### ppo_twist 应该达到的效果

✅ **Tracking Rewards** (应该较高):
- `tracking_keybody_pos`: > 0.5
- `tracking_joint_dof`: > 0.4
- `tracking_root_pose`: > 0.4
- `tracking_root_vel`: > 0.6

✅ **Penalty 较小**:
- `feet_slip`: < -0.05
- `dof_acc`: < -0.001
- `action_rate`: < -0.01

✅ **整体表现**:
- 总奖励稳定增长
- Critic explained variance > 0.7
- 机器人能精确跟踪参考运动

### ppo_motion_encoder 的问题表现

❌ **Tracking Rewards 低**:
- 跟踪精度明显下降
- 关键点位置误差大

❌ **Penalty 大**:
- `feet_slip` 惩罚增加
- `dof_acc/vel` 惩罚增加
- 整体运动不平滑

❌ **训练不稳定**:
- 奖励曲线波动大
- Explained variance 低

## Bug 修复记录

除了创建 `ppo_twist.py`，还修复了以下 Bug:

### 1. future_steps 配置错误
```yaml
# 之前 (错误): 只有 9 个未来帧
future_steps: [1, 2,3,4,5,6,7,8,9]

# 修复后: 10 个未来帧
future_steps: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### 2. future_steps dtype 错误
```python
# 之前 (错误): 没有指定 dtype
self.future_steps = torch.tensor(future_steps)

# 修复后: 明确指定 long 类型
self.future_steps = torch.tensor(future_steps, dtype=torch.long)
```

**文件**: `/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/active_adaptation/envs/mdp/commands/twist/command.py:142`

## 技术细节

### 网络架构

```python
Actor:
  Input: observation (42912 dims)
    ↓
  Linear(42912, 512) + LayerNorm + SiLU
    ↓
  Linear(512, 512) + SiLU
    ↓
  Linear(512, 256) + LayerNorm + SiLU  ← TWIST layer_norm 位置
    ↓
  Linear(256, 128) + SiLU
    ↓
  Output: action_mean, action_std

Critic:
  Input: [observation + priv_obs] (42912 + priv dims)
    ↓
  Linear(input, 512) + LayerNorm + SiLU
    ↓
  Linear(512, 512) + SiLU
    ↓
  Linear(512, 256) + LayerNorm + SiLU
    ↓
  Linear(256, 128) + SiLU
    ↓
  Linear(128, 1)
    ↓
  Output: state_value
```

### 自适应学习率

```python
if kl > desired_kl * 2.0:
    lr = max(1e-5, lr / 1.5)  # KL太大，降低学习率
elif kl < desired_kl / 2.0 and kl > 0.0:
    lr = min(1e-2, lr * 1.5)  # KL太小，提高学习率
```

### 不裁剪负奖励

```python
# TWIST 不裁剪负奖励（与 HDMI 默认 ppo 不同）
rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True)  # 保留负值
```

## 文件结构

```
active_adaptation/learning/ppo/
├── ppo.py                    # HDMI 默认 PPO
├── ppo_motion_encoder.py     # 有问题的版本
├── ppo_twist.py              # ✅ 新创建：TWIST-对齐版本
├── ppo_roa.py               # Teacher-Student 架构
└── common.py                # 公共模块
```

## 后续建议

1. **优先使用 ppo_twist** - 应该能达到与 TWIST-MASTER 类似的效果
2. **如果效果仍不理想** - 检查观察空间配置和奖励函数
3. **对比实验** - 同时运行 ppo_twist 和 ppo_motion_encoder，对比 WandB 曲线
4. **稀疏采样实验** - 可以尝试修改 `future_steps` 为 TWIST 的稀疏采样

## 参考

- TWIST-MASTER 配置: `/home/ubuntu/DATA2/workspace/xmh/TWIST-master/legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`
- TWIST 网络架构: `/home/ubuntu/DATA2/workspace/xmh/TWIST-master/rsl_rl/rsl_rl/modules/actor_critic_mimic.py`
- HDMI base PPO: `/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/active_adaptation/learning/ppo/ppo.py`
