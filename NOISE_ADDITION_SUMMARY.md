# Actor 观察噪声添加总结

## 修改概述

为 TWIST Teacher Actor 的观察空间添加了噪声支持，参考原始 TWIST 实现的噪声配置，以增强策略的鲁棒性。

---

## 修改的文件

### 1. `observations.py` - 观察函数类修改

**文件路径**: `/home/ubuntu/DATA2/workspace/xmh/HDMI-main/active_adaptation/envs/mdp/commands/twist/observations.py`

#### 修改1: `proprio_history_combined` 类

**位置**: Lines 879-984

**修改内容**:
- 添加 `noise_std` 参数到 `__init__` 方法 (Line 897)
- 在 `compute()` 方法中添加噪声应用逻辑 (Lines 979-982)

```python
def __init__(self, history_length: int = 11, noise_std: float = 0.0, **kwargs):
    self.noise_std = noise_std
    # ...

def compute(self):
    obs = self.history_buffer.view(self.num_envs, -1)

    # 添加噪声（如果设置了噪声标准差）
    if self.noise_std > 0.0:
        noise = torch.randn_like(obs).clamp(-3, 3) * self.noise_std
        obs = obs + noise

    return obs
```

#### 修改2: `ref_motion_windowed` 类

**位置**: Lines 987-1154

**修改内容**:
- 添加 `noise_std` 参数到 `__init__` 方法 (Line 1010)
- 在 `compute()` 方法中添加噪声应用逻辑 (Lines 1149-1152)

```python
def __init__(
    self,
    past_frames: int = 10,
    future_frames: int = 10,
    coordinate_frame: str = 'robot_root',
    noise_std: float = 0.0,
    **kwargs
):
    self.noise_std = noise_std
    # ...

def compute(self):
    obs = self.ref_motion_window_b.view(self.num_envs, -1)

    # 添加噪声（如果设置了噪声标准差）
    if self.noise_std > 0.0:
        noise = torch.randn_like(obs).clamp(-3, 3) * self.noise_std
        obs = obs + noise

    return obs
```

---

### 2. `twist-base-new.yaml` - 配置文件修改

**文件路径**: `/home/ubuntu/DATA2/workspace/xmh/HDMI-main/cfg/task/base/twist-base-new.yaml`

**位置**: Lines 80-130

**修改内容**:
- 为 `policy` 观察空间的两个观察函数添加噪声配置
- `priv` 观察空间保持无噪声（用于 Teacher/Critic）

#### Policy 观察（带噪声）

```yaml
policy:
  # 本体感受历史 (858 dims)
  proprio_history_combined:
    _target_: active_adaptation.envs.mdp.commands.twist.observations.proprio_history_combined
    history_length: 11
    noise_std: 0.03       # ✅ 添加中等噪声

  # 参考运动窗口 (672 dims)
  ref_motion_windowed:
    _target_: active_adaptation.envs.mdp.commands.twist.observations.ref_motion_windowed
    past_frames: 10
    future_frames: 10
    coordinate_frame: robot_root
    noise_std: 0.01       # ✅ 添加较小噪声
```

#### Priv 观察（无噪声）

```yaml
priv:
  # 本体感受历史 (无噪声)
  proprio_history_combined_no_noise:
    _target_: active_adaptation.envs.mdp.commands.twist.observations.proprio_history_combined
    history_length: 11
    noise_std: 0.0        # ✅ 无噪声

  # 参考运动窗口 (无噪声)
  ref_motion_windowed_no_noise:
    _target_: active_adaptation.envs.mdp.commands.twist.observations.ref_motion_windowed
    past_frames: 10
    future_frames: 10
    coordinate_frame: robot_root
    noise_std: 0.0        # ✅ 无噪声

  # 特权信息 (无噪声)
  twist_priv_info:
    _target_: active_adaptation.envs.mdp.commands.twist.observations.priv_info
```

---

## 噪声参数选择依据

### 参考 TWIST 原始实现 (twist-base.yaml)

```yaml
# 原始 TWIST 噪声配置
root_ang_vel_history:       {noise_std: 0.05}   # 根节点角速度
projected_gravity_history:  {noise_std: 0.05}   # 投影重力/姿态
joint_pos_history:          {noise_std: 0.015}  # 关节位置
joint_vel_history:          {noise_std: 0.05}   # 关节速度
```

### 我们的噪声选择

| 观察类型 | 噪声标准差 | 理由 |
|---------|-----------|------|
| **proprio_history_combined** | **0.03** | 包含角速度、姿态、关节位置/速度、动作历史，使用中等噪声 (0.015~0.05之间) |
| **ref_motion_windowed** | **0.01** | 参考运动目标需要保持清晰，使用较小噪声避免模糊目标 |

### 噪声实现细节

- **分布**: 高斯噪声 `torch.randn_like(obs)`
- **裁剪**: 限制在 `[-3σ, 3σ]` 范围内，避免极端值
- **应用时机**: 在 `compute()` 方法返回前应用，确保每次获取观察时都有新的噪声

---

## 与原始 TWIST 实现的对比

### 原始 TWIST

```python
# humanoid_mimic.py:384-389
if self.cfg.noise.add_noise and self.headless:
    obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * \
               min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24), 1.)
```

**特点**:
- 使用 **uniform noise**: `(2 * torch.rand() - 1)` → 范围 `[-1, 1]`
- 有 **noise curriculum**: 噪声随训练步数逐渐增加
- 不同观察分量使用 **不同噪声标准差** (`noise_scale_vec`)

### 我们的实现

```python
# observations.py
if self.noise_std > 0.0:
    noise = torch.randn_like(obs).clamp(-3, 3) * self.noise_std
    obs = obs + noise
```

**特点**:
- 使用 **Gaussian noise**: `torch.randn()` → 标准正态分布
- **固定噪声强度**: 没有curriculum，从训练开始就使用固定噪声
- 整个观察向量使用 **统一噪声标准差**

### 差异说明

| 特征 | 原始 TWIST | 我们的实现 | 影响 |
|------|-----------|-----------|------|
| 噪声分布 | Uniform | Gaussian | Gaussian更符合自然噪声特性 |
| Curriculum | 有（逐渐增加） | 无（固定） | 固定噪声训练更稳定但可能初期收敛慢 |
| 分量独立性 | 不同分量不同噪声 | 统一噪声 | 简化实现，影响较小 |

---

## 使用方法

### 训练 Teacher Policy

```bash
python scripts/train.py \
    algo=ppo_roa_train \
    task=G1/twist/your_task \
    ++task.observation.policy.proprio_history_combined.noise_std=0.03 \
    ++task.observation.policy.ref_motion_windowed.noise_std=0.01
```

### 调整噪声强度

如果需要调整噪声强度，可以通过命令行覆盖配置：

```bash
# 增加噪声（更鲁棒但可能收敛慢）
python scripts/train.py ... \
    ++task.observation.policy.proprio_history_combined.noise_std=0.05 \
    ++task.observation.policy.ref_motion_windowed.noise_std=0.02

# 减少噪声（收敛快但可能不够鲁棒）
python scripts/train.py ... \
    ++task.observation.policy.proprio_history_combined.noise_std=0.01 \
    ++task.observation.policy.ref_motion_windowed.noise_std=0.005

# 完全不用噪声（用于调试）
python scripts/train.py ... \
    ++task.observation.policy.proprio_history_combined.noise_std=0.0 \
    ++task.observation.policy.ref_motion_windowed.noise_std=0.0
```

---

## 预期效果

### 训练时

- **Actor (Policy)**: 看到带噪声的观察，学习更鲁棒的策略
- **Critic (Priv)**: 看到无噪声的观察 + 特权信息，提供准确的价值估计

### 部署时 (Student Policy)

- Student 通过 distillation 学习 Teacher 的策略
- Student 的 adaptation module 学习从噪声观察中提取特权信息
- 最终 Student 能够处理真实世界的噪声传感器数据

---

## 验证检查

训练前建议检查：

1. **观察维度是否正确**:
   ```bash
   # 打印观察空间维度
   python scripts/train.py ... --cfg job
   # 检查 observation_spec 输出
   ```

2. **噪声是否生效**:
   - 在训练日志中查看 policy 和 priv 观察的统计信息
   - Policy 观察应该有更大的方差

3. **性能对比**:
   - 对比有噪声和无噪声训练的收敛曲线
   - 评估最终策略在噪声环境下的表现

---

## 总结

✅ **完成的修改**:
1. `proprio_history_combined` 和 `ref_motion_windowed` 观察类支持噪声参数
2. 配置文件中 Policy 观察添加了合理的噪声值
3. Priv 观察保持无噪声，用于 Teacher/Critic

✅ **噪声配置**:
- Proprio history: `noise_std=0.03` (中等噪声)
- Reference motion: `noise_std=0.01` (较小噪声)
- Priv observations: `noise_std=0.0` (无噪声)

✅ **与 TWIST 一致性**:
- 噪声量级参考 TWIST 原始实现
- Actor-Critic 噪声分离设计一致
- 符合 Teacher-Student 训练范式
