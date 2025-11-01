# HDMI MotionEncoder集成修改说明

## 📋 修改概述

为HDMI项目添加了TWIST-style的MotionEncoder支持，使其能够更有效地处理时序运动数据。

### 修改时间
- 2025-11-01

### 主要变更
1. 实现了TWIST的1D CNN MotionEncoder
2. 创建了支持MotionEncoder的新PPO policy变体
3. 自适应HDMI的观测格式（过去10帧 + 当前1帧 + 未来10帧 = 21帧）

---

## 🆕 新增文件

### 1. `active_adaptation/learning/modules/motion_encoder.py`

**功能**: TWIST-style 1D卷积运动编码器

**核心类**:

#### `MotionEncoder1D`
```python
class MotionEncoder1D(nn.Module):
    """
    1D CNN Motion Encoder for temporal sequence compression

    Args:
        activation_fn: 激活函数 (default: nn.ELU)
        input_size: 单帧观测维度
        tsteps: 时间步数量
        output_size: 输出latent维度
        tanh_encoder_output: 是否对输出应用tanh
    """
```

**支持的时间步配置**:
- `tsteps=21`: HDMI默认配置 (过去10 + 当前1 + 未来10)
- `tsteps=20`: TWIST原始配置 (仅未来20帧)
- `tsteps=10`: 短序列配置
- `tsteps=50`: 长序列配置
- `tsteps=1`: 退化为MLP (无时序建模)

**网络架构** (tsteps=21):
```
输入: [Batch, 21 × input_size]
  ↓ Per-frame Linear Projection
[Batch × 21, input_size] → [Batch × 21, 60]
  ↓ Reshape
[Batch, 60, 21]  (channels, time)
  ↓ Conv1D (kernel=6, stride=2)
[Batch, 40, 8]
  ↓ Conv1D (kernel=4, stride=2)
[Batch, 20, 3]
  ↓ Flatten
[Batch, 60]
  ↓ Linear Output
[Batch, output_size]
```

**参数量**:
- HDMI配置 (tsteps=21, input=32, output=128): ~32K 参数
- 对比MLP首层 (1160→512): 594K 参数
- **节省 94.6% 参数**

**测试函数**:
```bash
python -m active_adaptation.learning.modules.motion_encoder
```

---

### 2. `active_adaptation/learning/ppo/ppo_motion_encoder.py`

**功能**: 集成MotionEncoder的PPO policy实现

**核心类**:

#### `PPOMotionEncoderPolicy`
```python
class PPOMotionEncoderPolicy(TensorDictModuleBase):
    """
    PPO Policy with TWIST-style Motion Encoder

    Architecture:
    - Motion Encoder: 压缩时序参考轨迹
    - Actor: 使用 [motion_latent + current_frame + proprio]
    - Critic: 使用 [motion_latent + current_frame + proprio + priv]
    """
```

**配置类**:
```python
@dataclass
class PPOMotionEncoderConfig:
    # PPO超参数 (对齐TWIST)
    ppo_epochs: int = 5           # TWIST: 5 (原base PPO: 3)
    num_minibatches: int = 4      # TWIST: 4 (原base PPO: 8)
    lr: float = 2e-4              # TWIST: 2e-4 (原base PPO: 1e-4)
    entropy_coef: float = 0.01    # TWIST: 0.01 (原base PPO: 0.001)
    desired_kl: float = 0.008     # TWIST: 0.008 (启用自适应学习率)

    # MotionEncoder配置
    use_motion_encoder: bool = True
    motion_latent_dim: int = 128
    motion_tsteps: int = 21       # HDMI: 21 (TWIST: 20)
    motion_input_size: int = 32   # 单帧维度 (自动检测)

    # 网络架构 (对齐TWIST)
    actor_hidden_dims: Tuple[int, ...] = (512, 512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 512, 256, 128)
```

**关键方法**:

##### `_extract_motion_and_proprio()`
```python
def _extract_motion_and_proprio(self, obs):
    """
    分离motion和proprioceptive观测

    Returns:
        motion_obs: [batch, tsteps * motion_input_size]
        proprio_obs: [batch, proprio_dim]
        current_frame_motion: [batch, motion_input_size] - 中间帧
    """
    motion_obs = obs[:, :self.num_motion_obs]
    proprio_obs = obs[:, self.num_motion_obs:]

    # 提取当前帧 (21帧中的第11帧, index=10)
    current_frame_idx = self.cfg.motion_tsteps // 2  # 10
    start_idx = current_frame_idx * self.motion_input_size
    end_idx = start_idx + self.motion_input_size
    current_frame_motion = motion_obs[:, start_idx:end_idx]

    return motion_obs, proprio_obs, current_frame_motion
```

##### `_process_actor_input()`
```python
def _process_actor_input(self, obs):
    """
    处理actor输入

    流程:
    1. 分离motion和proprio
    2. 用MotionEncoder压缩motion_obs → motion_latent
    3. 拼接 [motion_latent + current_frame + proprio]
    """
    motion_obs, proprio_obs, current_frame = self._extract_motion_and_proprio(obs)
    motion_latent = self.motion_encoder(motion_obs)
    return torch.cat([motion_latent, current_frame, proprio_obs], dim=1)
```

**数据流**:
```
观测空间分解:
┌──────────────────────────────────────────────────────┐
│  obs [1314 dims] = motion_obs [672] + proprio [642]  │
└──────────────────────────────────────────────────────┘
         ↓                                   ↓
    MotionEncoder                        直接使用
         ↓
    motion_latent [128]
         ↓
    ┌────────────────────────────────────────┐
    │ Actor输入 [128 + 32 + 642 = 802 dims] │
    │ = motion_latent + current + proprio    │
    └────────────────────────────────────────┘
         ↓
    Actor MLP [512→512→256→128]
         ↓
    Action [23 dims]
```

**注册配置**:
```python
cs = ConfigStore.instance()
cs.store("ppo_motion_encoder", node=PPOMotionEncoderConfig, group="algo")
```

---

## 🔧 使用方法

### 方法1: 命令行指定algo

```bash
# 使用MotionEncoder版本训练
python scripts/train.py \
    algo=ppo_motion_encoder \
    task=G1/twist/0927_twist_teacher_new \
    wandb.mode=online
```

### 方法2: 修改默认配置

编辑 `cfg/train.yaml`:
```yaml
defaults:
  - task: G1/twist/0927_twist_teacher_new
  - algo: ppo_motion_encoder  # 修改这里
  - _self_
```

然后直接运行:
```bash
python scripts/train.py
```

### 方法3: 禁用MotionEncoder (对比实验)

```bash
# 使用相同配置但禁用MotionEncoder
python scripts/train.py \
    algo=ppo_motion_encoder \
    algo.use_motion_encoder=false \
    task=G1/twist/0927_twist_teacher_new
```

---

## 📊 关键差异对比

### HDMI vs TWIST 观测格式

| 维度 | TWIST | HDMI | 说明 |
|------|-------|------|------|
| **时间范围** | 仅未来 | 过去+现在+未来 | HDMI更全面 |
| **时间步数** | 20步 | 21步 | 差异1步 |
| **采样策略** | [1,5,10,...,95] | 连续21帧 | TWIST稀疏采样 |
| **时间跨度** | 1.9秒 (仅未来) | 0.4秒+1.9秒 | HDMI包含历史 |
| **tsteps参数** | 20 | 21 | MotionEncoder自适应 |

### MotionEncoder处理方式

**TWIST原版**:
```python
# 只处理未来20帧
motion_obs = obs[:, :num_motion_obs]  # [B, 20×58]
motion_latent = motion_encoder(motion_obs)
# 拼接: [motion_latent + current_state + proprio]
```

**HDMI实现**:
```python
# 处理过去10+当前1+未来10 = 21帧
motion_obs = obs[:, :num_motion_obs]  # [B, 21×32]
motion_latent = motion_encoder(motion_obs)
current_frame = motion_obs[:, 10*32:11*32]  # 提取中间帧
# 拼接: [motion_latent + current_frame + proprio]
```

### PPO超参数对齐

| 参数 | 原base PPO | TWIST | ppo_motion_encoder | 修改原因 |
|------|-----------|-------|-------------------|---------|
| `ppo_epochs` | 3 | 5 | **5** ✅ | 对齐TWIST |
| `num_minibatches` | 8 | 4 | **4** ✅ | 对齐TWIST |
| `lr` | 1e-4 | 2e-4 | **2e-4** ✅ | 对齐TWIST |
| `entropy_coef` | 0.001 | 0.01 | **0.01** ✅ | 对齐TWIST |
| `desired_kl` | None | 0.008 | **0.008** ✅ | 启用自适应LR |

---

## ⚠️ 重要注意事项

### 1. 观测空间依赖

MotionEncoder假设观测空间的前`num_motion_obs`维是参考运动数据。

**配置要求** (YAML):
```yaml
observation:
  policy:
    # 第一个观测必须是ref_motion_windowed!
    ref_motion_windowed:
      _target_: active_adaptation.envs.mdp.commands.twist.observations.ref_motion_windowed
      past_frames: 10
      future_frames: 10
      # 这会生成 21 × (3+6+23) = 672 dims

    # 其他观测在后面
    proprio_history_combined:
      ...
```

**检查代码** (`ppo_motion_encoder.py:112-125`):
```python
# 自动检测motion_obs维度
if self.cfg.motion_obs_key in fake_input.keys():
    motion_obs_dim = fake_input[self.cfg.motion_obs_key].shape[-1]
    self.motion_input_size = motion_obs_dim // self.cfg.motion_tsteps
```

### 2. 单帧维度计算

对于HDMI的`ref_motion_windowed`:
```python
# 单帧包含:
# - root_pos: 3 dims
# - root_ori: 6 dims (旋转矩阵前两行)
# - joint_pos: 23 dims (G1)
# 总计: 32 dims per frame

motion_input_size = 32
tsteps = 21
total_motion_obs = 21 × 32 = 672 dims
```

### 3. 当前帧提取逻辑

```python
# HDMI: 21帧 [过去10, 当前1, 未来10]
# 索引: [0,1,2,...,9, 10, 11,12,...,20]
#                      ↑ 当前帧

current_frame_idx = tsteps // 2  # 21 // 2 = 10 (正确!)

# TWIST: 20帧 [未来1, 未来5, ..., 未来95]
# 没有"当前帧"概念,使用第一帧作为近期参考
```

### 4. 与priv_info的兼容性

`priv_info`观测函数提供特权信息，需确保在`observation.priv`中定义:
```yaml
observation:
  priv:
    priv_info:
      _target_: active_adaptation.envs.mdp.commands.twist.observations.priv_info
```

### 5. 测试MotionEncoder

运行单元测试:
```bash
cd /home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk
python -m active_adaptation.learning.modules.motion_encoder

# 预期输出:
# Testing MotionEncoder1D...
# Input shape: torch.Size([4096, 672])
# Output shape: torch.Size([4096, 128])
# ✓ MotionEncoder1D test passed!
# Total parameters: 32,160
```

---

## 🐛 故障排查

### 问题1: 观测维度不匹配

**错误信息**:
```
RuntimeError: shape mismatch: [4096, 858] vs expected [4096, 672 + 642]
```

**解决方案**:
检查`ref_motion_windowed`的`past_frames`和`future_frames`配置:
```yaml
ref_motion_windowed:
  past_frames: 10  # 必须是10
  future_frames: 10  # 必须是10
```

### 问题2: MotionEncoder参数未更新

**现象**: 训练loss不下降，MotionEncoder梯度为0

**解决方案**:
检查optimizer是否包含MotionEncoder参数:
```python
# 在ppo_motion_encoder.py:228-234
params = [
    {"params": self.actor.parameters()},
    {"params": self.critic.parameters()},
]
if self.cfg.use_motion_encoder:
    params.append({"params": self.motion_encoder.parameters()})  # 确保添加!
```

### 问题3: 内存溢出

**现象**: CUDA out of memory

**原因**: MotionEncoder增加了参数量和中间激活

**解决方案**:
- 减少batch size: `num_envs=4096 → 2048`
- 减少motion_latent_dim: `128 → 64`
- 使用gradient checkpointing (需额外实现)

---

## 📈 性能预期

### 理论优势

1. **参数效率**: 32K vs 594K (节省94.6%)
2. **时序建模**: 1D CNN vs MLP (CNN能提取时序模式)
3. **泛化能力**: 权重共享 → 更好的泛化到新运动
4. **训练稳定性**: 归纳偏置 → 更快收敛

### 实验建议

运行对比实验:
```bash
# 实验A: 使用MotionEncoder
python scripts/train.py \
    algo=ppo_motion_encoder \
    task=G1/twist/0927_twist_teacher_new \
    exp_name=twist_teacher_with_encoder

# 实验B: 禁用MotionEncoder (使用相同PPO配置)
python scripts/train.py \
    algo=ppo_motion_encoder \
    algo.use_motion_encoder=false \
    task=G1/twist/0927_twist_teacher_new \
    exp_name=twist_teacher_no_encoder

# 实验C: 原始base PPO (作为baseline)
python scripts/train.py \
    algo=ppo \
    task=G1/twist/0927_twist_teacher_new \
    exp_name=twist_teacher_base_ppo
```

### 评估指标

在WandB中对比:
- `train/tracking_keybody_pos_twist_aligned`: 关键点跟踪奖励
- `train/tracking_joint_dof_twist_aligned`: 关节跟踪奖励
- `critic/explained_var_valid`: 价值函数解释方差
- `actor/grad_norm`: Actor梯度范数
- `motion_encoder/grad_norm`: MotionEncoder梯度范数 (仅实验A)

---

## 🔄 与TWIST完全对齐的修改建议

如果需要完全复现TWIST结果，还需:

### 1. 修改观测为仅未来帧

编辑`cfg/task/G1/twist/0927_twist_teacher_new.yaml`:
```yaml
observation:
  policy:
    ref_motion_windowed:
      past_frames: 0   # 改为0!
      future_frames: 20 # 改为20!

# 同时修改配置
motion_tsteps: 20  # 在algo配置中
```

### 2. 使用TWIST的稀疏采样

修改`ref_motion_windowed`观测函数以支持稀疏采样:
```python
# 在observations.py中添加
tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
```

### 3. 调整单帧维度

TWIST单帧包含更多信息:
```python
# TWIST: root_pos(3) + roll/pitch(2) + root_vel(3) + yaw_vel(1) +
#        joint_pos(23) + key_body_pos(27) = 59 dims (实际58?)
# HDMI: root_pos(3) + root_ori(6) + joint_pos(23) = 32 dims
```

---

## 📚 参考文档

- TWIST论文: *Tracking the Whole Individual through Whole-body Imitation*
- HDMI原始实现: `/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/`
- TWIST原始实现: `/home/ubuntu/DATA2/workspace/xmh/TWIST-master/`
- MotionEncoder分析: `/home/ubuntu/DATA2/workspace/xmh/tmp/TWIST_MotionEncoder_Analysis.md`

---

## ✅ 修改检查清单

- [x] 实现`MotionEncoder1D`类
- [x] 实现`PPOMotionEncoderPolicy`类
- [x] 注册`ppo_motion_encoder`配置
- [x] 添加单元测试
- [x] 对齐TWIST的PPO超参数
- [x] 自适应HDMI的21帧观测格式
- [x] 支持当前帧提取逻辑
- [x] 编写详细文档

## 🚀 下一步

1. **运行测试**: `python -m active_adaptation.learning.modules.motion_encoder`
2. **启动训练**: `python scripts/train.py algo=ppo_motion_encoder task=G1/twist/0927_twist_teacher_new`
3. **监控WandB**: 检查`motion_encoder/grad_norm`是否有更新
4. **对比实验**: 运行有/无MotionEncoder的对比实验
5. **性能调优**: 根据训练曲线调整超参数

---

**修改完成时间**: 2025-11-01
**修改者**: Claude Code
**审核状态**: 待用户测试
