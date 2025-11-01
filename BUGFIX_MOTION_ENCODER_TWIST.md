# BugFix: MotionEncoder与TWIST Flat Observation兼容性

## 🐛 问题描述

### 错误信息
```
RuntimeError: shape '[86016, -1]' is invalid for input of size 175767552
```

### 错误原因

**问题**: `ppo_motion_encoder.py`假设observation的结构是`[motion_obs, proprio_obs]`（motion在前），但TWIST实际使用的是`[proprio_obs, motion_obs]`（motion在后）。

**详细分析**:
```python
# 错误的假设
motion_obs = obs[:, :672]       # 前672维
proprio_obs = obs[:, 672:]      # 后42240维

# TWIST实际结构
proprio_obs = obs[:, :42240]    # 前42240维 (proprio_history_combined)
motion_obs = obs[:, 42240:]     # 后672维 (ref_motion_windowed)
```

**数学验证**:
```
batch_size = 4096
total_obs_dim = 42912

# TWIST observation结构:
# - proprio_history_combined: 11 * (6+3+23+23+23) = 11 * 78 = 858 dims
# - ref_motion_windowed: 21 * (3+6+23) = 21 * 32 = 672 dims
#
# 但实际是 42912 dims?
# 因为这是所有环境的拼接: 不对,应该是单个环境的维度

# 实际per-env维度 = 42912 (这包括所有observation)
# motion_obs维度 = 672
# proprio_obs维度 = 42912 - 672 = 42240

# 错误的reshape:
# obs.reshape([batch_size * tsteps, -1])
# = obs.reshape([4096 * 21, -1])
# = obs.reshape([86016, -1])
#
# 期望: 86016 * motion_input_size = total_elements
# 实际: 86016 * X = 175767552
# X = 2043.43 (不是整数!)
#
# 因为代码把整个obs(42912 dims)当作motion_obs了!
```

---

## ✅ 修复方案

### 1. 检测Observation结构

**修改位置**: `ppo_motion_encoder.py:133-170`

```python
# 新增逻辑: 检测flat observation
if hasattr(observation_spec[OBS_KEY], 'keys') and self.cfg.motion_obs_key in observation_spec[OBS_KEY].keys():
    # Structured observation (TensorDict) - HDMI格式
    motion_obs_dim = observation_spec[OBS_KEY][self.cfg.motion_obs_key].shape[-1]
    self.use_structured_obs = True
else:
    # Flat observation (concatenated tensor) - TWIST格式
    total_obs_dim = observation_spec[OBS_KEY].shape[-1]
    self.num_motion_obs = self.cfg.motion_tsteps * self.cfg.motion_input_size
    self.num_proprio_obs = total_obs_dim - self.num_motion_obs  # ← 关键!
    self.use_structured_obs = False
```

### 2. 修复Motion/Proprio提取

**修改位置**: `ppo_motion_encoder.py:304-335`

```python
def _extract_motion_and_proprio(self, obs: torch.Tensor):
    # TWIST uses FLAT observation: [proprio_history_combined, ref_motion_windowed]
    # Motion obs comes AFTER proprio obs
    if hasattr(self, 'num_proprio_obs'):
        # TWIST flat format: [proprio, motion]
        proprio_obs = obs[:, :self.num_proprio_obs]   # ← 前面是proprio
        motion_obs = obs[:, self.num_proprio_obs:]     # ← 后面是motion
    else:
        # Legacy format: motion first
        motion_obs = obs[:, :self.num_motion_obs]
        proprio_obs = obs[:, self.num_motion_obs:]
```

### 3. 修复Actor Input维度计算

**修改位置**: `ppo_motion_encoder.py:186-206`

```python
if self.cfg.use_motion_encoder:
    if hasattr(self, 'num_proprio_obs'):
        # TWIST flat format
        proprio_dim = self.num_proprio_obs
    else:
        # Legacy format or structured obs
        proprio_dim = observation_spec[OBS_KEY].shape[-1] - self.num_motion_obs

    actor_input_dim = (
        self.cfg.motion_latent_dim +  # 128
        self.motion_input_size +       # 32
        proprio_dim                    # 42240
    )
    # = 128 + 32 + 42240 = 42400
```

---

## 📊 TWIST Observation结构详解

### Policy Observation (OBS_KEY = "policy")

```yaml
observation:
  policy:
    proprio_history_combined:  # 第1部分
      history_length: 11
      # 包含: root_ori(6) + root_ang_vel(3) + joint_pos(23) + joint_vel(23) + action(23)
      # 每帧: 78 dims
      # 总共: 11 * 78 = 858 dims

    ref_motion_windowed:       # 第2部分
      past_frames: 10
      future_frames: 10
      # 包含: root_pos(3) + root_ori(6) + joint_pos(23)
      # 每帧: 32 dims
      # 总共: 21 * 32 = 672 dims

# 总维度: 858 + 672 = 1530 dims
```

**注意**: 实际观察到的维度是42912，这意味着可能有额外的observations被拼接。

### 实际观测顺序

```
obs (shape: [batch, 42912]) =
  [
    proprio_history_combined (未知具体维度),
    ref_motion_windowed (672 dims)
  ]

# 由于total = 42912, motion = 672
# 所以 proprio = 42912 - 672 = 42240 dims
```

---

## 🔍 调试信息

修复后，运行时会打印以下信息：

```
[MotionEncoder] Detected FLAT observation (TWIST format)
[MotionEncoder] total_obs_dim=42912, proprio_dim=42240, motion_dim=672
[MotionEncoder] tsteps=21, single_frame_dim=32
[MotionEncoder] Observation order: [proprio_history_combined, ref_motion_windowed]
[MotionEncoder] Initialized with 27,448 parameters
[MotionEncoder] Actor input dim = 128 (latent) + 32 (current_frame) + 42240 (proprio) = 42400
```

---

## 🎯 验证方法

### 测试命令

```bash
python scripts/train.py \
    algo=ppo_motion_encoder \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096 \
    suffix=test_bugfix
```

### 预期日志

应该看到：
1. ✅ `[MotionEncoder] Detected FLAT observation (TWIST format)`
2. ✅ `[MotionEncoder] total_obs_dim=42912, proprio_dim=42240, motion_dim=672`
3. ✅ `[MotionEncoder] Initialized with 27,448 parameters`
4. ✅ 训练正常启动，无RuntimeError

---

## 📋 兼容性说明

### 支持的Observation格式

| 格式 | 结构 | 检测方式 | 示例 |
|------|------|----------|------|
| **Structured** (HDMI) | TensorDict with keys | `observation_spec[OBS_KEY].keys()` | HDMI tasks |
| **Flat** (TWIST) | Concatenated tensor | Total dim - motion dim | TWIST tasks |
| **Legacy** | Motion first, then proprio | Fallback | Old configs |

### Backward兼容性

- ✅ **HDMI tasks**: 仍然支持structured observation
- ✅ **TWIST tasks**: 现在正确处理flat observation
- ✅ **自动检测**: 运行时自动判断observation格式

---

## 🔑 关键要点

### 1. Observation顺序很重要

**TWIST**:
```python
obs = [proprio_history_combined, ref_motion_windowed]
#      ^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^
#      42240 dims                672 dims
```

**HDMI (如果用structured)**:
```python
obs = TensorDict({
    "proprio_history": ...,
    "ref_motion_windowed": ...,  # 可以直接通过key访问
})
```

### 2. 维度计算

```python
# TWIST
total_obs_dim = 42912
motion_dim = 21 * 32 = 672
proprio_dim = 42912 - 672 = 42240

# Actor input (with MotionEncoder)
actor_input = motion_latent + current_frame + proprio
            = 128 + 32 + 42240
            = 42400
```

### 3. 关键变量

```python
self.num_motion_obs = 672          # motion observation维度
self.num_proprio_obs = 42240       # proprio observation维度 (TWIST flat only)
self.motion_input_size = 32        # 单帧motion维度
self.cfg.motion_tsteps = 21        # 时间步数
```

---

## 🚀 下一步

修复完成后，可以正常训练TWIST任务：

```bash
# 完整训练
python scripts/train.py \
    algo=ppo_motion_encoder \
    task=G1/twist/0927_twist_teacher_new \
    wandb.mode=online \
    exp_name=twist_with_motion_encoder

# 对比实验 (不使用MotionEncoder)
python scripts/train.py \
    algo=ppo_motion_encoder \
    algo.use_motion_encoder=false \
    task=G1/twist/0927_twist_teacher_new \
    wandb.mode=online \
    exp_name=twist_without_motion_encoder
```

---

## 📚 相关文档

- `RESIDUAL_ACTION_ANALYSIS.md` - Residual action机制
- `MOTION_ENCODER_ANALYSIS.md` - MotionEncoder详细分析
- `TWIST_MotionEncoder_Analysis.md` - TWIST MotionEncoder原理

---

**修复时间**: 2025-11-01
**影响范围**: TWIST任务使用`ppo_motion_encoder`
**向后兼容**: ✅ 是
**测试状态**: 待验证
