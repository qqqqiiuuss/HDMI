# 验证HDMI vs TWIST Motion长度差异

## 问题：Episode Length差异的真正原因

如果reward值相同但episode_length不同（HDMI 80 vs TWIST 400），最可能的原因是：
**Motion数据的长度不同**

## 验证步骤

### 1. 检查Motion数据的平均长度

#### HDMI的motion数据：
```python
# 在Python环境中运行
import numpy as np
import os

data_dir = "/home/jane/workspace/xmh/HDMI/small_dataset"

# 读取所有motion文件
motion_lengths = []
for file in os.listdir(data_dir):
    if file.endswith('.npz'):
        motion = np.load(os.path.join(data_dir, file))
        # 假设motion数据的shape是 [T, ...]
        T = motion['joint_pos'].shape[0]
        fps = 50.0  # 从meta.json读取
        length_seconds = T / fps
        motion_lengths.append(length_seconds)
        print(f"{file}: {T} frames, {length_seconds:.2f} seconds, {T/fps*50:.0f} steps @ 50Hz")

print(f"\n平均motion长度: {np.mean(motion_lengths):.2f} 秒")
print(f"对应episode steps: {np.mean(motion_lengths) / 0.02:.0f} 步")
```

#### TWIST的motion数据：
```python
# 在TWIST环境中
from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg
from legged_gym.utils.motion_lib import MotionLib

cfg = G1MimicPrivCfg()
motion_lib = MotionLib(motion_file=cfg.motion.motion_file)

lengths = []
for i in range(motion_lib.num_motions()):
    length = motion_lib.get_motion_length(torch.tensor([i]))
    lengths.append(length.item())
    print(f"Motion {i}: {length.item():.2f} seconds, {length.item()/0.02:.0f} steps")

print(f"\n平均motion长度: {np.mean(lengths):.2f} 秒")
```

---

### 2. 检查HDMI是否实现motion_end终止

查看HDMI的终止条件配置：

```yaml
# 0927_twist_teacher_new.yaml
termination:
  cum_body_pos_error_local: enabled
  cum_body_z_error: enabled
  # ❌ 缺少 motion_end 终止条件！
```

对比TWIST-MAIN：
```python
# humanoid_mimic.py:286-290
motion_end = self.episode_length_buf * self.dt >= motion_length
self.reset_buf |= motion_end  # ✅ Motion结束时自动重置
```

---

### 3. 检查WandB记录的指标含义

确认你看到的"reward"和"episode_length"具体是什么：

#### 可能的指标：
1. **episode_reward_mean** - 每个episode的平均总reward
2. **episode_length_mean** - 每个episode的平均步数
3. **reward/step** - 每步的平均reward
4. **success_rate** - 成功完成motion的比例

#### 关键检查：
```
如果：
- HDMI的 episode_length_mean ≈ 80
- TWIST的 episode_length_mean ≈ 400
- 两者的 episode_reward_mean 相近

那么说明：
- HDMI的motion更短（1.6秒）
- TWIST的motion更长（8秒）
- 两者都成功完成了motion（所以reward相近）
```

---

## 预期结果

### 场景A: Motion长度不同（最可能）

如果HDMI的motion平均1.6秒，TWIST的motion平均8秒：
- ✅ 解释了episode_length差异（80 vs 400）
- ✅ 解释了reward相同（都成功跟踪）
- 🔧 解决方案：使用相同的motion数据

### 场景B: 缺少motion_end终止条件

如果两者motion长度相同，但HDMI缺少motion_end终止：
- HDMI在80步时tracking失败，提前终止
- TWIST在400步时motion正常结束
- 🔧 解决方案：添加motion_end终止条件

---

## 如何添加motion_end终止条件（如果需要）

### 方法1: 在terminations.py中添加

创建新的终止条件类：

```python
class motion_end(_cum_error_mixin, RobotTrackTermination):
    """Episode终止当motion播放完毕"""
    def __init__(self, min_steps: int = 1, **kwargs):
        BaseTermination.__init__(self, **kwargs)
        _cum_error_mixin.__init__(self, min_steps=min_steps, threshold=0.5)

    def update(self):
        # 检查当前时间是否超过motion长度
        current_time = self.env.episode_length_buf * self.env.step_dt
        motion_lengths = self.command_manager.motion_len

        # 如果motion播放完毕，error设为1.0（触发终止）
        motion_finished = current_time >= motion_lengths
        self.error[:] = motion_finished.float()
        super().update()
```

### 方法2: 在配置中启用

```yaml
termination:
  motion_end:
    _target_: active_adaptation.envs.mdp.commands.twist.terminations.motion_end
    min_steps: 1
    enabled: true
```

---

## 重要说明

**如果你的目标是复现TWIST-MAIN的训练曲线**，那么：

1. **必须使用相同的motion数据**
2. **必须实现motion_end终止条件**（TWIST核心机制之一）
3. **否则episode_length永远无法对齐**

TWIST的设计哲学是：
- Episode = 一次完整的motion播放
- 当motion结束时，episode干净地重置
- 而不是等tracking失败才重置

---

## 下一步

请运行验证步骤1的代码，确认：
1. HDMI的motion平均长度是多少？
2. TWIST的motion平均长度是多少？
3. 两者是否使用同一批数据？

然后我们可以：
- 如果motion长度不同 → 统一数据源
- 如果缺少motion_end → 实现该终止条件
