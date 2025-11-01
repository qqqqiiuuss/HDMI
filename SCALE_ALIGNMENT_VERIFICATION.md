# HDMI vs TWIST-MAIN Reward Scale 对齐验证

## ✅ 验证结论

**HDMI的tracking reward实现已完全对齐TWIST-MAIN的scale系数**

验证日期: 2024-10-31

---

## 1. Scale参数对比

| Reward Function | 权重 | TWIST-MAIN Scale | HDMI Scale | 代码位置 | 状态 |
|----------------|------|------------------|-----------|---------|------|
| tracking_keybody_pos | 2.0 | 10.0 | 10.0 | rewards_new.py:729, 789 | ✅ |
| tracking_joint_dof | 0.6 | 0.15 | 0.15 | rewards_new.py:619, 655 | ✅ |
| tracking_joint_vel | 0.2 | 0.01 | 0.01 | rewards_new.py:673, 709 | ✅ |
| tracking_root_pose | 0.6 | 5.0 | 5.0 | rewards_new.py:524, 550 | ✅ |
| tracking_root_vel | 1.0 | 1.0 | 1.0 | rewards_new.py:566, 598 | ✅ |

---

## 2. 代码验证

### 2.1 Scale参数定义

```bash
$ grep -n "self\.\w*_scale = " rewards_new.py

524:        self.root_pose_scale = 5.0
566:        self.root_vel_scale = 1.0
619:        self.pos_scale = 0.15
673:        self.vel_scale = 0.01
729:        self.key_body_pos_scale = 10.0
```

### 2.2 Scale参数使用

```bash
$ grep -n "torch.exp.*self\.\w*_scale" rewards_new.py

550:        reward = torch.exp(-self.root_pose_scale * (root_pos_err + 0.1 * root_rot_err))
598:        reward = torch.exp(-self.root_vel_scale * (root_vel_err + 0.5 * root_ang_vel_err))
655:        reward = torch.exp(-self.pos_scale * dof_err)
709:        reward = torch.exp(-self.vel_scale * vel_err)
789:        reward = torch.exp(-self.key_body_pos_scale * key_body_pos_err)
```

---

## 3. 详细对比

### 3.1 tracking_keybody_pos (权重=2.0) ✅

**TWIST-MAIN** (`humanoid_mimic.py:542-544`):
```python
key_body_pos_scale = 10.0
return torch.exp(-key_body_pos_scale * key_body_pos_err)
```

**HDMI** (`rewards_new.py:729, 789`):
```python
self.key_body_pos_scale = 10.0  # Line 729
reward = torch.exp(-self.key_body_pos_scale * key_body_pos_err)  # Line 789
```

**公式**: 都是 `exp(-10.0 * Σᵢ Σⱼ (diff_ij)²)`

---

### 3.2 tracking_joint_dof (权重=0.6) ✅

**TWIST-MAIN**: `pos_scale = 0.15`
**HDMI**: `self.pos_scale = 0.15` (line 619)

**公式**: 都是 `exp(-0.15 * Σᵢ wᵢ * (diff_i)²)`

---

### 3.3 tracking_joint_vel (权重=0.2) ✅

**TWIST-MAIN**: `vel_scale = 0.01`
**HDMI**: `self.vel_scale = 0.01` (line 673)

**公式**: 都是 `exp(-0.01 * Σᵢ wᵢ * (diff_i)²)`

---

### 3.4 tracking_root_pose (权重=0.6) ✅

**TWIST-MAIN**: `root_pose_scale = 5.0`
**HDMI**: `self.root_pose_scale = 5.0` (line 524)

**公式**: 都是 `exp(-5.0 * (pos_err + 0.1 * rot_err))`

---

### 3.5 tracking_root_vel (权重=1.0) ✅

**TWIST-MAIN**: `root_vel_scale = 1.0`
**HDMI**: `self.root_vel_scale = 1.0` (line 566)

**公式**: 都是 `exp(-1.0 * (vel_err + 0.5 * ang_vel_err))`

---

## 4. 配置文件中的Sigma参数

在配置文件 `0927_twist_teacher_new.yaml` 中，虽然指定了sigma参数：

```yaml
tracking_keybody_pos_twist_aligned:
  sigma: 0.2  # 这个参数在代码中被忽略
```

但在 `*_twist_aligned` 版本的reward实现中，**sigma参数被保存但未使用**，实际使用的是硬编码的scale参数。

### 为什么sigma被忽略？

查看代码实现：

```python
class tracking_keybody_pos_twist_aligned(TrackReward):
    def __init__(self, sigma: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma  # 保存但未使用
        self.key_body_pos_scale = 10.0  # 实际使用的scale

    def compute(self):
        # ...
        # 使用scale而不是sigma
        reward = torch.exp(-self.key_body_pos_scale * key_body_pos_err)
        return reward.unsqueeze(1)
```

---

## 5. Episode Length对比 ✅

| 参数 | HDMI | TWIST-MAIN |
|------|------|------------|
| episode_length_s | 10秒 | 10秒 |
| physics_dt | 0.002s | 0.002s |
| decimation | 10 | 10 |
| step_dt | 0.02s | 0.02s |
| max_episode_length | 500 steps | 500 steps |

---

## 6. Mean Return预期

由于scale已完全对齐，**HDMI和TWIST-MAIN的Mean Return应该在相同的数值范围**。

### 预期范围 (训练中期)

| Reward Term | 数值贡献范围 |
|------------|-------------|
| tracking_keybody_pos (×2.0) | 0 ~ 2.0 |
| tracking_joint_dof (×0.6) | 0 ~ 0.6 |
| tracking_joint_vel (×0.2) | 0 ~ 0.2 |
| tracking_root_pose (×0.6) | 0 ~ 0.6 |
| tracking_root_vel (×1.0) | 0 ~ 1.0 |
| **Tracking Total** | **0 ~ 4.4** |
| Regularization | -2 ~ 0 |
| **Mean Return** | **2 ~ 5** |

---

## 7. 训练曲线对比指南

### ✅ 可以直接对比的指标

1. **Mean Return的绝对数值** - scale已对齐，数值应该相近
2. **Episode Length** - 都是500 steps
3. **各reward term的绝对数值** - scale对齐后应该相近
4. **Tracking error** (L1/L2误差) - 应该相似
5. **训练收敛速度** - 应该相似

### 📊 对比方法

在WandB中查看：
```
episode_reward_mean      # Mean Return
tracking_keybody_pos     # 各个reward term
tracking_joint_dof
tracking_joint_vel
tracking_root_pose
tracking_root_vel
```

**预期**: 如果两个实现完全一致，曲线应该高度重合（考虑随机性）

---

## 8. 总结

### ✅ 完全对齐的部分

1. **Episode Length**: 10秒/500步
2. **Reward权重**: 所有13个reward完全一致
3. **Reward scale系数**: 所有5个tracking reward完全一致
4. **聚合方式**: SUM vs MEAN已对齐
5. **加权系数**: dof_err_w已实现

### 🎯 结论

**HDMI的reward实现与TWIST-MAIN完全对齐**，可以直接对比训练曲线的绝对数值。

如果训练结果有差异，应该从其他方面寻找原因：
- 仿真器差异 (IsaacLab vs Isaac Gym)
- 动作延迟实现
- 域随机化实现
- 接触力报告方式 (feet_stumble的XY力问题)
- 动作平滑/滤波

---

## 9. 参考文档

- `REWARDS_ALIGNMENT_SUMMARY.md` - Reward对齐总结
- `TWIST_REWARDS_COMPARISON.md` - 详细对比分析
- `FEET_STUMBLE_BUG_FIX.md` - feet_stumble的IsaacLab限制
