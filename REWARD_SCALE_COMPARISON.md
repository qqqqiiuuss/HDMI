# HDMI vs TWIST Reward Scale 对比分析

## ✅ 结论：HDMI已完全对齐TWIST-MAIN的Scale

**验证日期**: 2024-10-31

经过代码验证，**HDMI的tracking reward实现已经完全对齐TWIST-MAIN的scale系数**：

| Reward | TWIST-MAIN Scale | HDMI Scale | 状态 |
|--------|------------------|-----------|------|
| tracking_keybody_pos | 10.0 | 10.0 | ✅ 完全一致 |
| tracking_joint_dof | 0.15 | 0.15 | ✅ 完全一致 |
| tracking_joint_vel | 0.01 | 0.01 | ✅ 完全一致 |
| tracking_root_pose | 5.0 | 5.0 | ✅ 完全一致 |
| tracking_root_vel | 1.0 | 1.0 | ✅ 完全一致 |

**验证代码位置**:
- `rewards_new.py:729` - `self.key_body_pos_scale = 10.0`
- `rewards_new.py:619` - `self.pos_scale = 0.15`
- `rewards_new.py:673` - `self.vel_scale = 0.01`
- `rewards_new.py:524` - `self.root_pose_scale = 5.0`
- `rewards_new.py:566` - `self.root_vel_scale = 1.0`

**重要说明**: 配置文件中的`sigma`参数虽然存在，但在`*_twist_aligned`版本的reward函数中**被忽略**，实际使用的是硬编码的scale参数。

---

## 1. Episode Length - ✅ 完全一致

| 参数 | HDMI | TWIST-MAIN |
|------|------|------------|
| episode_length_s | 10秒 | 10秒 |
| physics_dt | 0.002s (MuJoCo) / 0.005s (Isaac) | 0.002s |
| decimation | 10 | 10 |
| step_dt | 0.02s | 0.02s |
| max_episode_length | 500 steps | 500 steps |

**结论**: 时序尺度完全一致。

---

## 2. Mean Return Scale - ✅ 完全一致

由于HDMI已完全对齐TWIST-MAIN的scale系数，**Mean Return的数值范围应该一致**。

### Scale对齐验证表

| Reward | 权重 | TWIST Scale | HDMI Scale | 状态 |
|--------|------|-------------|-----------|------|
| tracking_keybody_pos | 2.0 | 10.0 | 10.0 | ✅ 一致 |
| tracking_joint_dof | 0.6 | 0.15 | 0.15 | ✅ 一致 |
| tracking_joint_vel | 0.2 | 0.01 | 0.01 | ✅ 一致 |
| tracking_root_pose | 0.6 | 5.0 | 5.0 | ✅ 一致 |
| tracking_root_vel | 1.0 | 1.0 | 1.0 | ✅ 一致 |

---

## 3. 详细公式对比

### 3.1 tracking_keybody_pos (权重=2.0) - ✅ 完全一致

**TWIST-MAIN** (`humanoid_mimic.py:524-544`):
```python
key_body_pos_err = torch.sum(key_body_pos_diff ** 2, dim=-1)  # per body
key_body_pos_err = torch.sum(key_body_pos_err, dim=-1)        # sum over 9 bodies
key_body_pos_scale = 10.0
reward = torch.exp(-key_body_pos_scale * key_body_pos_err)
```
**公式**: `r = exp(-10.0 * Σᵢ₌₁⁹ Σⱼ₌₁³ (diff_ij)²)`

---

**HDMI** (`rewards_new.py:714-791`):
```python
# Line 729: self.key_body_pos_scale = 10.0
key_body_pos_diff = key_body_pos - ref_key_body_pos
key_body_pos_err = (key_body_pos_diff ** 2).sum(dim=-1)  # sum over xyz
key_body_pos_err = key_body_pos_err.sum(dim=-1)  # sum over all key bodies
reward = torch.exp(-self.key_body_pos_scale * key_body_pos_err)  # Line 789
```
**公式**: `r = exp(-10.0 * Σᵢ Σⱼ (diff_ij)²)`

**对比结果**: ✅ **完全一致** - 聚合方式（SUM）和scale (10.0) 都相同

---

### 3.2 tracking_joint_dof (权重=0.6)

**TWIST-MAIN** (`humanoid_mimic.py:438-442`):
```python
pos_diff = self._ref_dof_pos - self.dof_pos
pos_err = torch.sum(self._dof_err_w * pos_diff ** 2, dim=-1)  # weighted sum
pos_scale = 0.15
reward = torch.exp(-pos_scale * pos_err)
```
**公式**: `r = exp(-0.15 * Σᵢ wᵢ * (diff_i)²)`

**特点**:
- 加权: dof_err_w = [1.0, 0.8, ..., 0.5] (23个关节)
- Scale: **0.15**
- 聚合: 加权SUM

---

**HDMI** (sigma=0.2):
```python
# 公式: exp(-error / 0.2) = exp(-5.0 * error)
```
**特点**:
- Scale: **5.0** (通过sigma实现)
- 可能也使用dof_err_w加权（需要确认）

**差异**: HDMI的scale (5.0) 是TWIST (0.15) 的33倍！

---

### 3.3 tracking_joint_vel (权重=0.2)

**TWIST-MAIN** (`humanoid_mimic.py:449-453`):
```python
vel_diff = self._ref_dof_vel - self.dof_vel
vel_err = torch.sum(self._dof_err_w * vel_diff ** 2, dim=-1)
vel_scale = 0.01
reward = torch.exp(-vel_scale * vel_err)
```
**公式**: `r = exp(-0.01 * Σᵢ wᵢ * (diff_i)²)`

**特点**:
- Scale: **0.01**
- 加权: 与joint_dof相同

---

**HDMI** (sigma=0.5):
```python
# exp(-error / 0.5) = exp(-2.0 * error)
```
**特点**:
- Scale: **2.0**

**差异**: HDMI的scale (2.0) 是TWIST (0.01) 的200倍！

---

### 3.4 tracking_root_pose (权重=0.6)

**TWIST-MAIN** (`humanoid_mimic.py:461-476`):
```python
root_pos_err = torch.sum(root_pos_diff ** 2, dim=-1)
root_rot_err = quat_diff_angle(...) ** 2
root_pose_scale = 5.0
reward = torch.exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))
```
**公式**: `r = exp(-5.0 * (pos_err + 0.1 * rot_err))`

**特点**:
- Scale: **5.0**
- 位置:旋转比例 = 1:0.1

---

**HDMI** (sigma=0.125 for rotation):
```python
# 配置: sigma=0.125 (对应rotation的tracking_sigma_ang)
# 可能: exp(-pos_err/0.2 - rot_err/0.125)
#     = exp(-5.0*pos_err - 8.0*rot_err)
```
**特点**:
- Position scale: ~5.0 (如果sigma=0.2)
- Rotation scale: ~8.0 (如果sigma=0.125)

**差异**: 旋转误差的权重可能不同（TWIST是0.1×pos_scale，HDMI可能更高）

---

### 3.5 tracking_root_vel (权重=1.0)

**TWIST-MAIN** (`humanoid_mimic.py:490-507`):
```python
root_vel_err = torch.sum(root_vel_diff ** 2, dim=-1)
root_ang_vel_err = torch.sum(root_ang_vel_diff ** 2, dim=-1)
root_vel_scale = 1.0
reward = torch.exp(-root_vel_scale * (root_vel_err + 0.5 * root_ang_vel_err))
```
**公式**: `r = exp(-1.0 * (vel_err + 0.5 * ang_vel_err))`

**特点**:
- Scale: **1.0**
- 线速度:角速度比例 = 1:0.5

---

**HDMI** (sigma=0.5):
```python
# exp(-error / 0.5) = exp(-2.0 * error)
```
**特点**:
- Scale: **2.0**

**差异**: HDMI的scale (2.0) 是TWIST (1.0) 的2倍

---

## 4. Mean Return预期范围

基于上述scale差异，预期的reward数值：

### TWIST-MAIN (Isaac Gym)

| Reward | 数值范围 | 备注 |
|--------|---------|------|
| tracking_keybody_pos (×2.0) | 0 ~ 2.0 | scale=10.0, 主要贡献 |
| tracking_joint_dof (×0.6) | 0 ~ 0.6 | scale=0.15 |
| tracking_joint_vel (×0.2) | 0 ~ 0.2 | scale=0.01 |
| tracking_root_pose (×0.6) | 0 ~ 0.6 | scale=5.0 |
| tracking_root_vel (×1.0) | 0 ~ 1.0 | scale=1.0 |
| **Tracking Total** | **0 ~ 4.4** | 理想情况 |
| Regularization | -2 ~ 0 | 惩罚项 |
| **Mean Return** | **2 ~ 5** | 训练中期 |

---

### HDMI (IsaacLab)

由于scale系数不同，mean return的**绝对数值可能不同**，但**训练曲线的形状应该相似**。

---

## 5. 实验验证建议

### 方法1: 打印单步reward分解

在训练中打印每个reward term的数值：

```python
# 在环境step后
print(f"keybody_pos: {reward_dict['tracking_keybody_pos_twist_aligned'].mean():.4f}")
print(f"joint_dof: {reward_dict['tracking_joint_dof_twist_aligned'].mean():.4f}")
print(f"joint_vel: {reward_dict['tracking_joint_vel_twist_aligned'].mean():.4f}")
print(f"root_pose: {reward_dict['tracking_root_pose_twist_aligned'].mean():.4f}")
print(f"root_vel: {reward_dict['tracking_root_vel_twist_aligned'].mean():.4f}")
print(f"Total: {total_reward.mean():.4f}")
```

**预期对比**:
- 如果HDMI和TWIST的tracking reward数值相近 → scale已对齐
- 如果HDMI某项远大于/小于TWIST → scale未对齐

---

### 方法2: 对比WandB曲线

在WandB中查看：
1. `episode_reward_mean` (或 `mean_return`)
2. 各个reward term的分解图

**关注点**:
- **绝对数值**: 可能不同（因为scale不同）
- **相对比例**: 应该相似
- **收敛速度**: 应该相似
- **最终性能**: tracking error应该相似

---

### 方法3: 检查HDMI的实际实现

查看HDMI的tracking reward是否使用了与TWIST相同的scale：

```bash
# 搜索scale关键字
grep -n "key_body_pos_scale\|pos_scale\|vel_scale" \
  active_adaptation/envs/mdp/commands/twist/rewards_new.py
```

---

## 6. 如何解决差异

### 选项1: 保持现状（推荐）

**理由**:
- 只要reward权重一致，不同的scale只是**数值尺度**不同
- 训练效果和最终性能应该相似
- PPO是相对稳定的算法，对reward scale不敏感

**建议**:
- 继续训练，对比最终tracking error
- 如果性能相似，说明scale差异不影响

---

### 选项2: 修改HDMI以完全对齐TWIST

需要修改HDMI的tracking reward实现，添加显式的scale参数：

```python
class tracking_keybody_pos_twist_aligned(TrackReward):
    def __init__(self, sigma: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        # TWIST使用scale=10.0, 而不是通过sigma
        self.key_body_pos_scale = 10.0
        # 忽略sigma参数，直接使用scale

    def compute(self):
        # ... compute error ...
        return torch.exp(-self.key_body_pos_scale * error)
```

**优点**: 完全对齐TWIST的实现
**缺点**: 需要重新训练，之前的checkpoint可能不兼容

---

## 7. 总结

### ✅ 一致的部分
1. **Episode length**: 10秒/500步
2. **Reward权重**: 完全一致
3. **Reward算法**: 大部分对齐（除scale外）
4. **Termination条件**: 对齐

### ⚠️ 差异的部分
1. **Tracking reward的scale系数**: 不同的sigma和显式scale
2. **Mean return的绝对数值**: 可能不同
3. **训练曲线的Y轴尺度**: 需要分别解读

### 📝 关键要点

**在对比训练曲线时**:
- ❌ 不要直接对比 `mean_return` 的绝对数值
- ✅ 对比训练的**收敛速度**和**稳定性**
- ✅ 对比最终的**tracking error** (误差指标，不是reward)
- ✅ 对比各个reward term的**相对贡献比例**

**Mean return的意义**:
- TWIST: mean_return ≈ 3-5 表示良好的tracking
- HDMI: 由于scale不同，数值可能在不同范围，需要单独建立baseline

---

## 8. 后续行动

1. **✅ 验证scale差异**: 打印单步reward分解
2. **✅ 检查HDMI实现**: 确认是否使用了显式scale
3. **⚠️ 决策**: 是否需要修改HDMI以完全对齐scale
4. **📊 训练对比**: 在不同scale下训练，对比最终性能

---

## 附录: TWIST-MAIN的Scale汇总

| Function | Scale | 公式 |
|----------|-------|------|
| tracking_keybody_pos | 10.0 | `exp(-10.0 * sum(diff²))` |
| tracking_joint_dof | 0.15 | `exp(-0.15 * sum(w*diff²))` |
| tracking_joint_vel | 0.01 | `exp(-0.01 * sum(w*diff²))` |
| tracking_root_pose | 5.0 | `exp(-5.0 * (pos_err + 0.1*rot_err))` |
| tracking_root_vel | 1.0 | `exp(-1.0 * (vel_err + 0.5*ang_vel_err))` |

这些scale参数在TWIST-MAIN中是**硬编码**的，不通过sigma配置。
