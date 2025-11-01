# PPO_TWIST Penalty 过大问题诊断报告

## 问题症状

训练 ppo_twist 时出现以下异常表现：

### ❌ 异常奖励项
1. **`joint_torque_limit` penalty 特别大** - 关节力矩经常超限
2. **`dof_vel_limit` penalty 特别大** - 关节速度经常超限
3. **`tracking_root_vel` 很小** - 根节点速度跟踪差
4. **`tracking_joint_vel` 很小** - 关节速度跟踪差

### ✅ 正常奖励项
5. **`tracking_dof_pos` 还可以** - 关节位置跟踪好
6. **`tracking_root_pos` 还可以** - 根节点位置跟踪好

---

## 核心问题分析

### **关键洞察**

**位置能跟踪，但速度跟不上** → 机器人**动作过于激进/震荡/不平滑**

这说明机器人能到达目标位置，但运动过程**不连续、不平滑**，导致：
- 速度突变 → `tracking_joint_vel` 和 `tracking_root_vel` 差
- 加速度过大 → 需要更大力矩 → `joint_torque_limit` 超限
- 速度过大 → `dof_vel_limit` 超限

**典型场景**：机器人像"抽搐"一样运动，虽然每帧都能到达目标位置，但中间过程非常不平滑。

---

## 根本原因：PD 控制器增益不匹配

### 1. PD 增益对比

#### HDMI (实际值)
```python
# active_adaptation/assets/g1.py
NATURAL_FREQ = 10 * 2π = 62.83 rad/s
DAMPING_RATIO = 2.0

# 计算结果：
Hip pitch/yaw: K = 40.2 N·m/rad,  D = 2.6 N·m·s/rad
Hip roll/Knee: K = 99.1 N·m/rad,  D = 6.3 N·m·s/rad
Ankle/Waist:   K = 28.5 N·m/rad,  D = 1.8 N·m·s/rad
```

#### TWIST-MASTER (目标值)
```python
# TWIST-master/legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py
stiffness = {
    'hip_yaw':   100,  # K
    'hip_roll':  100,
    'hip_pitch': 100,
    'knee':      150,
    'ankle':     40,
    'waist':     150,
}

damping = {
    'hip_yaw':   2,    # D
    'hip_roll':  2,
    'hip_pitch': 2,
    'knee':      4,
    'ankle':     2,
    'waist':     4,
}
```

### 2. 关键差异

| 关节 | HDMI K | TWIST K | 比例 | HDMI D | TWIST D | 比例 |
|------|--------|---------|------|--------|---------|------|
| **Hip pitch/yaw** | 40.2 | 100 | **0.40x** ❌ | 2.6 | 2 | 1.30x |
| **Hip roll** | 99.1 | 100 | 0.99x ✅ | 6.3 | 2 | **3.15x** ❌ |
| **Knee** | 99.1 | 150 | **0.66x** ❌ | 6.3 | 4 | **1.58x** ❌ |
| **Ankle** | 28.5 | 40 | **0.71x** ❌ | 1.8 | 2 | 0.90x |
| **Waist** | 28.5 | 150 | **0.19x** ❌❌❌ | 1.8 | 4 | **0.45x** ❌ |

**关键发现**：
1. ❌ **Stiffness (K) 普遍偏低**：40%-70%，最严重的是 **Waist 只有 19%**
2. ❌ **Damping (D) 不一致**：有些偏高（Hip roll 3.15x），有些偏低（Waist 0.45x）

---

## 问题机理

### 为什么 PD 增益不匹配会导致这些症状？

#### 1. **Stiffness 过低** → 位置跟踪"软"

**影响**：
- 机器人无法快速响应目标位置变化
- 为了追赶目标，策略网络输出更大的 action
- **更大的 action** → **更大的速度和加速度**

**数学解释**：
```
PD 控制器：τ = K·(θ_target - θ) - D·θ̇

如果 K 太小：
  - 同样的位置误差产生的力矩更小
  - 机器人响应慢，位置误差持续存在
  - 策略为了消除误差，输出更极端的 θ_target
  - 导致 θ̇ 和 θ̈ 都很大
```

#### 2. **Waist 增益过低** → 全身运动失调

**Waist (腰部) 是关键**：
- Waist 连接上下半身
- **TWIST K=150**，但 **HDMI K=28.5** (只有 19%)
- Waist 无法提供足够的支撑力矩

**结果**：
- 上半身控制不稳定
- 为了补偿，腿部和臀部需要更大力矩
- **→ `joint_torque_limit` penalty 增加**

#### 3. **Damping 不匹配** → 运动震荡

**Hip roll damping 过高** (6.3 vs 2.0):
- 过度阻尼，运动"粘滞"
- 速度响应慢，无法跟踪快速运动
- **→ `tracking_joint_vel` 和 `tracking_root_vel` 下降**

**Waist damping 过低** (1.8 vs 4.0):
- 欠阻尼，运动震荡
- 上半身不稳定
- **→ 整体运动质量下降**

---

## 其他发现的差异

### 3. Action Delay 和 Filtering

#### HDMI 配置
```yaml
# cfg/task/base/twist-base.yaml:53-55
action:
  min_delay: 2
  max_delay: 6
  alpha: [0.8, 1.0]  # EMA 滤波器
```

**Action 处理流程**：
```python
# active_adaptation/envs/mdp/action.py:115
self.applied_action.lerp_(action.squeeze(-1), self.alpha)
# 等价于: applied_action = (1-α)·old + α·new
# α ∈ [0.8, 1.0]，大部分环境使用接近1.0的值
```

#### TWIST-MASTER
```python
# 没有 action delay 和 EMA 滤波
# Action 直接应用
```

**影响**：
- **Delay (2-6步)**: 引入 40-120ms 延迟
- **EMA 滤波**: 轻微平滑 action (α≈1.0时影响小)
- **总体**: 延迟可能导致控制不稳定，策略需要"预测"未来

---

## 为什么普通 PPO 效果更好？

### 对比分析

| 配置项 | ppo | ppo_twist | 影响 |
|--------|-----|-----------|------|
| **初始 action std** | 1.5 | 1.0 | ppo 探索更多 |
| **Std schedule** | 无（固定1.5） | 有（1.0→0.4） | ppo_twist 后期探索少 |
| **Entropy coef** | 0.001 | 0.01 | ppo_twist 策略更随机 |
| **网络深度** | 3层 | 4层 | ppo_twist 更复杂，收敛慢 |

### 关键点

1. **ppo 的高探索 (std=1.5) 可能"掩盖"了 PD 增益问题**
   - 更大的随机性 → 动作更平滑（被随机噪声平滑）
   - 碰巧找到了能在低 K 值下工作的策略

2. **ppo_twist 的低探索 (std=1.0→0.4) 暴露了 PD 增益问题**
   - 更确定性的策略 → 需要精确的控制
   - 低 K 值无法提供精确控制 → 失败

**类比**：
- **ppo**: 像新手开车，方向盘幅度大但平滑，适应了"软"的转向系统
- **ppo_twist**: 像老手开车，方向盘幅度小但精确，发现转向系统太"软"无法精确控制

---

## 解决方案

### **方案 1: 修正 PD 增益 (推荐) ⭐**

修改 `active_adaptation/assets/g1.py`，手动设置 PD 增益以匹配 TWIST：

```python
# 在 g1.py 中添加/修改 actuators 配置

actuators={
    "legs_hip": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
        ],
        effort_limit_sim={...},  # 保持不变
        velocity_limit_sim={...},  # 保持不变
        stiffness={
            ".*_hip_yaw_joint": 100.0,    # TWIST-aligned
            ".*_hip_roll_joint": 100.0,   # TWIST-aligned
            ".*_hip_pitch_joint": 100.0,  # TWIST-aligned
        },
        damping={
            ".*_hip_yaw_joint": 2.0,      # TWIST-aligned
            ".*_hip_roll_joint": 2.0,     # TWIST-aligned
            ".*_hip_pitch_joint": 2.0,    # TWIST-aligned
        },
        # armature 保持不变或调整
    ),
    "legs_knee": ImplicitActuatorCfg(
        joint_names_expr=[".*_knee_joint"],
        effort_limit_sim={...},
        velocity_limit_sim={...},
        stiffness={".*_knee_joint": 150.0},  # TWIST-aligned
        damping={".*_knee_joint": 4.0},      # TWIST-aligned
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=40.0,   # TWIST-aligned (per joint)
        damping=2.0,      # TWIST-aligned
    ),
    "waist": ImplicitActuatorCfg(
        joint_names_expr=["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"],
        stiffness=150.0,  # TWIST-aligned ⚠️ 重要！
        damping=4.0,      # TWIST-aligned
    ),
    # ... shoulders, elbows 同理
}
```

**预期效果**：
- ✅ 运动更平滑，速度跟踪改善
- ✅ 力矩需求降低，`joint_torque_limit` penalty 减少
- ✅ 速度更可控，`dof_vel_limit` penalty 减少
- ✅ 整体 reward 提升，接近 TWIST-MASTER 水平

---

### **方案 2: 移除 Action Delay**

修改 `cfg/task/G1/twist/0927_twist_teacher_new.yaml`:

```yaml
action:
  _target_: active_adaptation.envs.mdp.action.JointPosition
  action_scaling: {...}
  min_delay: 0   # 修改：移除延迟
  max_delay: 0   # 修改：移除延迟
  alpha: [1.0, 1.0]  # 修改：禁用 EMA 滤波
```

**预期效果**：
- 减少控制延迟
- 策略响应更及时
- **但不能解决 PD 增益问题**

---

### **方案 3: 临时调整训练超参数（不推荐）**

如果无法修改 PD 增益，可以临时调整训练配置：

```yaml
# 增加探索，掩盖 PD 问题（不治本）
algo:
  init_noise_scale: 1.5  # 提高初始 std
  use_std_schedule: false  # 禁用 std 调度
  entropy_coef: 0.001  # 降低熵，让策略更确定
```

**缺点**：
- 治标不治本
- 性能仍然不如正确的 PD 增益
- 无法复现 TWIST-MASTER 效果

---

## 验证方法

### 1. 修改后重新训练

```bash
python scripts/train.py \
    algo=ppo_twist \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=4096 \
    suffix=twist_fixed_pd_gains
```

### 2. 监控关键指标

在 WandB 查看以下曲线变化：

**应该改善的指标**：
- ✅ `reward/tracking_root_vel` ↑ (应该显著提升)
- ✅ `reward/tracking_joint_vel` ↑ (应该显著提升)
- ✅ `reward/joint_torque_limits_twist` ↑ (penalty 减少，值变大)
- ✅ `reward/dof_vel_twist` ↑ (penalty 减少，值变大)

**应该保持的指标**：
- ✅ `reward/tracking_dof_pos` ≈ (保持不变)
- ✅ `reward/tracking_root_pos` ≈ (保持不变)

### 3. 对比 TWIST-MASTER

修复后，ppo_twist 的 reward 曲线应该与 TWIST-MASTER teacher 非常接近。

---

## 总结

### 核心问题

**PD 控制器增益严重不匹配**，尤其是：
1. **Waist K=28.5 (应为 150)** - 只有 19% ❌❌❌
2. **Hip pitch/yaw K=40.2 (应为 100)** - 只有 40% ❌
3. **Knee K=99.1 (应为 150)** - 只有 66% ❌
4. **Damping 值不一致** - 导致运动震荡

### 症状链

```
PD 增益过低
  ↓
关节响应慢、"软"
  ↓
策略输出更大 action 补偿
  ↓
速度和加速度过大
  ↓
┌─────────────────┬──────────────────┐
│ 力矩需求增加    │ 速度跟踪变差    │
│ ↓               │ ↓                │
│ torque_limit ↑  │ tracking_vel ↓   │
└─────────────────┴──────────────────┘
```

### 优先级

**Priority 1 (立即修复)**:
- ✅ 修正 PD 增益，尤其是 **Waist** (影响最大)

**Priority 2 (建议修复)**:
- ✅ 移除 action delay
- ✅ 禁用 EMA 滤波 (alpha=1.0)

**Priority 3 (可选)**:
- 验证其他超参数是否完全对齐

---

## 代码修改位置

**文件**: `/home/ubuntu/DATA2/workspace/xmh/tmp/HDMI-todesk/active_adaptation/assets/g1.py`

需要修改的行数：约 100-200 行（actuators 配置部分）

**建议**: 创建一个新的 robot type `g1_29dof_nohand-feet_sphere_twist` 专门用于 TWIST 训练，使用 TWIST-aligned PD 增益，避免影响其他任务的配置。
