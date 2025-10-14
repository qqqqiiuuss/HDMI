# TWIST Teacher Observation 结构分析

## TWIST配置 (g1_mimic_distill_config.py)

### 关键参数
```python
num_actions = 23  # G1机器人23个关节
tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]  # 20个未来步
```

### Observation维度计算

#### 1. **本体感知 (Proprioception) - n_proprio = 75**
```python
n_proprio = 3 + 2 + 3*num_actions
         = 3 + 2 + 3*23
         = 3 + 2 + 69
         = 75
```
**组成**:
- `3`: 根角速度 (root angular velocity)
- `2`: 重力投影 (projected gravity, 只有XY，Z可以推导)
- `3*23 = 69`: 关节状态
  - `23`: 关节位置 (joint position)
  - `23`: 关节速度 (joint velocity)
  - `23`: 上一个动作 (previous action)

#### 2. **运动跟踪观察 (Mimic Observations) - n_priv_mimic_obs = 1160**
```python
n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*9)
                 = 20 * (8 + 23 + 27)
                 = 20 * 58
                 = 1160
```
**每个时间步 (58维)**:
- `8`: 根状态 (root state)
  - `3`: 根位置 (root position)
  - `4`: 根旋转 (root quaternion, wxyz)
  - `1`: 根高度 (root height) - **实际上是3+4=7，可能有1维额外信息**
- `23`: 参考关节位置 (reference joint position)
- `27 = 3*9`: 9个关键点的3D位置 (9 key body positions)

**关键点 (9个)**:
- 左右脚踝 (left/right ankle_roll_link)
- 左右膝盖 (left/right knee_link)
- 左右肘部 (left/right elbow_link)
- 左右手 (left/right hand marker)
- 头部 (head_link)

#### 3. **特权信息 (Privileged Info) - n_priv_info = 85**
```python
n_priv_info = 3 + 1 + 3*9 + 2 + 4 + 1 + 2*num_actions
            = 3 + 1 + 27 + 2 + 4 + 1 + 46
            = 85
```
**组成**:
- `3`: 根线速度 (base linear velocity) - 世界坐标系
- `1`: 根高度 (root height)
- `27 = 3*9`: 9个关键点的3D位置 (当前时刻)
- `2`: 接触掩码 (contact mask) - 左右脚
- `4`: 特权潜变量 (privileged latent)
- `1`: **未知维度**
- `46 = 2*23`: 关节信息
  - `23`: **可能是关节力矩或刚度**
  - `23`: **可能是关节阻尼或其他属性**

### Actor和Critic的输入

#### **Actor输入 (Teacher)**
```python
# 总维度: 1160 + 75 + 85 = 1320
actor_obs = [
    n_priv_mimic_obs,  # 1160: 20步未来运动参考
    n_proprio,         # 75: 本体感知
    n_priv_info,       # 85: 特权信息
]
```

**网络结构**:
1. **MotionEncoder**:
   - 输入: `1160` (20步 × 58维)
   - 输出: `motion_latent_dim = 64`
   - 结构: `Linear(58, 60) -> Conv1d -> Conv1d -> Linear(60, 64)`

2. **Actor Backbone**:
   - 输入: `75 + 85 - 1160 + 64 + 58 = 142` 维
     - `75`: 本体感知 (去除motion obs)
     - `85`: 特权信息
     - `64`: motion latent
     - `58`: 当前时刻的运动观察
   - 隐藏层: `[256, 256, 256]`
   - 输出: `23` 个动作

#### **Critic输入 (Teacher)**
```python
# 总维度: 1160 + 75 + 85 + 3 = 1323
critic_obs = [
    n_priv_mimic_obs,  # 1160: 20步未来运动参考
    n_proprio,         # 75: 本体感知
    n_priv_info,       # 85: 特权信息
    extra_critic_obs,  # 3: 额外的critic观察 (可能是奖励相关信息)
]
```

**网络结构**:
- 与Actor类似，也使用MotionEncoder
- 输入: `142 + 3 = 145` 维 (经过encoder处理后)
- 隐藏层: `[256, 256, 256]`
- 输出: `1` (value)

---

## HDMI当前配置对比

### HDMI twist-base.yaml (需要修改)

当前配置:
```yaml
observation:
  policy:
    joint_pos_history: {history_steps: [0], noise_std: 0.015}  # [23]
    joint_vel_history: {history_steps: [0], noise_std: 0.05}   # [23]
    prev_actions: {steps: 1}                                    # [23]
    root_ang_vel_history: {history_steps: [0], noise_std: 0.05}  # [3]
    projected_gravity_history: {history_steps: [0], noise_std: 0.05}  # [3]

  command:
    ref_joint_pos_future: {}  # 需要定义维度
```

**问题**:
1. ❌ 缺少20步未来运动参考的完整定义
2. ❌ 缺少关键点位置观察
3. ❌ 缺少根状态观察
4. ❌ 特权信息不完整

---

## 需要修改的HDMI文件

### 1. **配置文件**
- ✅ `/home/ubuntu/DATA2/workspace/xmh/HDMI-main/cfg/task/base/twist-base-new.yaml`

### 2. **Observation函数** (核心)
- ✅ `/home/ubuntu/DATA2/workspace/xmh/HDMI-main/active_adaptation/envs/mdp/commands/twist/observations.py`
  - 需要添加:
    - `multi_step_ref_tracking`: 20步未来运动参考 (1160维)
    - `ref_key_body_pos`: 关键点位置
    - `ref_root_state`: 根状态

### 3. **Command类** (TwistMotionTracking)
- ✅ `/home/ubuntu/DATA2/workspace/xmh/HDMI-main/active_adaptation/envs/mdp/commands/twist/command.py`
  - 确保在`update()`中提供:
    - `ref_joint_pos_future_`: [num_envs, 20, 23]
    - `ref_key_body_pos_future_`: [num_envs, 20, 9, 3]
    - `ref_root_pos_future_w`: [num_envs, 20, 3]
    - `ref_root_quat_future_w`: [num_envs, 20, 4]

### 4. **网络结构** (如果要完全复制TWIST)
- ⚠️ `/home/ubuntu/DATA2/workspace/xmh/HDMI-main/active_adaptation/learning/modules/`
  - 需要添加 `MotionEncoder` (1D CNN)
  - 修改 Actor/Critic 使用MotionEncoder

### 5. **PPO算法** (可选)
- ⚠️ `/home/ubuntu/DATA2/workspace/xmh/HDMI-main/active_adaptation/learning/ppo/ppo_roa.py`
  - 确保observation spec匹配

---

## 关键差异总结

| 项目 | TWIST | HDMI (当前) | 需要修改 |
|------|-------|------------|---------|
| **未来步数** | 20步 | 10步 | ✅ 改为20步 |
| **未来步索引** | [1,5,10,...,95] | [1,2,3,...,10] | ✅ 改为TWIST索引 |
| **Motion Obs维度** | 1160 (20×58) | 未完整定义 | ✅ 添加完整定义 |
| **关键点数量** | 9个 | 未定义 | ✅ 定义9个关键点 |
| **本体感知** | 75维 | ~75维 | ✅ 确认维度匹配 |
| **特权信息** | 85维 | 不完整 | ✅ 添加完整特权信息 |
| **Motion Encoder** | 1D CNN | 未使用 | ⚠️ 可选添加 |

---

## 优先级

### 🔴 必须修改 (否则无法运行)
1. 配置文件: 定义完整的observation
2. Observation函数: 实现multi_step_ref_tracking
3. Command类: 提供20步未来参考

### 🟡 推荐修改 (提升性能)
4. 网络结构: 添加MotionEncoder (1D CNN)

### 🟢 可选修改
5. 其他超参数对齐 (学习率、奖励权重等)
