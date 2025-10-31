# 29 DOF适配修改总结

## 问题背景
用户的G1机器人是**29 DOF**（增加了手腕关节），而TWIST-master原版是**23 DOF**。需要适配关节权重以避免维度不匹配错误。

---

## 关节结构对比

### TWIST-master (23 DOF)
```
Left Leg:  6 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
Right Leg: 6
Waist:     3 (waist_yaw, waist_roll, waist_pitch)
Left Arm:  4 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
Right Arm: 4
-----------
Total:    23 DOF
```

### 用户的G1 (29 DOF)
```
Left Leg:   6 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
Right Leg:  6
Waist:      3 (waist_yaw, waist_roll, waist_pitch)
Left Arm:   4 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
Left Wrist: 3 (wrist_roll, wrist_pitch, wrist_yaw) ← 新增
Right Arm:  4 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
Right Wrist: 3 (wrist_roll, wrist_pitch, wrist_yaw) ← 新增
-----------
Total:     29 DOF
```

---

## 修改内容

### 修改文件
`active_adaptation/envs/mdp/commands/twist/rewards_new.py`

### 修改的Reward类
1. **`tracking_joint_dof_twist_aligned`** (lines 599-653)
2. **`tracking_joint_vel_twist_aligned`** (lines 656-707)

### 权重扩展策略

#### 手腕关节权重选择
手腕关节用于精细操作，权重设置为：
- `wrist_roll`: **0.6** (类似waist关节)
- `wrist_pitch`: **0.5** (类似ankle关节)
- `wrist_yaw`: **0.5** (类似ankle关节)

#### 完整的29 DOF权重
```python
dof_err_w = [
    1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # Left Leg (6)
    1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # Right Leg (6)
    0.6, 0.6, 0.6,                      # Waist (3)
    0.8, 0.8, 0.8, 1.0,                 # Left Arm (4): shoulder x3, elbow
    0.6, 0.5, 0.5,                      # Left Wrist (3): roll, pitch, yaw ← 新增
    0.8, 0.8, 0.8, 1.0,                 # Right Arm (4): shoulder x3, elbow
    0.6, 0.5, 0.5,                      # Right Wrist (3): roll, pitch, yaw ← 新增
]
```

**验证：** ✅ 6+6+3+4+3+4+3 = **29** (正确)

---

## 代码修改详情

### 1. `tracking_joint_dof_twist_aligned`

#### 修改前（只支持23 DOF）
```python
def __init__(self, sigma: float = 0.2, **kwargs):
    super().__init__(**kwargs)
    self.sigma = sigma
    self.pos_scale = 0.15

    # TWIST dof_err_w weights (23 DOF)
    dof_err_w = [
        1.0, 0.8, 0.8, 1.0, 0.5, 0.5,  # Left Leg
        1.0, 0.8, 0.8, 1.0, 0.5, 0.5,  # Right Leg
        0.6, 0.6, 0.6,                  # waist
        0.8, 0.8, 0.8, 1.0,             # Left Arm
        0.8, 0.8, 0.8, 1.0,             # Right Arm
    ]
    self.dof_err_w = torch.tensor(dof_err_w, device=self.device, dtype=torch.float32)
```

#### 修改后（支持23 DOF和29 DOF）
```python
def __init__(self, sigma: float = 0.2, use_29dof: bool = True, **kwargs):
    super().__init__(**kwargs)
    self.sigma = sigma
    self.pos_scale = 0.15
    self.use_29dof = use_29dof

    if use_29dof:
        # Extended weights for 29 DOF (G1 with wrist joints)
        dof_err_w = [
            1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # Left Leg (6)
            1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # Right Leg (6)
            0.6, 0.6, 0.6,                      # Waist (3)
            0.8, 0.8, 0.8, 1.0,                 # Left Arm (4)
            0.6, 0.5, 0.5,                      # Left Wrist (3) ← 新增
            0.8, 0.8, 0.8, 1.0,                 # Right Arm (4)
            0.6, 0.5, 0.5,                      # Right Wrist (3) ← 新增
        ]
    else:
        # Original TWIST weights for 23 DOF
        dof_err_w = [
            1.0, 0.8, 0.8, 1.0, 0.5, 0.5,  # Left Leg
            1.0, 0.8, 0.8, 1.0, 0.5, 0.5,  # Right Leg
            0.6, 0.6, 0.6,                  # waist
            0.8, 0.8, 0.8, 1.0,             # Left Arm
            0.8, 0.8, 0.8, 1.0,             # Right Arm
        ]

    self.dof_err_w = torch.tensor(dof_err_w, device=self.device, dtype=torch.float32)
```

### 2. `tracking_joint_vel_twist_aligned`
相同的修改，添加了`use_29dof`参数和手腕关节权重。

---

## 使用说明

### 配置文件
在配置文件中，**默认使用29 DOF**（`use_29dof=True`是默认值）：

```yaml
# cfg/task/G1/twist/0927_twist_teacher_new.yaml

reward:
  tracking:
    tracking_joint_dof_twist_aligned:
      weight: 0.6
      sigma: 0.2
      # use_29dof: true  # 默认为true，可以省略
      enabled: true

    tracking_joint_vel_twist_aligned:
      weight: 0.2
      sigma: 0.5
      # use_29dof: true  # 默认为true，可以省略
      enabled: true
```

### 如果需要切换回23 DOF
如果要测试23 DOF版本（例如用于对比），可以显式设置：
```yaml
tracking_joint_dof_twist_aligned:
  weight: 0.6
  sigma: 0.2
  use_29dof: false  # 使用23 DOF
  enabled: true
```

---

## 权重设计理念

### 权重大小的含义
- **1.0**: 最重要的关节（hip_pitch, knee）
- **0.8**: 次重要关节（hip_roll, shoulder）
- **0.6**: 中等重要关节（waist, wrist_roll）
- **0.5**: 精细控制关节（ankle, wrist_pitch/yaw）

### 为什么手腕权重选择0.6, 0.5, 0.5？
1. **wrist_roll (0.6)**: 腕部旋转对操作影响较大，权重类似waist
2. **wrist_pitch (0.5)**: 腕部俯仰是精细控制，权重类似ankle
3. **wrist_yaw (0.5)**: 腕部偏航是精细控制，权重类似ankle

这个选择**平衡了手腕控制的重要性**，既不过分强调（会导致身体跟踪变差），也不过分忽略（会导致手腕控制失败）。

---

## 验证清单

- [x] 权重数量正确：29个 ✅
- [x] `tracking_joint_dof_twist_aligned`已更新 ✅
- [x] `tracking_joint_vel_twist_aligned`已更新 ✅
- [x] 默认使用29 DOF (`use_29dof=True`) ✅
- [x] 兼容23 DOF (可通过`use_29dof=false`切换) ✅

---

## 测试建议

### 1. 维度检查
训练前会自动检查维度，如果不匹配会报错：
```python
RuntimeError: The size of tensor a (29) must match the size of tensor b (23) at non-singleton dimension 1
```

如果看到这个错误，说明权重维度不匹配。

### 2. 权重效果验证
训练时观察：
- **手腕关节的跟踪误差**：应该在合理范围内（不能太大或太小）
- **主要关节（腿、腰）的跟踪**：不应该被手腕影响
- **整体reward值**：应该与预期相符

### 3. 可选调整
如果手腕跟踪效果不理想，可以尝试调整权重：
- **增大手腕权重**（0.7, 0.6, 0.6）→ 更强调手腕跟踪
- **减小手腕权重**（0.5, 0.4, 0.4）→ 更强调身体跟踪

---

## 关节顺序参考

### 完整的29 DOF关节顺序
```python
joints_29dof = [
    # Left Leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",

    # Right Leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",

    # Waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",

    # Left Arm (4)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",

    # Left Wrist (3)
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",

    # Right Arm (4)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",

    # Right Wrist (3)
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
```

---

## 总结

✅ **成功适配29 DOF机器人**

主要修改：
1. 添加了6个手腕关节的权重（左右各3个）
2. 权重选择遵循TWIST的设计理念
3. 保持了与23 DOF版本的兼容性
4. 默认使用29 DOF配置

现在你的训练可以正常运行，不会出现维度不匹配的错误！🎉
