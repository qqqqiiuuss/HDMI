# 数据格式一致性检查报告

## 检查目标

验证 TWIST frozen policy 在训练（PKL格式）和HDMI推理（NPZ格式）时使用的数据格式是否一致：
1. **四元数格式** (wxyz vs xyzw)
2. **Root位置表示**
3. **关节位置顺序**

## 检查结果

### ✅ 1. 四元数格式：一致 (wxyz)

#### NPZ格式 (HDMI任务数据)
- **文件**: `data/motion/g1/omomo/sub1_suitcase_011/meta.json`
- **格式声明**: `body_quat_w: 世界坐标系下的身体四元数 [T, N_bodies, 4] (wxyz格式)`
- **代码确认**: `active_adaptation/envs/isaac/mujoco.py:272`
  ```python
  body_quat_w = self.mj_data.xquat[self.body_adrs_read] # wxyz
  ```

#### PKL格式 (TWIST训练数据)
- **文件**: `/home/ubuntu/DATA2/Dataset-G1/AMASS_G1/accad/*.pkl`
- **原始格式**: `root_rot` 字段存储为 **xyzw** 格式
- **加载时转换**: `active_adaptation/utils/twist_motion.py:337`
  ```python
  root_rot = np.array(motion_data["root_rot"])[:, [3, 0, 1, 2]]   # [T, 4] wxyz
  ```
  **转换说明**: PKL文件原始存储xyzw，加载时通过索引 `[3,0,1,2]` 转换为 **wxyz**

**结论**: ✅ **一致** - 两种格式最终都使用 **wxyz** 格式

---

### ✅ 2. 关节顺序：一致 (29关节)

#### NPZ格式 (HDMI任务数据)
- **关节数量**: 29个关节
- **关节顺序** (`data/motion/g1/omomo/sub1_suitcase_011/meta.json`):
  ```python
  0-5:   left_leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
  6-11:  right_leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
  12-14: waist (yaw, roll, pitch)
  15-18: left_arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
  19-21: left_wrist (roll, pitch, yaw)
  22-25: right_arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
  26-28: right_wrist (roll, pitch, yaw)
  ```

#### PKL格式 (TWIST训练数据)
- **关节数量**: 29个关节 (用户确认)
- **关节顺序** (`active_adaptation/utils/motion.py:14-44` - unitree_joint_names):
  ```python
  # 与 NPZ 完全相同的顺序定义
  unitree_joint_names = [
    "left_hip_pitch_joint",  # 0
    "left_hip_roll_joint",   # 1
    ... (完整29个关节)
  ]
  ```

**结论**: ✅ **一致** - 两种格式使用相同的29关节顺序（unitree_joint_names）

---

### ✅ 3. Root位置表示：一致

#### NPZ格式
- **字段**: `body_pos_w[:, 0, :]` - 第0个body是root (pelvis)
- **格式**: `[T, N_bodies, 3]` 其中body_names[0] = "pelvis"

#### PKL格式
- **原始字段**: `root_pos` - `[T, 3]`
- **转换后**: 存储到 `body_pos_w[:, 0, :]` (motion.py:426)
  ```python
  body_pos_w[t, 0] = root_pos[t] + R_root.apply(local_body_pos[t, 0])
  ```

**结论**: ✅ **一致** - PKL的root_pos被正确转换为body_pos_w的第0个body

---

## 关键代码路径

### 1. PKL加载与格式转换
**文件**: `active_adaptation/utils/twist_motion.py`

**关键转换 (lines 336-428)**:
```python
# Line 337: 四元数 xyzw → wxyz
root_rot = np.array(motion_data["root_rot"])[:, [3, 0, 1, 2]]   # wxyz

# Line 338: 使用 unitree_joint_names (29关节)
dof_pos = np.array(motion_data["dof_pos"])    # [T, 29]

# Line 417: body_quat_w 全部设置为 wxyz
body_quat_w = np.zeros((T, N_bodies, 4))  # wxyz

# Line 428: Root旋转赋值
body_quat_w[t, b] = root_rot[t]  # wxyz格式
```

### 2. NPZ格式定义
**文件**: `active_adaptation/envs/isaac/mujoco.py:272`
```python
body_quat_w = self.mj_data.xquat[self.body_adrs_read] # wxyz
```

**元数据**: `data/motion/g1/omomo/sub1_suitcase_011/meta.json`
```json
{
  "joint_names": [...],  // 29个关节，unitree顺序
  "body_quat_w": "wxyz格式"
}
```

### 3. TWIST命令管理器使用
**文件**: `active_adaptation/envs/mdp/commands/twist/command.py:111-145`
```python
# 自动检测NPZ格式
is_npz_format = False
if isinstance(data_path, str):
    path_obj = Path(data_path)
    if path_obj.is_dir() and (path_obj / "motion.npz").exists():
        is_npz_format = True

if is_npz_format:
    # 使用 MotionDataset (NPZ) - 已确认wxyz格式
    self.dataset = MotionDataset.create_from_path(...)
else:
    # 使用 TwistMotionDataset (PKL) - 加载时转换为wxyz
    self.dataset = TwistMotionDataset.create_from_path(...)
```

---

## 训练vs推理对比

### TWIST训练时 (PKL格式)
1. **数据源**: `/home/ubuntu/DATA2/Dataset-G1/AMASS_G1/accad/*.pkl`
2. **配置**: `cfg/task/G1/twist/0927_twist_teacher_new.yaml:59`
   ```yaml
   command:
     data_path: .../twist_dataset.yaml
   ```
3. **加载流程**:
   - PKL文件 (xyzw) → 加载时转换 (wxyz) → TwistMotionData
   - 29关节 (unitree_joint_names)
   - root_pos → body_pos_w[:, 0]

### HDMI推理时 (NPZ格式)
1. **数据源**: `data/motion/g1/omomo/sub1_suitcase_011/motion.npz`
2. **配置**: `cfg/task/G1/hdmi/move_suitcase_twist_ref.yaml`
   ```yaml
   command:
     data_path: data/motion/g1/omomo/sub1_suitcase_011
   ```
3. **加载流程**:
   - NPZ文件 (已经是wxyz) → 直接加载 → MotionData
   - 29关节 (与PKL相同)
   - body_pos_w[:, 0] = root位置

### Frozen Policy观察构建
**文件**: `active_adaptation/envs/mdp/commands/dual_command_manager.py:181-213`
```python
# CRITICAL: 临时替换 env.command_manager 为 TWIST manager
actual_env = getattr(self.env, 'base_env', self.env)
original_command_manager = actual_env.command_manager
actual_env.command_manager = self.command_manager  # TWIST manager

try:
    # 创建 TWIST 观察函数（使用正确的数据格式）
    self.obs_functions["proprio_history_combined"] = ...
    self.obs_functions["ref_motion_windowed"] = ...
finally:
    # 恢复原始 command manager
    actual_env.command_manager = original_command_manager
```

---

## 最终结论

### ✅ 所有格式检查通过

| 检查项 | 训练格式 (PKL) | 推理格式 (NPZ) | 一致性 |
|--------|----------------|----------------|--------|
| **四元数格式** | wxyz (加载时转换) | wxyz (原生) | ✅ 一致 |
| **关节数量** | 29 | 29 | ✅ 一致 |
| **关节顺序** | unitree_joint_names | unitree_joint_names | ✅ 一致 |
| **Root位置** | body_pos_w[:, 0] | body_pos_w[:, 0] | ✅ 一致 |

### 为什么可以安全使用不同格式？

1. **统一的数据抽象层**:
   - PKL → `TwistMotionDataset` → `TwistMotionData`
   - NPZ → `MotionDataset` → `MotionData`
   - 两者都提供相同的字段接口 (body_pos_w, body_quat_w, joint_pos等)

2. **加载时自动转换**:
   - PKL在加载时 (`twist_motion.py:337`) 自动转换四元数格式
   - 最终内存中的数据格式完全一致

3. **兼容性字段**:
   - `MotionData` 添加了 `root_lin_vel_w`, `root_ang_vel_w` 字段
   - 确保 TWIST command manager 可以正常访问所需字段

4. **命令管理器隔离**:
   - TWIST frozen policy 使用独立的 `TwistMotionTracking` command manager
   - 通过 `TwistObservationAdapter` 临时切换确保观察计算正确

---

## 潜在问题与建议

### ⚠️ 注意事项

1. **数据内容不同**:
   - 虽然**格式一致**，但**内容不同**
   - PKL: AMASS通用运动数据 (crouch, walk, hop等)
   - NPZ: HDMI任务特定数据 (推箱子任务)
   - **这是正常的** - frozen policy提供通用运动先验，residual policy学习任务特定调整

2. **Body数量可能不同**:
   - PKL的body数量取决于 `local_body_pos` 形状
   - NPZ的body数量在meta.json中定义
   - 当前代码假设body_names匹配 - 如不匹配可能导致索引错误

### ✅ 建议验证步骤

1. **加载测试**:
   ```bash
   # 测试TWIST frozen policy能否正确加载NPZ数据
   python scripts/play.py \
       algo=ppo_roa_finetune \
       task=G1/hdmi/move_suitcase_twist_ref \
       checkpoint_path=run:<twist_checkpoint>
   ```

2. **观察维度检查**:
   - 检查 `ref_motion_windowed` 观察维度
   - 预期: `3 + 6 + 29 = 38` (root_pos + root_ori_6d + joint_pos)
   - 如果是32，说明关节数不对

3. **运行时验证**:
   ```python
   # 在训练脚本中添加
   print(f"TWIST obs shape: {twist_obs.shape}")
   print(f"TWIST ref_motion shape: {twist_manager.get_ref_motion().shape}")
   ```

---

## 相关文件清单

### 核心代码
- `active_adaptation/utils/twist_motion.py` - PKL加载与格式转换
- `active_adaptation/utils/motion.py` - NPZ加载与MotionData定义
- `active_adaptation/envs/mdp/commands/twist/command.py` - TWIST命令管理器
- `active_adaptation/envs/mdp/commands/dual_command_manager.py` - 双命令管理器

### 配置文件
- `cfg/task/G1/twist/0927_twist_teacher_new.yaml` - TWIST训练配置
- `cfg/task/G1/twist/twist_dataset.yaml` - TWIST PKL数据集列表
- `cfg/task/G1/hdmi/move_suitcase_twist_ref.yaml` - HDMI+TWIST推理配置

### 数据文件
- PKL训练数据: `/home/ubuntu/DATA2/Dataset-G1/AMASS_G1/accad/*.pkl`
- NPZ推理数据: `data/motion/g1/omomo/sub1_suitcase_011/motion.npz`

---

**检查日期**: 2025-11-07
**检查人**: Claude Code
**结论**: ✅ **格式完全一致，可以安全使用TWIST frozen policy作为reference**
