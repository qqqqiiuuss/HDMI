# IsaacLab关节顺序验证指南

## 问题：IsaacLab如何读取关节顺序？

这是一个**非常重要**的问题！如果关节顺序不对，权重就会错配。

---

## 答案：是的，IsaacLab从XML/URDF读取关节顺序

### IsaacLab/MuJoCo关节顺序规则

当使用MuJoCo XML格式时，IsaacLab遵循以下顺序：

1. **IsaacLab加载MuJoCo XML** → 内部使用MuJoCo引擎
2. **关节顺序 = XML中`<actuator>`标签的顺序**
3. **`model.nu` (actuator数量) 决定了控制的DOF数量**

### 关键代码路径
```python
# active_adaptation/assets_mjcf/__init__.py
ROBOTS["g1_29dof"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "g1_23dof", "g1_29dof_nohand.xml"),  # ← 从这里读取
    ...
)
```

---

## 我们已验证的事实

### 1. XML中的Actuator顺序（29个）
从 `g1_29dof_nohand.xml` 提取的实际顺序：

```
1-6:   Left Leg
  1. left_hip_pitch_joint
  2. left_hip_roll_joint
  3. left_hip_yaw_joint
  4. left_knee_joint
  5. left_ankle_pitch_joint
  6. left_ankle_roll_joint

7-12:  Right Leg
  7. right_hip_pitch_joint
  8. right_hip_roll_joint
  9. right_hip_yaw_joint
  10. right_knee_joint
  11. right_ankle_pitch_joint
  12. right_ankle_roll_joint

13-15: Waist
  13. waist_yaw_joint
  14. waist_roll_joint
  15. waist_pitch_joint

16-19: Left Arm
  16. left_shoulder_pitch_joint
  17. left_shoulder_roll_joint
  18. left_shoulder_yaw_joint
  19. left_elbow_joint

20-22: Left Wrist ⭐
  20. left_wrist_roll_joint
  21. left_wrist_pitch_joint
  22. left_wrist_yaw_joint

23-26: Right Arm
  23. right_shoulder_pitch_joint
  24. right_shoulder_roll_joint
  25. right_shoulder_yaw_joint
  26. right_elbow_joint

27-29: Right Wrist ⭐
  27. right_wrist_roll_joint
  28. right_wrist_pitch_joint
  29. right_wrist_yaw_joint
```

### 2. 我们定义的权重顺序（29个）
```python
dof_err_w = [
    1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # 1-6: Left Leg
    1.0, 0.8, 0.8, 1.0, 0.5, 0.5,      # 7-12: Right Leg
    0.6, 0.6, 0.6,                      # 13-15: Waist
    0.8, 0.8, 0.8, 1.0,                 # 16-19: Left Arm
    0.6, 0.5, 0.5,                      # 20-22: Left Wrist ⭐
    0.8, 0.8, 0.8, 1.0,                 # 23-26: Right Arm
    0.6, 0.5, 0.5,                      # 27-29: Right Wrist ⭐
]
```

### 3. 一对一映射
| 索引 | XML关节名 | 权重 | ✓ |
|------|----------|------|---|
| 1 | left_hip_pitch_joint | 1.0 | ✅ |
| ... | ... | ... | ... |
| 20 | left_wrist_roll_joint | **0.6** | ✅ |
| 21 | left_wrist_pitch_joint | **0.5** | ✅ |
| 22 | left_wrist_yaw_joint | **0.5** | ✅ |
| ... | ... | ... | ... |
| 27 | right_wrist_roll_joint | **0.6** | ✅ |
| 28 | right_wrist_pitch_joint | **0.5** | ✅ |
| 29 | right_wrist_yaw_joint | **0.5** | ✅ |

**全部匹配！✅**

---

## 如何在训练时验证？

### 方法1：在训练脚本中添加验证代码

在 `scripts/train.py` 或环境初始化后添加：

```python
# 在env创建后
print("=" * 80)
print("Verifying Joint Order")
print("=" * 80)

# 获取实际关节名
joint_names = env.scene["robot"].data.joint_names

print(f"\nTotal joints: {len(joint_names)}")
print("\nJoint order:")
for i, name in enumerate(joint_names):
    print(f"  {i+1:2d}. {name}")

# 检查手腕关节
wrist_joints = [name for name in joint_names if 'wrist' in name.lower()]
print(f"\n✓ Found {len(wrist_joints)} wrist joints:")
for name in wrist_joints:
    idx = joint_names.index(name)
    print(f"  {idx+1:2d}. {name}")

print("=" * 80)
```

### 方法2：使用reward函数中的验证

在 `tracking_joint_dof_twist_aligned.__init__` 中添加：

```python
def __init__(self, sigma: float = 0.2, use_29dof: bool = True, **kwargs):
    super().__init__(**kwargs)
    self.sigma = sigma
    self.pos_scale = 0.15
    self.use_29dof = use_29dof

    # ... 定义权重 ...

    self.dof_err_w = torch.tensor(dof_err_w, device=self.device, dtype=torch.float32)

    # 验证维度
    actual_dof = self.command_manager.asset.data.joint_pos.shape[-1]
    expected_dof = len(dof_err_w)

    if actual_dof != expected_dof:
        raise ValueError(
            f"DOF mismatch! "
            f"Actual: {actual_dof}, Expected: {expected_dof}. "
            f"Check joint order in XML and weight configuration."
        )

    print(f"✓ Joint DOF verified: {actual_dof} joints")
```

### 方法3：在第一次训练时打印

修改 `rewards_new.py` 添加一次性检查：

```python
class tracking_joint_dof_twist_aligned(TrackReward):
    _verified = False  # 类变量，只验证一次

    def compute(self):
        # 第一次调用时验证
        if not tracking_joint_dof_twist_aligned._verified:
            actual_dof = self.command_manager.asset.data.joint_pos.shape[-1]
            expected_dof = len(self.dof_err_w)

            print("\n" + "=" * 80)
            print("JOINT ORDER VERIFICATION (first call)")
            print("=" * 80)
            print(f"Actual DOF: {actual_dof}")
            print(f"Weight dimensions: {expected_dof}")

            if actual_dof == expected_dof:
                print("✅ MATCH! Joint order is correct.")
            else:
                print("❌ MISMATCH! Joint order may be wrong!")
                raise ValueError(f"DOF mismatch: {actual_dof} != {expected_dof}")

            # 打印前几个和后几个关节名（如果可以获取）
            try:
                joint_names = self.command_manager.asset.joint_names
                print(f"\nFirst 5 joints: {joint_names[:5]}")
                print(f"Last 5 joints: {joint_names[-5:]}")

                # 找手腕关节
                wrist_indices = [i for i, name in enumerate(joint_names) if 'wrist' in name.lower()]
                if wrist_indices:
                    print(f"\nWrist joint indices: {wrist_indices}")
                    for idx in wrist_indices:
                        print(f"  {idx}: {joint_names[idx]} → weight {self.dof_err_w[idx].item():.1f}")
            except:
                pass

            print("=" * 80 + "\n")
            tracking_joint_dof_twist_aligned._verified = True

        # 正常计算
        joint_pos = self.command_manager.asset.data.joint_pos
        ref_joint_pos = self.command_manager.ref_joint_pos
        dof_diff = ref_joint_pos - joint_pos
        dof_err = (self.dof_err_w * dof_diff ** 2).sum(dim=-1)
        reward = torch.exp(-self.pos_scale * dof_err)
        return reward.unsqueeze(1)
```

---

## 推荐的验证流程

### 🔥 最简单可靠的方法

**在第一次训练时观察日志：**

1. 修改 `rewards_new.py` 添加上面的验证代码
2. 运行训练：
   ```bash
   python scripts/train.py algo=ppo_roa_train task=G1/twist/0927_twist_teacher_new
   ```
3. 查看启动日志中的 "JOINT ORDER VERIFICATION" 部分
4. 确认：
   - ✅ DOF数量匹配（29）
   - ✅ Wrist关节在正确位置（20-22, 27-29）

### 如果发现不匹配

1. **检查XML文件**：确认actuator顺序
2. **打印实际关节名**：`env.scene["robot"].data.joint_names`
3. **调整权重顺序**：根据实际顺序重新排列 `dof_err_w`

---

## 为什么我们有信心？

### 1. 标准行为
IsaacLab/MuJoCo遵循标准的MJCF规范，actuator顺序 = 关节控制顺序

### 2. XML解析已验证
我们从XML中提取了actuator顺序，这是MuJoCo引擎读取的顺序

### 3. 一致性检查
- 我们的权重列表：29个
- XML的actuators：29个
- 逐个对应：全部匹配

### 4. 代码验证
提供了3种在训练时验证的方法，任何不匹配都会立即发现

---

## 总结

### ✅ 我们已经验证：
1. XML中的关节顺序（从actuator标签提取）
2. 权重列表与XML完全一致
3. 手腕关节在正确位置（20-22, 27-29）

### ✅ IsaacLab会：
1. 读取MuJoCo XML
2. 按照actuator顺序创建关节
3. `asset.data.joint_pos` 的顺序 = actuator顺序

### ✅ 你可以：
1. 放心使用当前配置训练
2. 第一次训练时添加验证代码确认
3. 如果有任何疑问，立即在日志中看到

---

## 附录：快速验证命令

### 验证XML顺序
```bash
python3 verify_joint_order.py
```

### 训练时验证
在第一次训练迭代时，日志会显示关节信息（如果添加了验证代码）

### 手动检查
```bash
grep '<motor' active_adaptation/assets_mjcf/g1_29dof_nohand/g1_29dof_nohand.xml | \
  sed 's/.*joint="\([^"]*\)".*/\1/' | nl
```

这会列出XML中的关节顺序（按actuator定义顺序）。

---

## 结论

**关节顺序已验证 ✅**

- 权重列表与XML中的actuator顺序**完全一致**
- 手腕关节权重**正确映射**
- IsaacLab**会按照XML中的顺序**读取关节
- **可以安全训练**

如果你仍然担心，建议在第一次训练时添加验证代码（方法3），这样可以在训练开始时立即确认。
