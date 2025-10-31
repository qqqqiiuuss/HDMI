# 🔧 Quick Fix: perturb_body_mass 冲突修复

## 🐛 错误信息
```
ValueError: Multiple matches for 'pelvis': '.*' and 'pelvis'!
```

## 🔍 根本原因

**Base配置** (`twist-base.yaml` line 169):
```yaml
randomization:
  perturb_body_mass: {.*: [0.9, 1.1]}  # 匹配所有body
```

**子配置** (`0927_twist_teacher_new.yaml` line 263):
```yaml
randomization:
  perturb_body_mass:
    pelvis: [0.7, 1.3]  # 精确匹配pelvis
```

**冲突**: `pelvis`同时被`.*`（通配符）和`pelvis`（精确匹配）匹配，导致重复。

---

## ✅ 修复方案

### 方案1: 完全覆盖（已采用）
在子配置中完全重写`perturb_body_mass`，列出所有需要随机化的body：

```yaml
randomization:
  perturb_body_mass:
    pelvis: [0.7, 1.3]                    # 主要质量变化
    torso_link: [0.95, 1.05]              # 其他部位轻微变化
    left_hip_pitch_link: [0.95, 1.05]
    right_hip_pitch_link: [0.95, 1.05]
    left_knee_link: [0.95, 1.05]
    right_knee_link: [0.95, 1.05]
    left_ankle_roll_link: [0.95, 1.05]
    right_ankle_roll_link: [0.95, 1.05]
```

**优点**:
- ✅ 精确控制每个body的质量范围
- ✅ 符合TWIST原版只随机化pelvis的主要质量
- ✅ 避免配置合并冲突

---

### 方案2: 禁用质量随机化（备选）
如果暂时不需要质量随机化，可以注释掉：

```yaml
randomization:
  # perturb_body_mass:
  #   pelvis: [0.7, 1.3]
```

**适用场景**: 调试阶段，简化变量

---

### 方案3: 修改Base配置（不推荐）
修改`twist-base.yaml`，去掉通配符：

```yaml
# twist-base.yaml
randomization:
  # perturb_body_mass: {.*: [0.9, 1.1]}  # 注释掉
```

**缺点**: 影响所有继承该base的配置

---

## 🧪 验证修复

```bash
cd /home/ubuntu/DATA2/workspace/xmh/tmp/HDMI
python scripts/train.py \
    algo=ppo \
    task=G1/twist/0927_twist_teacher_new \
    task.num_envs=6 \
    suffix=test_fix
```

**预期输出**: 无错误，环境正常创建

---

## 📝 Hydra配置合并规则

### 默认行为（导致问题）
```yaml
# base.yaml
config:
  .*: value1

# child.yaml
config:
  specific_key: value2

# 合并结果（错误！）
config:
  .*: value1           # 来自base
  specific_key: value2  # 来自child
  # specific_key同时被.*和自己匹配 → 冲突！
```

### 正确做法：完全覆盖
```yaml
# child.yaml
config:
  specific_key: value2
  other_key: value1
  # 不使用.*，显式列出所有key
```

---

## 🎯 关键点总结

1. ❌ **不要在子配置中添加到包含通配符的配置**
2. ✅ **完全覆盖包含通配符的配置**
3. ⚠️ **通配符`.*`会匹配所有名称，容易产生冲突**
4. 💡 **IsaacLab的`resolve_matching_names_values`会检测重复匹配**

---

## 🔗 相关代码

**错误来源**:
- `active_adaptation/envs/mdp/randomizations.py` line 349
- `isaaclab/utils/string.py` line 323 (`resolve_matching_names_values`)

**检查逻辑**:
```python
# IsaacLab会验证每个匹配的名称只能被一个pattern匹配
if len(matched_patterns) > 1:
    raise ValueError(f"Multiple matches for '{name}': {matched_patterns}!")
```

这是一个好的设计，防止配置歧义！
