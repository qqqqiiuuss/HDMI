# HDMI Residual Action机制详细分析

## 🎯 核心发现

**在HDMI (move_suitcase等任务)中,确实存在"基于参考关节位置的Residual Action"机制!**

但这与我之前分析的"基于default_joint_pos的residual action"不同。这里是**基于参考运动(reference motion)的residual action**。

---

## 📊 完整数据流分析

### 1. 配置层面

**位置**: `cfg/task/base/hdmi-base.yaml:192-193`

```yaml
observation:
  priv:
    # 参考关节位置观察
    ref_joint_pos_:
      ref_joint_pos_action_policy: {}  # 参考关节位置动作策略
```

**关键点**:
- `ref_joint_pos_` 是observation的一个**命名空间**
- `ref_joint_pos_action_policy` 是具体的observation函数
- 这个observation提供"参考运动中的关节位置,转换为策略动作空间"

---

### 2. Observation实现

**位置**: `active_adaptation/envs/mdp/commands/hdmi/observations.py:76-99`

```python
class ref_joint_pos_action_policy(RobotTrackObservation):
    """
    参考关节位置策略观察函数

    返回经过动作缩放处理的参考关节位置,用于策略网络训练。
    将参考关节位置转换为策略动作空间。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 获取动作管理器
        action_manager = self.env.action_manager
        action_joint_names = action_manager.joint_names

        # 找到动作关节在运动数据中的索引
        self.action_indices_motion = [
            self.command_manager.dataset.joint_names.index(joint_name)
            for joint_name in action_joint_names
        ]

        # 获取动作缩放参数和默认关节位置
        self.action_scaling = action_manager.action_scaling
        self.default_joint_pos = action_manager.default_joint_pos[:, action_manager.joint_ids]

    def compute(self):
        # 获取参考运动中的关节位置
        ref_joint_pos = self.command_manager.current_ref_motion.joint_pos[:, self.action_indices_motion]

        # ← 关键转换: 将参考关节位置转换为"策略action空间"
        ref_joint_action = (ref_joint_pos - self.default_joint_pos) / self.action_scaling

        return ref_joint_action
```

**数学公式**:
```
ref_joint_action = (ref_joint_pos - default_joint_pos) / action_scaling
```

**具体例子** (某个膝关节):
- `ref_joint_pos[knee]` = 0.6 rad (参考运动中的目标位置)
- `default_joint_pos[knee]` = 0.0 rad (站立姿态)
- `action_scaling[knee]` = 0.5

计算:
```
ref_joint_action[knee] = (0.6 - 0.0) / 0.5 = 1.2
```

**关键洞察**:
- ✅ 这个observation提供的是"如果policy想达到ref_joint_pos,应该输出的action值"
- ✅ 它是参考运动位置的"反向缩放"
- ✅ 单位是"策略action空间"(无量纲),不是rad

---

### 3. PPO-ROA/PPO-AMP中的Residual Module

**位置**: `active_adaptation/learning/ppo/ppo_roa.py:223-236`

```python
# REF_JPOS_KEY = "ref_joint_pos_"  (定义在line 35)

if cfg.phase == "train" and cfg.enable_residual_distillation:
    # 训练阶段 + 启用residual distillation
    assert REF_JPOS_KEY in observation_spec, f"{REF_JPOS_KEY} should be in observation_spec"

    class RefJointPos(nn.Module):
        """将policy输出的action与参考关节action相加"""
        def forward(self, ref_jpos, action):
            return (ref_jpos + action,)  # ← 关键: 相加!

    residual_module_cls = RefJointPos
else:
    # 测试阶段 或 禁用residual distillation
    class DummyRefJointPos(nn.Module):
        """直接返回policy的action,不加参考"""
        def forward(self, ref_jpos, action):
            return action  # ← 不相加

    residual_module_cls = DummyRefJointPos

# 构建TensorDict模块
in_keys = [REF_JPOS_KEY, "loc"]  # 输入: ref_joint_pos_, loc (policy输出)
out_keys = ["loc"]                # 输出: loc (最终action)
residual_module = Mod(residual_module_cls(), in_keys, out_keys)
```

**数据流**:
```
TensorDict:
  "ref_joint_pos_" -> ref_joint_action (来自observation)
  "loc"            -> policy输出的action均值

RefJointPos.forward(ref_joint_action, loc):
  return ref_joint_action + loc

输出:
  "loc" -> final_action = ref_joint_action + policy_residual
```

---

### 4. Actor网络结构

**位置**: `active_adaptation/learning/ppo/ppo_roa.py:238-260`

```python
def build_actor(in_keys, dist_cls, dist_keys, residual_module=None):
    actor_modules = [
        # Step 1: 拼接输入 (cmd, obs, priv_feature)
        CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),

        # Step 2: MLP backbone
        Mod(make_mlp([512, 256, 256]), ["_actor_inp"], ["_actor_feature"]),

        # Step 3: Actor head (输出loc和scale)
        Mod(Actor(action_dim, ...), ["_actor_feature"], dist_keys)
        # 输出: "loc" (action均值), "scale" (action标准差)
    ]

    # Step 4: 如果启用residual,添加residual module
    if residual_module is not None:
        actor_modules.append(residual_module)
        # 这会把 "loc" 从 policy_output 变为 (ref_joint_action + policy_output)

    actor_module = Seq(*actor_modules)

    # Step 5: 包装为概率actor
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=dist_keys,  # ["loc", "scale"]
        out_keys=[ACTION_KEY],  # ["action"]
        distribution_class=dist_cls,  # IndependentNormal
        return_log_prob=True
    )
    return actor
```

**完整前向传播**:
```
输入: TensorDict {
    "cmd": [...],
    "obs": [...],
    "priv_feature": [...],
    "ref_joint_pos_": [1.2, -0.5, ...]  ← 参考关节action
}

↓ Step 1: CatTensors
["_actor_inp"] = concat([cmd, obs, priv_feature])

↓ Step 2: MLP
["_actor_feature"] = MLP([cmd, obs, priv_feature])

↓ Step 3: Actor
["loc"] = 0.1        ← policy输出的residual (很小!)
["scale"] = 0.5

↓ Step 4: RefJointPos (如果启用)
["loc"] = ref_joint_pos_ + loc
        = 1.2 + 0.1 = 1.3  ← 最终action均值

↓ Step 5: ProbabilisticActor
["action"] ~ Normal(loc=1.3, scale=0.5)
        = 1.3 + 0.5 * randn()
        ≈ 1.4  (采样结果)
```

---

### 5. Action Manager应用

**位置**: `active_adaptation/envs/mdp/action.py:117-119`

```python
def __call__(self, action: torch.Tensor, substep: int):
    # ... (delay和平滑处理)

    # 最终应用
    pos_target = self.default_joint_pos + self.offset
    pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
    self.asset.set_joint_position_target(pos_target)
```

**数学**:
```
pos_target = default_joint_pos + offset + (final_action × action_scaling)
           = default_joint_pos + offset + ((ref_joint_action + policy_residual) × action_scaling)
```

**展开计算** (某个膝关节):

假设:
- `ref_joint_pos` = 0.6 rad (参考运动中的目标)
- `default_joint_pos` = 0.0 rad
- `action_scaling` = 0.5
- `offset` = 0.005 rad
- `policy_residual` = 0.1 (policy输出的微调)

计算:
```
# 1. 参考关节action (在observation中计算)
ref_joint_action = (ref_joint_pos - default_joint_pos) / action_scaling
                 = (0.6 - 0.0) / 0.5
                 = 1.2

# 2. 最终action (在RefJointPos中计算)
final_action = ref_joint_action + policy_residual
             = 1.2 + 0.1
             = 1.3

# 3. 目标位置 (在Action Manager中计算)
pos_target = default_joint_pos + offset + (final_action × action_scaling)
           = 0.0 + 0.005 + (1.3 × 0.5)
           = 0.0 + 0.005 + 0.65
           = 0.655 rad
```

**验证**:
如果没有policy residual (policy_residual = 0):
```
final_action = 1.2 + 0 = 1.2
pos_target = 0.0 + 0.005 + (1.2 × 0.5) = 0.605 rad
```

如果ref_joint_pos = 0.6 rad,那么:
```
理想位置 = ref_joint_pos + offset = 0.6 + 0.005 = 0.605 rad  ✓ 一致!
```

**关键洞察**:
- ✅ **当policy_residual = 0时,机器人完美跟踪参考运动**
- ✅ **policy_residual是对参考运动的微调/偏移**
- ✅ **这就是"Residual Learning"的本质**

---

## 🔍 两种Residual Action对比

### Residual Type 1: 基于Default Pose (TWIST)

```python
# Policy直接输出action
action = policy(obs)  # 例如: 1.2

# 目标位置
pos_target = default_joint_pos + action × scaling
           = 0.0 + 1.2 × 0.5
           = 0.6 rad
```

**特点**:
- ✅ Policy学习的是"相对于默认姿态的偏移"
- ✅ 适用于没有参考运动的任务 (如locomotion)
- ✅ TWIST使用这种方式

### Residual Type 2: 基于Reference Motion (HDMI)

```python
# 参考运动提供基准action
ref_joint_action = (ref_joint_pos - default_joint_pos) / scaling
                 = (0.6 - 0.0) / 0.5
                 = 1.2

# Policy输出residual (微调)
policy_residual = policy(obs)  # 例如: 0.1

# 最终action
final_action = ref_joint_action + policy_residual
             = 1.2 + 0.1
             = 1.3

# 目标位置
pos_target = default_joint_pos + final_action × scaling
           = 0.0 + 1.3 × 0.5
           = 0.65 rad
```

**特点**:
- ✅ Policy学习的是"相对于参考运动的微调"
- ✅ 适用于有参考运动的任务 (如motion tracking, imitation)
- ✅ HDMI使用这种方式
- ✅ Policy只需学习小的修正,训练更快更稳定

---

## 📋 HDMI配置详解

### 在哪些任务中启用?

**启用Residual Distillation的配置**:
- `cfg/task/base/hdmi-base.yaml` → `move_suitcase.yaml`继承
- `cfg/task/base/hdmi-base-noobj.yaml`
- `cfg/task/base/tracking-base.yaml`
- `cfg/task/base/hdmi-hardware.yaml`

**关键配置**:
```yaml
# algo配置 (在Python代码中)
algo:
  phase: "train"  # 训练阶段
  enable_residual_distillation: true  # 启用residual distillation

# observation配置
observation:
  priv:
    ref_joint_pos_:
      ref_joint_pos_action_policy: {}  # ← 必须有这个!
```

### TWIST配置对比

**TWIST** (`cfg/task/base/twist-base.yaml`):
```yaml
observation:
  policy:
    # TWIST使用ref_motion_windowed,但不提供ref_joint_pos_
    ref_motion_windowed:
      past_frames: 10
      future_frames: 10
    # ❌ 没有 ref_joint_pos_action_policy
```

**结论**:
- ✅ **TWIST不使用Residual Learning (基于参考运动)**
- ✅ **TWIST的policy直接输出完整的action**
- ✅ **HDMI使用Residual Learning,policy只输出微调**

---

## 🎓 Residual Distillation的优势

### 1. 训练稳定性

**HDMI (with residual)**:
```python
policy需要学习: policy_residual ≈ 0.1 (小值)
action范围: [-0.5, +0.5] (小范围)
```

**TWIST (without residual)**:
```python
policy需要学习: action ≈ 1.2 (大值)
action范围: [-2.0, +2.0] (大范围)
```

**优势**:
- ✅ **小值更容易学习**: policy初始时输出接近0,已经接近最优
- ✅ **更小的探索空间**: 只需微调,不需要从头学习完整动作
- ✅ **更快收敛**: 初始性能已经很好 (跟踪参考运动)

### 2. 数据效率

**HDMI**:
- 参考运动提供了"正确动作"的先验
- Policy只需学习如何应对扰动、适应环境

**TWIST**:
- Policy需要从零学习完整的运动模式
- 需要更多数据才能收敛

### 3. Generalization

**HDMI**:
- 对于新的参考运动,policy可以立即获得合理的初始性能
- 微调能力可以迁移到新任务

**TWIST**:
- 对于新的运动模式,需要重新训练

---

## 💡 实现细节

### 何时启用Residual?

**条件1**: 算法配置
```python
# ppo_roa.py:223
if cfg.phase == "train" and cfg.enable_residual_distillation:
    # 使用RefJointPos (相加)
else:
    # 使用DummyRefJointPos (不相加)
```

**条件2**: Observation必须包含`ref_joint_pos_`
```python
assert REF_JPOS_KEY in observation_spec
# REF_JPOS_KEY = "ref_joint_pos_"
```

### Teacher vs Student

**Teacher (训练阶段)**:
```python
# ppo_roa.py:260
self.actor = build_actor(
    in_keys=[CMD_KEY, OBS_KEY, "priv_feature"],
    residual_module=residual_module  # ← 启用residual
)
```

**Student (部署阶段)**:
```python
# ppo_roa.py:276-277
self.actor_adapt = build_actor(
    in_keys=[CMD_KEY, OBS_KEY, "priv_pred"],
    # residual_module=None  ← 默认不使用residual
)
```

**为什么Student不用Residual?**
- ❌ Student在部署时没有参考运动数据
- ❌ `ref_joint_pos_`是privileged information,真实硬件没有
- ✅ Student必须学会完全自主生成action

**Training策略**:
1. **Phase 1 (train)**: Teacher用residual学习微调
2. **Phase 2 (distill)**: Student模仿Teacher的完整输出 (ref + residual)
3. **Phase 3 (finetune)**: Student在没有参考的情况下fine-tune

---

## 📊 完整示例

### 场景: move_suitcase任务训练

**配置**:
```yaml
# move_suitcase.yaml继承hdmi-base.yaml
observation:
  priv:
    ref_joint_pos_:
      ref_joint_pos_action_policy: {}

# 算法 (PPO-ROA)
algo: ppo_roa_train  # phase="train", enable_residual_distillation=true
```

**运行时数据流**:

```python
# 1. 环境step
ref_motion = dataset.get_motion(motion_id)
ref_joint_pos = ref_motion.joint_pos  # [0.6, -0.3, 0.8, ...]

# 2. Observation计算
ref_joint_action = (ref_joint_pos - default_joint_pos) / action_scaling
                 = ([0.6, -0.3, 0.8] - [0.0, 0.0, 0.0]) / [0.5, 0.5, 0.5]
                 = [1.2, -0.6, 1.6]

obs = TensorDict({
    "cmd": [...],
    "obs": [...],
    "priv_feature": [...],
    "ref_joint_pos_": [1.2, -0.6, 1.6]  # ← 参考action
})

# 3. Policy forward
policy_residual = actor.get_distribution(obs).mean
                = [0.1, -0.05, 0.08]  # ← policy学习的微调

# 4. RefJointPos module
final_action = ref_joint_action + policy_residual
             = [1.2, -0.6, 1.6] + [0.1, -0.05, 0.08]
             = [1.3, -0.65, 1.68]

# 5. Action Manager
pos_target = default_joint_pos + offset + final_action × action_scaling
           = [0.0, 0.0, 0.0] + [0.005, 0.005, 0.005] + [1.3, -0.65, 1.68] × [0.5, 0.5, 0.5]
           = [0.0, 0.0, 0.0] + [0.005, 0.005, 0.005] + [0.65, -0.325, 0.84]
           = [0.655, -0.32, 0.845]

# 验证: 如果policy_residual = 0
pos_target_ideal = [0.0, 0.0, 0.0] + [0.005, 0.005, 0.005] + [0.6, -0.3, 0.8]
                 = [0.605, -0.295, 0.805]
# 与ref_joint_pos + offset一致! ✓
```

---

## 🔑 关键要点总结

### 1. HDMI的Residual Action机制

```
最终action = ref_joint_action + policy_residual
            ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
            来自参考运动       policy学习的微调
```

### 2. 数学公式完整推导

```
# Step 1: Observation计算
ref_joint_action = (ref_joint_pos - default_joint_pos) / action_scaling

# Step 2: Policy输出
policy_residual = actor(obs)

# Step 3: RefJointPos合并
final_action = ref_joint_action + policy_residual

# Step 4: Action Manager应用
pos_target = default_joint_pos + offset + (final_action × action_scaling)

# 展开:
pos_target = default_joint_pos + offset
           + ((ref_joint_pos - default_joint_pos)/scaling + policy_residual) × scaling
           = default_joint_pos + offset
           + (ref_joint_pos - default_joint_pos) + policy_residual × scaling
           = ref_joint_pos + offset + policy_residual × scaling
```

### 3. 关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| **Observation实现** | `active_adaptation/envs/mdp/commands/hdmi/observations.py` | 76-99 |
| **RefJointPos模块** | `active_adaptation/learning/ppo/ppo_roa.py` | 223-236 |
| **Actor构建** | `active_adaptation/learning/ppo/ppo_roa.py` | 238-260 |
| **Action应用** | `active_adaptation/envs/mdp/action.py` | 117-119 |
| **配置** | `cfg/task/base/hdmi-base.yaml` | 192-193 |

### 4. HDMI vs TWIST

| 维度 | TWIST | HDMI |
|------|-------|------|
| **Residual类型** | 基于default pose | 基于reference motion |
| **Policy输出** | 完整action | 微调residual |
| **初始性能** | 需要探索 | 已接近最优 |
| **训练速度** | 较慢 | 较快 |
| **数据效率** | 较低 | 较高 |
| **部署** | 直接使用 | 需要distillation到student |

---

## 🚀 使用建议

### 何时使用Residual Learning?

**适用场景**:
- ✅ 有高质量参考运动数据 (human demos, motion capture)
- ✅ 需要精确跟踪特定运动模式
- ✅ 希望加快训练速度
- ✅ 数据有限,需要提高样本效率

**不适用场景**:
- ❌ 没有参考运动数据
- ❌ 需要学习完全新的运动模式
- ❌ 部署环境无法提供参考 (需要额外的distillation步骤)

### 配置检查清单

训练HDMI任务时,确保:
1. ✅ 配置文件继承自`hdmi-base.yaml`或类似配置
2. ✅ Observation包含`ref_joint_pos_: {ref_joint_pos_action_policy: {}}`
3. ✅ 使用`algo=ppo_roa_train`或`ppo_amp`
4. ✅ `enable_residual_distillation=true` (默认)
5. ✅ 有有效的参考运动数据路径

---

**最后更新**: 2025-11-01
**相关文档**: `ACTION_TYPE_ANALYSIS.md`, `OFFSET_ANALYSIS.md`
