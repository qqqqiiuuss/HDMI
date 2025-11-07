# Frozen TWIST Policy Reference Implementation

## 概述

本文档详细说明了将预训练的TWIST teacher policy作为frozen reference集成到HDMI residual learning训练中的实现。

## 核心思想

**Residual Learning架构**:
```
final_action = frozen_twist_reference + hdmi_student_residual
```

- **Frozen TWIST Policy**: 预训练的locomotion policy,提供稳定的base motion
- **HDMI Student**: 学习在TWIST基础上的residual correction,专注于object interaction

## 修改的文件

### 1. `active_adaptation/learning/ppo/ppo_roa.py`

#### 主要修改点:

##### A. 初始化frozen policy (lines 160-179)
```python
def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
    ...
    # 新增: 初始化frozen policy wrapper
    if self.cfg.use_frozen_policy_ref:
        from active_adaptation.learning.ppo.frozen_policy_wrapper import FrozenPolicyWrapper
        print(f"[PPOROA] Initializing frozen policy reference from {self.cfg.frozen_policy_checkpoint}")
        self.frozen_policy_wrapper = FrozenPolicyWrapper(
            checkpoint_path=self.cfg.frozen_policy_checkpoint,
            device=self.device
        )
        print("[PPOROA] Frozen policy loaded successfully")
    else:
        self.frozen_policy_wrapper = None
```

**作用**: 加载预训练的TWIST checkpoint,保存为frozen policy wrapper

##### B. FrozenRefComputer模块 (lines 200-230)
```python
class FrozenRefComputer(TensorDictModuleBase):
    """Module that computes frozen policy reference during rollout"""
    def __init__(self, ppo_roa):
        super().__init__()
        self.ppo_roa = ppo_roa
        self.in_keys = []
        self.out_keys = ["_frozen_policy_ref"]

    def forward(self, tensordict):
        # Compute frozen reference
        ref_action = self.ppo_roa.compute_frozen_policy_reference(tensordict)
        # Add to tensordict
        tensordict.set("_frozen_policy_ref", ref_action)
        return tensordict

self.frozen_ref_computer = FrozenRefComputer(self).to(self.device)
```

**作用**: 自定义TensorDictModule,在rollout时计算frozen policy的输出并存入tensordict的`_frozen_policy_ref`字段

##### C. FrozenPolicyRefModule残差模块 (lines 295-319)
```python
class FrozenPolicyRefModule(TensorDictModuleBase):
    """Module that adds frozen policy reference to student residual"""
    def __init__(self):
        super().__init__()
        self.in_keys = ["loc"]
        self.out_keys = ["loc"]

    def forward(self, tensordict):
        action = tensordict.get("loc")
        frozen_ref = tensordict.get("_frozen_policy_ref", None)

        if frozen_ref is not None:
            # Residual learning: final_action = frozen_reference + student_residual
            final_action = frozen_ref + action
        else:
            # During initialization, frozen_ref might not be available yet
            final_action = action

        tensordict.set("loc", final_action)
        return tensordict

residual_module = FrozenPolicyRefModule()
```

**作用**:
- 自定义TensorDictModule,直接从tensordict读取`_frozen_policy_ref`
- 执行残差加法: `final_action = frozen_ref + student_residual`
- 处理初始化时frozen_ref可能不存在的情况

##### D. compute_frozen_policy_reference方法 (lines 560-600)
```python
def compute_frozen_policy_reference(self, tensordict: TensorDict):
    """
    Compute frozen policy reference action

    Returns:
        ref_action: [num_envs, action_dim] frozen policy output
    """
    if not self.cfg.use_frozen_policy_ref or self.frozen_policy_wrapper is None:
        return None

    with torch.inference_mode():
        # Build TWIST observations from HDMI data
        twist_obs = self.env.twist_command_manager.build_twist_observations(tensordict)

        # Normalize if needed
        if self.vecnorm is not None:
            twist_obs_normalized = self.vecnorm.normalize_obs(twist_obs)
        else:
            twist_obs_normalized = twist_obs

        # Frozen policy forward pass
        ref_action = self.frozen_policy_wrapper(twist_obs_normalized)

    return ref_action
```

**作用**:
1. 从HDMI的tensordict中提取数据
2. 通过`twist_command_manager`构建TWIST需要的observations
3. 调用frozen policy获取reference action
4. 使用`torch.inference_mode()`确保不计算梯度

##### E. Actor Sequential修改 (lines 335-345)
```python
if self.cfg.use_frozen_policy_ref:
    # Use custom FrozenPolicyRefModule for residual addition
    actor_module = nn.Sequential(
        encoder,
        actor_mlp,
        residual_module,  # FrozenPolicyRefModule
        prob_module
    )
else:
    # Original: RefJointPos module
    actor_module = nn.Sequential(...)
```

**作用**: 将`FrozenPolicyRefModule`插入actor pipeline,在生成分布前添加frozen reference

##### F. Rollout loop修改 (lines 710-715)
```python
def rollout(self, tensordict: TensorDict, deterministic=False, rollout_len=None):
    ...
    for step_i in range(rollout_len):
        # Compute frozen policy reference if enabled
        if self.cfg.use_frozen_policy_ref:
            tensordict = self.frozen_ref_computer(tensordict)

        # Student policy forward (will use _frozen_policy_ref)
        tensordict = self.actor.get_dist(tensordict)
        ...
```

**作用**: 在每个rollout step,先用`FrozenRefComputer`计算并存储frozen reference,然后执行student policy

### 2. `active_adaptation/envs/mdp/commands/twist/observations.py`

#### 主要修改点:

##### A. 修复inference tensor in-place操作错误

多处修改,使用非in-place操作避免在`torch.inference_mode()`下的错误:

**例1: proprio_history_combined.update() (lines 1035-1038)**
```python
# OLD (会失败):
self.history_buffer[:, :-1, :] = self.history_buffer[:, 1:, :].clone()
self.history_buffer[:, -1, :] = current_proprio

# NEW (正确):
self.history_buffer = torch.cat([
    self.history_buffer[:, 1:, :].clone(),  # 移除最旧帧
    current_proprio.unsqueeze(1)            # 添加当前帧
], dim=1)
```

**例2: ref_motion_windowed.update() (lines 1274-1277)**
```python
self.past_buffer = torch.cat([
    self.past_buffer[:, 1:, :].clone(),
    current_ref.unsqueeze(1)
], dim=1)
```

**例3: reset方法中的clone (lines 997-999, 1193-1196)**
```python
def reset(self, env_ids):
    if self._initialized:
        self.history_buffer = self.history_buffer.clone()
        self.history_buffer[env_ids] = 0.0
```

**原因**: 当frozen policy在`torch.inference_mode()`下运行时,所有接触到的tensor都变成inference tensor,不允许in-place修改

### 3. `active_adaptation/envs/mdp/commands/dual_command_manager.py`

#### TwistObservationAdapter (lines 50-120)

```python
class TwistObservationAdapter:
    """Adapts HDMI data to TWIST observation format"""

    def build_twist_observations(self, hdmi_tensordict: TensorDict) -> TensorDict:
        """
        Convert HDMI tensordict to TWIST observation format

        Args:
            hdmi_tensordict: HDMI's tensordict containing motion and robot state

        Returns:
            twist_obs: TensorDict with TWIST observation keys
        """
        twist_obs = TensorDict({}, batch_size=hdmi_tensordict.batch_size, device=self.device)

        # Extract data using TWIST observation functions
        for obs_name, obs_fn in self.twist_obs_functions.items():
            twist_obs[obs_name] = obs_fn(self.env, hdmi_tensordict)

        return twist_obs
```

**作用**: 将HDMI的observation/motion数据转换为TWIST policy期望的格式

### 4. 配置文件

#### `cfg/algo/ppo_roa_train_twist_ref.yaml`
```yaml
defaults:
  - ppo_roa_train

use_frozen_policy_ref: true
frozen_policy_checkpoint: ???  # 需要通过命令行指定
```

#### `cfg/task/G1/hdmi/move_suitcase_twist_ref.yaml`
```yaml
defaults:
  - ../../base/hdmi-base-twist-ref

reference:
  type: "frozen_twist_policy"
  twist_policy:
    checkpoint_path: ???  # 通过命令行覆盖
```

#### `cfg/task/base/hdmi-base-twist-ref.yaml`
```yaml
defaults:
  - hdmi-base

# TWIST observation configuration
twist_observations:
  enabled: true
  groups:
    - proprio_history_combined
    - ref_motion_windowed
    - height_scan
  history_length: 10
  ref_window_past: 3
  ref_window_future: 3
```

## 数据流程

### 1. 初始化阶段
```
1. PPOROA.__init__()
   ↓
2. 加载frozen TWIST checkpoint → FrozenPolicyWrapper
   ↓
3. 创建FrozenRefComputer (rollout时使用)
   ↓
4. 创建FrozenPolicyRefModule (添加residual)
   ↓
5. 构建actor pipeline: encoder → actor_mlp → FrozenPolicyRefModule → prob_module
```

### 2. Rollout阶段
```
每个step:
1. FrozenRefComputer.forward(tensordict)
   ↓
2. compute_frozen_policy_reference()
   - 从tensordict提取HDMI数据
   - twist_command_manager.build_twist_observations() → TWIST obs
   - frozen_policy_wrapper(twist_obs) → ref_action
   ↓
3. tensordict.set("_frozen_policy_ref", ref_action)
   ↓
4. actor.get_dist(tensordict)
   - encoder(obs) → feature
   - actor_mlp(feature) → student_residual (loc)
   - FrozenPolicyRefModule.forward(tensordict):
     * action = tensordict["loc"]  (student residual)
     * frozen_ref = tensordict["_frozen_policy_ref"]
     * final_action = frozen_ref + action
     * tensordict.set("loc", final_action)
   - prob_module(final_action) → distribution
   ↓
5. 采样action → 环境执行
   ↓
6. 存储到rollout buffer
```

### 3. 训练阶段
```
1. train_policy(tensordict)
   - tensordict已包含rollout时存储的_frozen_policy_ref
   - 不需要重新计算frozen reference
   ↓
2. 切分minibatch
   - tensordict自动切分,_frozen_policy_ref也被切分
   ↓
3. _update_ppo(minibatch)
   - actor.get_dist(minibatch)使用已有的_frozen_policy_ref
   - 计算policy loss, update student parameters
   - frozen policy参数保持不变(requires_grad=False)
```

## 关键设计决策

### 1. 为什么用自定义TensorDictModuleBase?

**问题**: TensorDictModule要求所有`in_keys`在tensordict中存在,但`_frozen_policy_ref`是动态添加的

**尝试的方案**:
- ❌ 添加到TensorDictPrimer: `action_spec`在primer创建时不可用
- ❌ 使用`default_interaction_type="random"`: 该参数不存在

**最终方案**:
- 创建`FrozenPolicyRefModule(TensorDictModuleBase)`,直接访问tensordict
- 使用`tensordict.get("_frozen_policy_ref", None)`提供fallback
- 在forward中处理key可能不存在的情况

### 2. 为什么frozen reference只在rollout时计算?

**原因**:
1. **效率**: 避免在training时重复计算frozen reference
2. **正确性**: rollout时计算一次,存入tensordict,训练时直接使用
3. **一致性**: 确保training使用的frozen_ref与rollout采样时完全一致

**实现**:
- Rollout: `FrozenRefComputer`计算并存储到tensordict
- Training: 直接从tensordict读取,不重新计算

### 3. 为什么需要TwistObservationAdapter?

**原因**: TWIST和HDMI使用不同的observation space

| Feature | HDMI | TWIST |
|---------|------|-------|
| Observation keys | `policy`, `command`, `object`, `priv` | `proprio_history`, `ref_motion`, `height_scan` |
| Motion source | SMPL-X retargeted | BVH locomotion data |
| Task focus | Object interaction | Pure locomotion |

**TwistObservationAdapter作用**:
- 从HDMI的`motion_loader`提取相同时刻的motion data
- 调用TWIST observation functions重建TWIST obs
- 处理坐标系转换和数据格式差异

### 4. Inference tensor错误修复

**问题**: `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`

**原因**:
```python
with torch.inference_mode():
    frozen_ref = frozen_policy(twist_obs)  # 所有相关tensor变为inference tensor
    # TWIST obs functions中的in-place操作会失败
```

**解决方案**: 将所有in-place操作改为非in-place:
```python
# 错误:
buffer[:, :-1] = buffer[:, 1:].clone()

# 正确:
buffer = torch.cat([buffer[:, 1:].clone(), new_data.unsqueeze(1)], dim=1)
```

## 已知问题和解决方案

### ✅ 已解决的问题

1. **Circular reference导致递归**: 使用weakref避免
2. **Batch dimension mismatch**: 只在rollout时计算,避免在minibatch slicing时重复计算
3. **Inference tensor in-place错误**: 改用非in-place操作
4. **TensorDictModule missing keys**: 使用自定义TensorDictModuleBase

### ⚠️ 当前状态

**训练能够运行,但reward可能很低,需要检查**:

#### 可能的问题:

1. **Action scale不匹配**
   - TWIST policy输出的action scale可能与HDMI不同
   - 需要检查两者的action normalization是否一致

   **检查方法**:
   ```python
   # 在compute_frozen_policy_reference中添加:
   print(f"Frozen ref action stats: mean={ref_action.mean()}, std={ref_action.std()}, min={ref_action.min()}, max={ref_action.max()}")
   ```

2. **Observation不匹配**
   - TwistObservationAdapter可能没有正确构建TWIST observations
   - Motion timing可能不同步

   **检查方法**:
   ```python
   # 在build_twist_observations中添加:
   print(f"TWIST obs keys: {twist_obs.keys()}")
   print(f"TWIST obs shapes: {[(k, v.shape) for k, v in twist_obs.items()]}")
   ```

3. **Coordinate system不一致**
   - TWIST使用不同的坐标系约定
   - Joint order可能不同

   **检查方法**:
   - 对比TWIST和HDMI的joint_names
   - 检查是否需要joint reordering

4. **Frozen reference太强**
   - Student residual被压制,无法学习
   - 可能需要调整residual weight

   **解决方案**:
   ```python
   # 在FrozenPolicyRefModule中:
   final_action = frozen_ref + self.residual_weight * action
   # residual_weight可以从0.1开始逐渐增加到1.0
   ```

5. **Motion library不匹配**
   - TWIST在locomotion motion上训练
   - HDMI使用object interaction motion
   - Frozen reference可能与当前task不compatible

   **诊断**:
   - 对比frozen_ref和ref_joint_pos的差异
   - 可视化两者的动作轨迹

## 调试命令

### 1. 训练with frozen reference
```bash
python scripts/train.py \
    algo=ppo_roa_train_twist_ref \
    task=G1/hdmi/move_suitcase_twist_ref \
    task.reference.twist_policy.checkpoint_path="/path/to/twist/checkpoint_9000.pt" \
    wandb.mode=disabled \
    total_frames=100000
```

### 2. 添加debug输出
在`ppo_roa.py`的关键位置添加:
```python
# In compute_frozen_policy_reference:
print(f"[DEBUG] Frozen ref: shape={ref_action.shape}, mean={ref_action.mean():.4f}, std={ref_action.std():.4f}")

# In FrozenPolicyRefModule.forward:
print(f"[DEBUG] Student residual: shape={action.shape}, mean={action.mean():.4f}, std={action.std():.4f}")
print(f"[DEBUG] Final action: shape={final_action.shape}, mean={final_action.mean():.4f}, std={final_action.std():.4f}")

# In rollout after env step:
print(f"[DEBUG] Reward: {tensordict['reward'].mean():.4f}")
```

### 3. 可视化frozen reference
```python
# 在play.py中比较frozen ref vs student action
python scripts/play.py \
    algo=ppo_roa_train_twist_ref \
    task=G1/hdmi/move_suitcase_twist_ref \
    checkpoint_path=run:<wandb_run_path> \
    record_video=true
```

## 下一步工作

1. **诊断reward低的原因**
   - 添加debug输出,检查action统计
   - 可视化frozen ref vs student residual
   - 对比有/无frozen ref的reward

2. **优化TwistObservationAdapter**
   - 验证observation构建正确性
   - 检查motion timing同步
   - 确保坐标系一致

3. **调整residual weight**
   - 实现可训练的residual weight
   - 尝试curriculum: 0.1 → 1.0

4. **Ablation实验**
   - 只用frozen ref (student residual=0)
   - 只用student (no frozen ref)
   - Full residual learning

## 总结

本实现成功集成了frozen TWIST policy作为HDMI的reference,使用residual learning架构。核心挑战在于:

1. ✅ TensorDict key的动态管理
2. ✅ Observation space的转换
3. ✅ Inference mode下的tensor操作
4. ⚠️ Action scale和coordination的匹配 (待验证)

当前系统可以运行,但需要进一步调试以确保frozen reference与HDMI student正确协作。
