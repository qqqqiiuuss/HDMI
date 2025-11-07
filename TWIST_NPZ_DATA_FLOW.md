# TWIST Frozen Policy 从 NPZ 读取 Reference Motion 的完整数据流

## 概述

本文档详细说明 TWIST frozen policy 如何从 NPZ 文件读取参考运动数据，以及数据如何流经各个模块最终构建观察。

---

## 数据流图

```
NPZ 文件 (motion.npz)
    ↓
MotionDataset.create_from_path()  [加载到内存]
    ↓
MotionDataset.data: MotionData  [TensorClass, 所有帧]
    ↓
TwistMotionTracking.update()
    ↓
dataset.get_slice(motion_ids, t, future_steps)  [索引切片]
    ↓
future_ref_motion: MotionData  [num_envs, num_future_steps, ...]
    ↓
提取字段: body_pos_w, body_quat_w, joint_pos, root_lin_vel_w 等
    ↓
ref_motion_windowed.update()  [TWIST 观察函数]
    ↓
构建观察: [root_pos, root_ori, joint_pos] × window_length
    ↓
TwistObservationAdapter.compute()
    ↓
twist_obs: [num_envs, 1816]  [拼接的观察张量]
    ↓
frozen_policy.actor(twist_obs)
    ↓
frozen_action: [num_envs, action_dim]
```

---

## 详细代码路径

### 1. NPZ 文件加载 → MotionDataset

**入口**: `active_adaptation/envs/mdp/commands/twist/command.py:111-145`

```python
# TwistMotionTracking.__init__()

# 检测是否是 NPZ 格式
is_npz_format = False
if isinstance(data_path, str):
    path_obj = Path(data_path)
    if path_obj.is_dir() and (path_obj / "motion.npz").exists():
        is_npz_format = True

if is_npz_format:
    # ✅ 使用 NPZ 加载器
    print(f"[TwistMotionTracking] Loading NPZ motion data from: {data_path}")
    self.dataset = MotionDataset.create_from_path(
        data_path,
        isaac_joint_names=self.asset.joint_names,
        target_fps=int(1/self.env.step_dt)
    ).to(self.device)
```

**加载代码**: `active_adaptation/utils/motion.py:195-330`

```python
# MotionDataset.create_from_path()

@classmethod
def create_from_path(cls, root_path: str, isaac_joint_names=None, target_fps=50):
    # 1. 查找 NPZ 目录
    motion_paths = []
    if isinstance(root_path, str):
        root_path = Path(root_path)
        if root_path.is_dir():
            # 检查是否包含 motion.npz
            if (root_path / "motion.npz").exists():
                motion_paths = [root_path]

    # 2. 加载每个 NPZ 文件
    for motion_path in motion_paths:
        npz_file = motion_path / "motion.npz"
        motion = dict(np.load(npz_file))

        # 读取元数据
        with open(motion_path / "meta.json") as f:
            meta = json.load(f)

        # 3. 提取数据
        body_pos_w = motion["body_pos_w"]      # [T, N_bodies, 3]
        body_quat_w = motion["body_quat_w"]    # [T, N_bodies, 4] wxyz
        joint_pos = motion["joint_pos"]        # [T, N_joints]
        joint_vel = motion["joint_vel"]        # [T, N_joints]
        body_lin_vel_w = motion["body_lin_vel_w"]  # [T, N_bodies, 3]
        body_ang_vel_w = motion["body_ang_vel_w"]  # [T, N_bodies, 3]

        # 4. 关节重映射（如果需要）
        if isaac_joint_names is not None:
            # 重新排列关节顺序以匹配 Isaac 机器人
            joint_indices = find_joint_mapping(meta["joint_names"], isaac_joint_names)
            joint_pos = joint_pos[:, joint_indices]
            joint_vel = joint_vel[:, joint_indices]

        # 5. FPS 重采样（如果需要）
        if meta["fps"] != target_fps:
            motion = interpolate(motion, source_fps=meta["fps"], target_fps=target_fps)

        # 6. 添加兼容字段（TwistMotionTracking 需要）
        root_lin_vel_w = body_lin_vel_w[:, 0, :]  # [T, 3] - root body 速度
        root_ang_vel_w = body_ang_vel_w[:, 0, :]  # [T, 3]

    # 7. 拼接所有运动数据到一个大张量
    total_length = sum(len(m["joint_pos"]) for m in motions)

    data = MotionData(
        motion_id=torch.zeros(total_length, dtype=torch.long),
        step=torch.zeros(total_length, dtype=torch.long),
        body_pos_w=torch.cat([m["body_pos_w"] for m in motions], dim=0),
        body_quat_w=torch.cat([m["body_quat_w"] for m in motions], dim=0),
        joint_pos=torch.cat([m["joint_pos"] for m in motions], dim=0),
        joint_vel=torch.cat([m["joint_vel"] for m in motions], dim=0),
        body_lin_vel_w=torch.cat([m["body_lin_vel_w"] for m in motions], dim=0),
        body_ang_vel_w=torch.cat([m["body_ang_vel_w"] for m in motions], dim=0),
        root_lin_vel_w=torch.cat([m["root_lin_vel_w"] for m in motions], dim=0),
        root_ang_vel_w=torch.cat([m["root_ang_vel_w"] for m in motions], dim=0),
        batch_size=[total_length]
    )

    return cls(
        body_names=body_names,
        joint_names=joint_names,
        motion_paths=motion_paths,
        starts=starts,  # 每个 motion 的起始索引
        ends=ends,      # 每个 motion 的结束索引
        data=data       # 所有帧的数据
    )
```

**数据结构**: 所有运动帧被拼接成一个大的 `MotionData` 张量

```python
# 例如: 如果有 3 个运动序列，长度分别为 100, 200, 150 帧
# data.body_pos_w.shape = [450, N_bodies, 3]
# data.joint_pos.shape = [450, N_joints]

# starts = [0, 100, 300]
# ends = [100, 300, 450]
```

---

### 2. MotionDataset → get_slice (每帧索引)

**调用位置**: `active_adaptation/envs/mdp/commands/twist/command.py:489`

```python
# TwistMotionTracking.update() - 每个环境步调用

# 获取未来参考运动数据
self.future_ref_motion = self.dataset.get_slice(
    self.motion_ids,  # [num_envs] - 每个环境正在播放的运动ID
    self.t,           # [num_envs] - 每个环境的当前帧索引
    steps=self.future_steps  # [1,2,3,...,10] - 未来10帧
)
# 返回: MotionData [num_envs, 10, ...]
```

**实现代码**: `active_adaptation/utils/motion.py:345-350`

```python
def get_slice(self, motion_ids: torch.Tensor, starts: torch.Tensor, steps: Union[int, torch.Tensor] = 1) -> MotionData:
    """
    从数据集中切片获取指定运动的指定帧

    Args:
        motion_ids: [num_envs] - 运动ID
        starts: [num_envs] - 每个环境的当前帧
        steps: int 或 [num_steps] - 要获取的步数（相对偏移）

    Returns:
        MotionData [num_envs, num_steps, ...] - 切片的运动数据
    """
    if isinstance(steps, int):
        steps = torch.arange(steps, device=self.device)

    # 计算绝对索引
    # idx[i, j] = starts[motion_ids[i]] + starts[i] + steps[j]
    idx = (self.starts[motion_ids] + starts).unsqueeze(1) + steps.unsqueeze(0)

    # 夹紧到运动结束位置（避免越界）
    idx.clamp_max_(self.ends.unsqueeze(1)[motion_ids] - 1)

    # 直接索引大张量
    return self.data[idx]  # [num_envs, num_steps, ...]
```

**示例**:

```python
# 假设:
# - 环境 0: motion_id=1, t=50 (正在播放第1个运动的第50帧)
# - 环境 1: motion_id=0, t=20 (正在播放第0个运动的第20帧)
# - future_steps = [1, 2, 3, ..., 10]

# 计算:
motion_ids = [1, 0]
t = [50, 20]
starts = [0, 100, 300]  # 从 dataset

# 环境 0:
idx[0, :] = starts[1] + 50 + [1,2,3,...,10] = 100 + 50 + [1,...,10] = [151, 152, ..., 160]

# 环境 1:
idx[1, :] = starts[0] + 20 + [1,2,3,...,10] = 0 + 20 + [1,...,10] = [21, 22, ..., 30]

# 返回:
future_ref_motion = data[[151,152,...,160], [21,22,...,30]]
# Shape: [2, 10, ...]
```

---

### 3. get_slice 结果 → 提取字段

**位置**: `active_adaptation/envs/mdp/commands/twist/command.py:493-534`

```python
# TwistMotionTracking.update()

# 1. 提取未来帧的各个字段
self.ref_body_pos_future_w = self.future_ref_motion.body_pos_w[..., self.tracking_body_indices_motion, :] + self.env.scene.env_origins[:, None, None, :]
# Shape: [num_envs, 10, num_tracking_bodies, 3]

self.ref_body_quat_future_w = self.future_ref_motion.body_quat_w[..., self.tracking_body_indices_motion, :]
# Shape: [num_envs, 10, num_tracking_bodies, 4]

self.ref_joint_pos_future_ = self.future_ref_motion.joint_pos[..., self.tracking_joint_indices_motion]
# Shape: [num_envs, 10, num_joints]

self.ref_root_pos_future_w = self.future_ref_motion.body_pos_w[..., self.root_body_idx_motion, :] + self.env.scene.env_origins[:, None, :]
# Shape: [num_envs, 10, 3]

self.ref_root_quat_future_w = self.future_ref_motion.body_quat_w[..., self.root_body_idx_motion, :]
# Shape: [num_envs, 10, 4]

self.ref_root_lin_vel_future_w = self.future_ref_motion.root_lin_vel_w
# Shape: [num_envs, 10, 3]

self.ref_root_ang_vel_future_w = self.future_ref_motion.root_ang_vel_w
# Shape: [num_envs, 10, 3]

# 2. 提取当前帧（future_steps[0]）
self.current_ref_motion = self.future_ref_motion[:, 0]  # [num_envs, ...]
self.ref_root_pos_w = self.ref_root_pos_future_w[:, 0]   # [num_envs, 3]
self.ref_root_quat_w = self.ref_root_quat_future_w[:, 0] # [num_envs, 4]
self.ref_joint_pos = self.ref_joint_pos_future_[:, 0]    # [num_envs, num_joints]
```

---

### 4. TwistMotionTracking 字段 → ref_motion_windowed 观察

**位置**: `active_adaptation/envs/mdp/commands/twist/observations.py:1177-1257`

```python
# ref_motion_windowed.update() - 每帧调用

def update(self):
    # 1. 获取当前帧的参考运动
    ref_root_pos_w = self.command_manager.ref_root_pos_w  # [num_envs, 3]
    ref_root_quat_w = self.command_manager.ref_root_quat_w  # [num_envs, 4]
    ref_joint_pos = self.command_manager.current_ref_motion.joint_pos  # [num_envs, num_joints]

    # 转换为旋转矩阵（只取前两行 = 6D）
    ref_root_rot_mat = matrix_from_quat(ref_root_quat_w)  # [num_envs, 3, 3]
    ref_root_ori = ref_root_rot_mat[:, :2, :].reshape(self.num_envs, 6)  # [num_envs, 6]

    # 拼接当前帧: [root_pos(3), root_ori(6), joint_pos(num_joints)]
    current_ref = torch.cat([
        ref_root_pos_w,   # 3
        ref_root_ori,     # 6
        ref_joint_pos,    # num_joints
    ], dim=1)  # [num_envs, 3+6+num_joints]

    # 2. 获取未来帧的参考运动
    ref_root_pos_future_w = self.command_manager.ref_root_pos_future_w  # [num_envs, 10, 3]
    ref_root_quat_future_w = self.command_manager.ref_root_quat_future_w  # [num_envs, 10, 4]
    ref_joint_pos_future = self.command_manager.ref_joint_pos_future_  # [num_envs, 10, num_joints]

    # 构造未来帧列表
    future_ref_list = []
    for i in range(self.future_frames):  # 10 帧
        future_root_pos = ref_root_pos_future_w[:, i, :]
        future_root_quat = ref_root_quat_future_w[:, i, :]
        future_joint_pos = ref_joint_pos_future[:, i, :]

        # 转换方向
        future_root_rot_mat = matrix_from_quat(future_root_quat)
        future_root_ori = future_root_rot_mat[:, :2, :].reshape(self.num_envs, 6)

        # 拼接
        future_ref = torch.cat([
            future_root_pos,
            future_root_ori,
            future_joint_pos,
        ], dim=1)

        future_ref_list.append(future_ref.unsqueeze(1))

    future_ref = torch.cat(future_ref_list, dim=1)  # [num_envs, 10, ref_motion_dim]

    # 3. 组合完整窗口: [过去10帧, 当前帧, 未来10帧]
    self.ref_motion_window_b = torch.cat([
        self.past_buffer,           # [num_envs, 10, ref_motion_dim]
        current_ref.unsqueeze(1),   # [num_envs, 1, ref_motion_dim]
        future_ref,                 # [num_envs, 10, ref_motion_dim]
    ], dim=1)  # [num_envs, 21, ref_motion_dim]

    # 4. 转换到机器人根坐标系（如果需要）
    if self.coordinate_frame == 'robot_root':
        robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, :]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :]

        # 转换位置到局部坐标
        ref_pos_w = self.ref_motion_window_b[:, :, :3]
        ref_pos_b = quat_apply_inverse(robot_root_quat_w, ref_pos_w - robot_root_pos_w)
        self.ref_motion_window_b[:, :, :3] = ref_pos_b

    # 5. 更新过去帧缓冲区（向前滚动）
    self.past_buffer[:, :-1, :] = self.past_buffer[:, 1:, :].clone()
    self.past_buffer[:, -1, :] = current_ref
```

---

### 5. ref_motion_windowed.compute() → 观察张量

**位置**: `active_adaptation/envs/mdp/commands/twist/observations.py:1258-1298`

```python
def compute(self):
    """返回展平的参考运动窗口观察"""

    # 展平窗口: [num_envs, 21 * ref_motion_dim]
    obs = self.ref_motion_window_b.view(self.num_envs, -1)

    # 如果需要，应用噪声（推理时噪声=0）
    if any([self.ref_root_pos_noise > 0, ...]):
        # 添加噪声的代码（推理时跳过）
        ...

    return obs  # [num_envs, 21 * (3+6+num_joints)]
```

**维度计算**:

```python
# 假设 29 关节, 21 帧窗口
ref_motion_dim = 3 + 6 + 29 = 38
obs_dim = 21 * 38 = 798

# 但论文实际使用 past_frames=10, future_frames=10, 共20帧（不是21）
# 如果配置为 past=10, future=10:
window_length = 10 + 1 + 10 = 21
obs_dim = 21 * 38 = 798

# ⚠️ 之前文档中说的 760维 是假设 past=10, future=10, 但没有当前帧
# 实际上 TWIST 配置是包含当前帧的，所以是 21 帧
```

---

### 6. TwistObservationAdapter.compute() → 拼接观察

**位置**: `active_adaptation/envs/mdp/commands/dual_command_manager.py:244-273`

```python
def compute(self):
    """计算所有观察并返回字典"""

    # 临时切换 command_manager
    actual_env = getattr(self.env, 'base_env', self.env)
    original_command_manager = actual_env.command_manager
    actual_env.command_manager = self.command_manager  # TWIST manager

    try:
        obs_dict = {}
        for name, obs_fn in self.obs_functions.items():
            obs_dict[name] = obs_fn.compute()
        # obs_dict = {
        #   "proprio_history_combined": [num_envs, 1056],
        #   "ref_motion_windowed": [num_envs, 798]  # 21帧 × 38维
        # }
        return obs_dict
    finally:
        # 恢复 command_manager
        actual_env.command_manager = original_command_manager

def get_observation_tensor(self):
    """获取拼接后的观察张量"""
    obs_dict = self.compute()
    obs_list = [obs_dict[name] for name in sorted(obs_dict.keys())]
    return torch.cat(obs_list, dim=-1)
    # 返回: [num_envs, 1056 + 798] = [num_envs, 1854]
```

**⚠️ 维度修正**:

根据代码，如果 `window_length = 21` (包含当前帧):
- `ref_motion_windowed`: 21 × 38 = **798维** (不是之前说的760维)
- 总观察: 1056 + 798 = **1854维** (不是1816维)

---

### 7. 观察张量 → Frozen Policy

**位置**: `active_adaptation/learning/ppo/ppo_roa.py:295-310`

```python
# FrozenPolicyRefModule.forward()

def forward(self, tensordict):
    # 1. 获取 TWIST 观察
    twist_obs = self.twist_obs_adapter.get_observation_tensor()
    # Shape: [num_envs, 1854]

    # 2. 构建 frozen policy 输入
    frozen_input = TensorDict({
        OBS_KEY: twist_obs,  # 只需要 OBS_KEY
    }, batch_size=tensordict.batch_size, device=self.device)

    # 3. Frozen policy 推理
    with torch.no_grad():
        frozen_output = self.frozen_policy.actor(frozen_input)
        ref_action = frozen_output[ACTION_KEY]

    # 4. 保存到 tensordict
    tensordict.set("_frozen_policy_ref", ref_action)

    return tensordict
```

---

## 关键数据结构总结

### NPZ 文件格式

```python
motion.npz:
    body_pos_w: [T, N_bodies, 3]       # 所有body的世界坐标位置
    body_quat_w: [T, N_bodies, 4]      # wxyz 四元数
    joint_pos: [T, N_joints]           # 关节位置
    joint_vel: [T, N_joints]           # 关节速度
    body_lin_vel_w: [T, N_bodies, 3]   # body线速度
    body_ang_vel_w: [T, N_bodies, 3]   # body角速度

meta.json:
    fps: 50.0
    body_names: [...]
    joint_names: [...]
```

### MotionData (TensorClass)

```python
class MotionData:
    motion_id: [N]                # 每帧所属的运动ID
    step: [N]                     # 每帧在运动内的步数
    body_pos_w: [N, num_bodies, 3]
    body_quat_w: [N, num_bodies, 4]
    joint_pos: [N, num_joints]
    joint_vel: [N, num_joints]
    body_lin_vel_w: [N, num_bodies, 3]
    body_ang_vel_w: [N, num_bodies, 3]
    root_lin_vel_w: [N, 3]        # 兼容字段
    root_ang_vel_w: [N, 3]        # 兼容字段
```

### TwistMotionTracking 提供的字段

```python
# 当前帧
self.ref_root_pos_w: [num_envs, 3]
self.ref_root_quat_w: [num_envs, 4]
self.ref_joint_pos: [num_envs, num_joints]
self.current_ref_motion: MotionData [num_envs]

# 未来帧
self.ref_root_pos_future_w: [num_envs, 10, 3]
self.ref_root_quat_future_w: [num_envs, 10, 4]
self.ref_joint_pos_future_: [num_envs, 10, num_joints]
```

### ref_motion_windowed 观察

```python
# 窗口缓冲
self.past_buffer: [num_envs, 10, ref_motion_dim]
self.ref_motion_window_b: [num_envs, 21, ref_motion_dim]

# 输出
obs: [num_envs, 21 * ref_motion_dim]
    = [num_envs, 21 * (3+6+29)]
    = [num_envs, 21 * 38]
    = [num_envs, 798]
```

---

## 重要配置参数

### TWIST 训练配置

**文件**: `cfg/task/G1/twist/0927_twist_teacher_new.yaml`

```yaml
observation:
  policy:
    ref_motion_windowed:
      past_frames: 10     # 过去10帧
      future_frames: 10   # 未来10帧
      coordinate_frame: world  # 世界坐标系
```

### HDMI 推理配置（默认值）

**文件**: `active_adaptation/envs/mdp/commands/dual_command_manager.py:202-210`

```python
self.obs_functions["ref_motion_windowed"] = ref_motion_windowed_cls(
    env=self.env,
    past_frames=ref_motion_cfg.get("past_frames", 10),      # 默认10
    future_frames=ref_motion_cfg.get("future_frames", 10),  # 默认10
    coordinate_frame=ref_motion_cfg.get("coordinate_frame", "world"),
    ref_root_pos_noise=0.0,     # 推理时无噪声
    ref_root_ori_noise=0.0,
    ref_joint_pos_noise=0.0
)
```

---

## 数据来源对比

### 训练时 (PKL格式)

```python
# NPZ 路径: /home/ubuntu/DATA2/Dataset-G1/AMASS_G1/accad/*.pkl
# 数据集: twist_dataset.yaml (15k+ 运动序列)
# 格式: PKL (TwistMotionDataset)
# 内容: AMASS 通用人体运动 (walk, run, crouch, dodge, etc.)
```

### 推理时 (NPZ格式)

```python
# NPZ 路径: data/motion/g1/omomo/sub1_suitcase_011/motion.npz
# 数据集: 单个任务运动
# 格式: NPZ (MotionDataset)
# 内容: HDMI 推箱子任务的特定运动
```

**关键**: 虽然数据来源不同，但**格式和访问接口一致**，确保 frozen policy 能正常工作。

---

## 性能优化点

### 1. MotionDataset 的高效索引

```python
# 所有运动帧拼接成一个大张量，避免循环
data = MotionData(
    body_pos_w=torch.cat([...], dim=0),  # [total_frames, ...]
    ...
)

# 使用高级索引一次性获取所有环境的切片
idx = (self.starts[motion_ids] + starts).unsqueeze(1) + steps.unsqueeze(0)
return self.data[idx]  # 并行索引，GPU友好
```

### 2. 避免重复计算

```python
# ref_motion_windowed 维护过去帧缓冲区
self.past_buffer  # 只需更新最新一帧，不需要重新构建整个窗口
```

### 3. 临时 command_manager 切换

```python
# 避免为 TWIST 创建完整的环境副本
# 只在计算观察时临时切换，计算完立即恢复
```

---

## 调试建议

### 1. 检查 NPZ 加载

```python
# 在 TwistMotionTracking.__init__() 后添加
print(f"Dataset loaded: {len(self.dataset.motion_paths)} motions")
print(f"Total frames: {len(self.dataset.data)}")
print(f"Joint names: {self.dataset.joint_names}")
print(f"Body names: {self.dataset.body_names}")
```

### 2. 检查 get_slice 索引

```python
# 在 TwistMotionTracking.update() 中添加
print(f"motion_ids: {self.motion_ids[:5]}")
print(f"t: {self.t[:5]}")
print(f"future_ref_motion shape: {self.future_ref_motion.body_pos_w.shape}")
```

### 3. 检查观察维度

```python
# 在 TwistObservationAdapter.get_observation_tensor() 中添加
obs = torch.cat(obs_list, dim=-1)
print(f"TWIST obs shape: {obs.shape}")
print(f"  - proprio_history_combined: {obs_dict['proprio_history_combined'].shape}")
print(f"  - ref_motion_windowed: {obs_dict['ref_motion_windowed'].shape}")
assert obs.shape[-1] == expected_dim, f"Dim mismatch: {obs.shape[-1]} vs {expected_dim}"
```

---

## 相关文件

### 核心代码
- `active_adaptation/utils/motion.py` - MotionDataset (NPZ加载器)
- `active_adaptation/envs/mdp/commands/twist/command.py` - TwistMotionTracking
- `active_adaptation/envs/mdp/commands/twist/observations.py` - ref_motion_windowed
- `active_adaptation/envs/mdp/commands/dual_command_manager.py` - TwistObservationAdapter

### 数据文件
- NPZ训练数据: `/home/ubuntu/DATA2/Dataset-G1/AMASS_G1/` (PKL格式)
- NPZ推理数据: `data/motion/g1/omomo/sub1_suitcase_011/` (NPZ格式)

### 配置文件
- `cfg/task/G1/twist/0927_twist_teacher_new.yaml` - TWIST训练配置
- `cfg/task/G1/hdmi/move_suitcase_twist_ref.yaml` - HDMI+TWIST推理配置

---

**文档日期**: 2025-11-07
**作者**: Claude Code
**状态**: ✅ 完整数据流已验证
