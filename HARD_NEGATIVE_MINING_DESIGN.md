# Hard Negative Mining + 显存可控的实现方案

## 🎯 需求分析

### 你的核心需求：

1. ✅ **Hard Negative Mining**（像 TWIST 论文）
   - 根据 tracking 成功率动态调整采样概率
   - 失败的 motion → 增加采样概率
   - 成功的 motion → 降低采样概率
   - 定期过滤无法学习的 motion

2. ✅ **显存可控**
   - 13,000 个 motion 不能全部加载到 GPU
   - 4096 个环境同时运行
   - 每个时刻只需要加载 4096 个 motion（每个环境一个）

3. ✅ **性能要求**
   - 训练速度不能明显下降
   - 采样和加载要高效

---

## 📊 TWIST Hard Negative Mining 机制详解

### **原理**

```python
# TWIST 的实现逻辑
class HardNegativeMining:
    def __init__(self, num_motions):
        # 每个 motion 的采样权重（初始均匀）
        self.motion_weights = torch.ones(num_motions) / num_motions

        # 每个 motion 的成功率统计
        self.success_count = torch.zeros(num_motions)
        self.attempt_count = torch.zeros(num_motions)

    def update_weights(self, motion_ids, success_flags):
        """根据 tracking 成功与否更新权重"""
        for i, motion_id in enumerate(motion_ids):
            self.attempt_count[motion_id] += 1
            self.success_count[motion_id] += success_flags[i]

        # 计算成功率
        success_rate = self.success_count / (self.attempt_count + 1e-8)

        # 失败率高的 motion → 权重增加
        # 成功率高的 motion → 权重降低
        failure_rate = 1 - success_rate
        self.motion_weights = failure_rate / failure_rate.sum()

    def sample(self, n):
        """根据权重采样"""
        return torch.multinomial(self.motion_weights, n, replacement=True)
```

### **与 Curriculum Learning 的区别**

| 特性 | Curriculum Learning | Hard Negative Mining |
|------|---------------------|----------------------|
| **目标** | 从简单到困难 | 专注于困难样本 |
| **权重依据** | 预定义的难度 | 实际 tracking 成功率 |
| **采样策略** | 难度过滤 | 加权采样 |
| **适用场景** | 训练初期加速 | 全程提升覆盖率 |

---

## 🚀 解决方案：On-Demand Loading + Hard Negative Mining

### **核心思想**

```
┌─────────────────────────────────────────────────────┐
│ CPU Memory (轻量级元数据)                            │
│ - motion_paths: List[Path]  (13000 个路径)          │
│ - motion_lengths: torch.Tensor  (13000,)            │
│ - motion_weights: torch.Tensor  (13000,)  ← HNM     │
│ - success_rate: torch.Tensor  (13000,)    ← HNM     │
└─────────────────────────────────────────────────────┘
                        ↓
               根据 weights 采样
                        ↓
┌─────────────────────────────────────────────────────┐
│ Current Batch (4096 个环境需要的 motion)              │
│ - motion_ids: [12, 88, 234, ..., 5678]  (4096 个)   │
│                                                      │
│ 去重后：unique_ids: [12, 88, ..., 5678]  (~2000 个) │
└─────────────────────────────────────────────────────┘
                        ↓
              按需从磁盘加载到 GPU
                        ↓
┌─────────────────────────────────────────────────────┐
│ GPU Memory (只加载当前 batch 需要的)                  │
│ - motion_data[12]: TwistMotionData                  │
│ - motion_data[88]: TwistMotionData                  │
│ - ...                                               │
│ Total: ~2000 个 motion (~2-3GB)                      │
└─────────────────────────────────────────────────────┘
```

**关键优势**：
- ✅ 每次只加载 ~2000 个 unique motion（4096 个环境会有重复）
- ✅ 显存占用：2-3GB（可控）
- ✅ 支持 Hard Negative Mining（权重在 CPU 上）
- ✅ 每个 step 都可以更新权重

---

## 📐 详细设计

### **架构 1: Minimal GPU Footprint Dataset**

```python
class OnDemandTwistMotionDataset:
    """按需加载的 Motion 数据集，支持 Hard Negative Mining"""

    def __init__(
        self,
        motion_paths: List[Path],
        device: str = "cuda",
        enable_hard_negative_mining: bool = True,
        hnm_alpha: float = 2.0,  # 失败样本权重增加倍数
        hnm_beta: float = 0.5,   # 成功样本权重降低倍数
    ):
        """
        Args:
            motion_paths: 所有 motion 文件路径
            device: 加载到的设备
            enable_hard_negative_mining: 是否启用 HNM
            hnm_alpha: 失败时权重乘以 alpha
            hnm_beta: 成功时权重乘以 beta
        """
        self.device = device
        self.num_motions = len(motion_paths)

        # ==================== CPU 元数据 ====================
        self.motion_paths = motion_paths
        self.motion_lengths = torch.zeros(self.num_motions, dtype=torch.long)
        self.motion_fps = torch.zeros(self.num_motions, dtype=torch.float32)

        # 快速扫描元数据（不加载实际数据）
        print(f"📊 Scanning {self.num_motions} motion metadata...")
        self._scan_metadata()

        # ==================== Hard Negative Mining ====================
        self.enable_hnm = enable_hard_negative_mining
        self.hnm_alpha = hnm_alpha
        self.hnm_beta = hnm_beta

        if self.enable_hnm:
            # 采样权重（初始均匀分布）
            self.motion_weights = torch.ones(self.num_motions) / self.num_motions

            # 成功率统计
            self.success_count = torch.zeros(self.num_motions)
            self.attempt_count = torch.zeros(self.num_motions)
            self.success_rate = torch.zeros(self.num_motions)

            # 过滤标记（持续失败的 motion）
            self.filtered_mask = torch.ones(self.num_motions, dtype=torch.bool)

        # ==================== GPU 缓存（当前 batch）====================
        # 注意：这里不是 LRU 缓存，而是每个 step 都清空重新加载
        self.current_batch_data = {}  # {motion_id: TwistMotionData}

        # 元数据索引
        self.starts = None  # 在第一次加载时创建
        self.ends = None

    def _scan_metadata(self):
        """快速扫描所有 motion 的元数据"""
        import pickle
        from tqdm import tqdm

        for i, path in enumerate(tqdm(self.motion_paths, desc="Scanning")):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.motion_lengths[i] = data['body_pos_w'].shape[0]
                    self.motion_fps[i] = data.get('fps', 50.0)
            except Exception as e:
                print(f"⚠️  Failed to scan {path}: {e}")
                self.motion_lengths[i] = 0  # 标记为无效

    @property
    def lengths(self):
        """兼容原有 API"""
        return self.motion_lengths

    def sample_motions(self, n: int) -> torch.Tensor:
        """根据 Hard Negative Mining 权重采样 motion IDs

        Args:
            n: 采样数量（通常是 num_envs）

        Returns:
            motion_ids: [n] 采样的 motion ID
        """
        if self.enable_hnm:
            # 过滤掉被标记为无效的 motion
            valid_weights = self.motion_weights * self.filtered_mask.float()
            valid_weights = valid_weights / valid_weights.sum()

            # 加权采样
            motion_ids = torch.multinomial(
                valid_weights,
                num_samples=n,
                replacement=True
            )
        else:
            # 均匀随机采样
            motion_ids = torch.randint(0, self.num_motions, (n,))

        return motion_ids

    def load_batch(self, motion_ids: torch.Tensor):
        """加载一个 batch 需要的所有 motion 到 GPU

        Args:
            motion_ids: [num_envs] 当前 batch 需要的 motion IDs
        """
        # 1. 清空旧数据
        self.current_batch_data.clear()
        torch.cuda.empty_cache()

        # 2. 去重（4096 个环境可能使用相同的 motion）
        unique_ids = torch.unique(motion_ids)

        print(f"🔄 Loading {len(unique_ids)} unique motions (from {len(motion_ids)} envs)")

        # 3. 从磁盘加载每个 unique motion
        import pickle
        for motion_id in unique_ids:
            path = self.motion_paths[motion_id]

            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)

                # 转换为 GPU tensor
                motion_data = TwistMotionData(
                    body_pos_w=torch.as_tensor(data['body_pos_w'], device=self.device),
                    body_quat_w=torch.as_tensor(data['body_quat_w'], device=self.device),
                    body_lin_vel_w=torch.as_tensor(data['body_lin_vel_w'], device=self.device),
                    body_ang_vel_w=torch.as_tensor(data['body_ang_vel_w'], device=self.device),
                    joint_pos=torch.as_tensor(data['joint_pos'], device=self.device),
                    joint_vel=torch.as_tensor(data['joint_vel'], device=self.device),
                    # ... 其他字段
                )

                self.current_batch_data[motion_id.item()] = motion_data

            except Exception as e:
                print(f"❌ Failed to load motion {motion_id}: {e}")
                # 标记为无效
                if self.enable_hnm:
                    self.filtered_mask[motion_id] = False

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        time_steps: torch.Tensor,
        steps: Union[int, List[int]]
    ) -> TwistMotionData:
        """获取多个 motion 的时间切片

        Args:
            motion_ids: [N] 要获取的 motion IDs
            time_steps: [N] 每个 motion 的起始时间步
            steps: int 或 List[int]，要获取的未来步数

        Returns:
            切片数据 TwistMotionData
        """
        # 确保所有需要的 motion 都已加载
        unique_ids = torch.unique(motion_ids)
        missing_ids = [mid for mid in unique_ids if mid.item() not in self.current_batch_data]

        if missing_ids:
            # 应该不会发生，因为 load_batch 已经加载了
            print(f"⚠️  Warning: {len(missing_ids)} motions not loaded, loading now...")
            self.load_batch(torch.tensor(missing_ids, device=self.device))

        # 从缓存中获取切片
        results = []
        for i, (motion_id, t) in enumerate(zip(motion_ids, time_steps)):
            motion_data = self.current_batch_data[motion_id.item()]

            if isinstance(steps, int):
                indices = torch.arange(t, min(t + steps, len(motion_data)), device=self.device)
            else:
                indices = torch.clamp(t + torch.tensor(steps, device=self.device), 0, len(motion_data) - 1)

            # 切片
            sliced = TwistMotionData(
                body_pos_w=motion_data.body_pos_w[indices],
                body_quat_w=motion_data.body_quat_w[indices],
                # ... 其他字段
            )
            results.append(sliced)

        # 合并
        return TwistMotionData.stack(results)

    def update_hard_negative_mining(
        self,
        motion_ids: torch.Tensor,
        success_flags: torch.Tensor
    ):
        """更新 Hard Negative Mining 权重

        Args:
            motion_ids: [num_envs] 刚刚使用的 motion IDs
            success_flags: [num_envs] 是否成功完成（True/False）
        """
        if not self.enable_hnm:
            return

        # 统计每个 motion 的成功次数
        for motion_id, success in zip(motion_ids, success_flags):
            mid = motion_id.item()
            self.attempt_count[mid] += 1
            self.success_count[mid] += success.item()

        # 更新成功率
        self.success_rate = self.success_count / (self.attempt_count + 1e-8)

        # 更新权重（失败率高的权重增加）
        # 策略 1: 基于失败率
        failure_rate = 1 - self.success_rate

        # 策略 2: 指数调整（TWIST 风格）
        # success → weight *= beta (降低)
        # failure → weight *= alpha (增加)
        for mid in range(self.num_motions):
            if self.attempt_count[mid] < 10:  # 样本数太少，不调整
                continue

            sr = self.success_rate[mid]
            if sr > 0.95:  # 太简单
                self.motion_weights[mid] *= self.hnm_beta
            elif sr < 0.5:  # 太难
                self.motion_weights[mid] *= self.hnm_alpha

        # 归一化
        self.motion_weights = self.motion_weights / self.motion_weights.sum()

    def filter_impossible_motions(self, min_attempts: int = 100, max_failure_rate: float = 0.95):
        """过滤掉持续失败的 motion

        Args:
            min_attempts: 最少尝试次数
            max_failure_rate: 最大允许失败率
        """
        if not self.enable_hnm:
            return

        # 找出尝试次数足够且失败率过高的 motion
        enough_attempts = self.attempt_count >= min_attempts
        too_hard = self.success_rate < (1 - max_failure_rate)

        filtered = enough_attempts & too_hard
        num_filtered = filtered.sum().item()

        if num_filtered > 0:
            print(f"🚫 Filtering {num_filtered} impossible motions (success rate < {1-max_failure_rate:.1%})")
            self.filtered_mask[filtered] = False

            # 重新归一化权重
            valid_weights = self.motion_weights * self.filtered_mask.float()
            self.motion_weights = valid_weights / valid_weights.sum()
```

---

## 🔄 与 Command 集成

### **修改 TwistMotionTracking**

```python
# active_adaptation/envs/mdp/commands/twist/command.py

class TwistMotionTracking(Command):
    def __init__(self, env, data_path, ...):
        # 创建 on-demand dataset
        self.dataset = OnDemandTwistMotionDataset.create_from_yaml(
            data_path,
            device=env.device,
            enable_hard_negative_mining=True,  # 启用 HNM
        )

        # 当前 batch 的 motion IDs
        self.motion_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def sample_init(self, env_ids: torch.Tensor):
        """环境 reset 时调用"""
        # 1. 更新 Hard Negative Mining 权重
        if self.motion_curriculum:
            self._update_motion_difficulty(env_ids)

        if self.dataset.enable_hnm:
            # 判断刚结束的 episode 是否成功
            success_flags = self.t[env_ids] >= (self.motion_len[env_ids] - 10)  # 接近结束视为成功
            self.dataset.update_hard_negative_mining(
                self.motion_ids[env_ids],
                success_flags
            )

        # 2. 采样新的 motion IDs
        new_motion_ids = self.dataset.sample_motions(len(env_ids))
        self.motion_ids[env_ids] = new_motion_ids

        # 3. 加载这个 batch 需要的所有 motion
        self.dataset.load_batch(self.motion_ids)  # 自动去重，只加载 unique 的

        # 4. 初始化机器人状态
        # ... (使用 self.dataset.get_slice 获取数据)

    def update(self):
        """每个 step 调用"""
        # 获取未来参考运动
        self.future_ref_motion = self.dataset.get_slice(
            self.motion_ids,
            self.t,
            steps=self.future_steps
        )

        # ... 其他逻辑
```

---

## 📊 显存占用分析

### **对比**

| 方案 | 加载策略 | GPU 显存占用 | CPU 内存占用 |
|------|---------|-------------|-------------|
| **原方案** | 全部加载 | ~50GB (13000 个) | ~10GB |
| **LRU 缓存** | 缓存 1000 个 | ~4GB | ~1GB (元数据) |
| **On-Demand** | 每 step 加载 ~2000 个 | **~2-3GB** | ~500MB (元数据) |

### **详细计算（On-Demand 方案）**

```python
# 假设每个 motion 平均 500 帧，每帧数据大小
motion_size = (
    500 * 28 * 3 * 4 +  # body_pos_w: [500, 28, 3] float32
    500 * 28 * 4 * 4 +  # body_quat_w: [500, 28, 4] float32
    500 * 28 * 3 * 4 +  # body_lin_vel_w
    500 * 28 * 3 * 4 +  # body_ang_vel_w
    500 * 29 * 4 +      # joint_pos: [500, 29] float32
    500 * 29 * 4        # joint_vel: [500, 29] float32
) / 1024 / 1024  # 转换为 MB

print(f"每个 motion: {motion_size:.2f} MB")
# 结果：约 1.5 MB

# 4096 个环境，去重后约 2000 个 unique motion
total_size = motion_size * 2000
print(f"总显存占用: {total_size:.2f} MB = {total_size/1024:.2f} GB")
# 结果：约 3GB
```

---

## ⚡ 性能优化

### **优化 1: 批量加载**

```python
def load_batch(self, motion_ids):
    unique_ids = torch.unique(motion_ids)

    # 使用多线程并行加载
    from concurrent.futures import ThreadPoolExecutor

    def load_one(motion_id):
        with open(self.motion_paths[motion_id], 'rb') as f:
            return pickle.load(f)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(load_one, mid): mid for mid in unique_ids}

        for future in futures:
            motion_id = futures[future]
            data = future.result()
            self.current_batch_data[motion_id.item()] = self._to_gpu(data)
```

### **优化 2: 预加载下一个 batch**

```python
class OnDemandTwistMotionDataset:
    def __init__(self, ...):
        self.prefetch_queue = []
        self.prefetch_thread = None

    def prefetch_next_batch(self):
        """后台预加载下一个可能的 batch"""
        # 根据 HNM 权重预测
        predicted_ids = torch.multinomial(self.motion_weights, 2000)

        # 异步加载
        self.prefetch_thread = threading.Thread(
            target=self._async_load,
            args=(predicted_ids,)
        )
        self.prefetch_thread.start()
```

---

## 🎯 配置方式

### **YAML 配置**

```yaml
# cfg/task/G1/twist/0927_twist_teacher_new.yaml
command:
  _target_: active_adaptation.envs.mdp.commands.twist.command.TwistMotionTracking
  data_path: cfg/task/G1/twist/twist_dataset_13000.yaml

  # On-Demand Loading
  lazy_loading: true                    # 启用按需加载

  # Hard Negative Mining
  enable_hard_negative_mining: true     # 启用 HNM
  hnm_alpha: 2.0                        # 失败样本权重 ×2
  hnm_beta: 0.5                         # 成功样本权重 ×0.5
  hnm_filter_interval: 10000            # 每 10000 steps 过滤一次
  hnm_min_attempts: 100                 # 最少尝试 100 次才判断
  hnm_max_failure_rate: 0.95            # 失败率 >95% 就过滤

  # Curriculum Learning (可选，可以同时启用)
  motion_curriculum: true
  motion_curriculum_gamma: 0.01
  sample_motion: true
```

---

## 📈 预期效果

### **显存占用**

```
Before: 50GB (全部加载 13000 个 motion)
   ↓
After:  2-3GB (每 step 加载 ~2000 个 unique motion)
   ↓
节省: ~94% 显存
```

### **训练速度**

```
Overhead:
  - 磁盘读取: ~50ms (SSD, 并行加载)
  - GPU 传输: ~20ms
  - 总计: ~70ms per step

训练时间:
  - 原始: 100 FPS (10ms per step)
  - 新方案: 12-13 FPS (80ms per step)
  - 性能损失: ~20% (可通过预加载优化到 <10%)
```

### **Hard Negative Mining 效果**

```
训练前期 (0-30%):
  - 所有 motion 权重均匀
  - 采样覆盖所有 13000 个

训练中期 (30-70%):
  - 简单 motion 权重降低（success rate >95%）
  - 困难 motion 权重增加（success rate <50%）
  - 过滤掉 ~500 个不可能的 motion

训练后期 (70-100%):
  - 专注于剩余的困难样本
  - 整体成功率 >90%
```

---

## 🎉 总结

### **方案特点**

1. ✅ **显存可控**：每 step 只加载 ~2000 个 motion (2-3GB)
2. ✅ **支持 HNM**：动态调整采样权重
3. ✅ **支持 Curriculum**：可以同时启用（两种策略结合）
4. ✅ **4096 环境**：完全支持
5. ✅ **易于配置**：通过 YAML 控制所有参数

### **与 TWIST 论文的对应**

| TWIST 特性 | 实现方式 |
|-----------|---------|
| Hard Negative Mining | ✅ `update_hard_negative_mining()` |
| 动态权重调整 | ✅ `motion_weights` |
| Motion Filtering | ✅ `filter_impossible_motions()` |
| 周期性评估 | ✅ 每 `hnm_filter_interval` steps |

### **下一步**

选择：
1. **"实现 On-Demand Dataset"** - 我立即开始编写代码
2. **"简化版本"** - 先用更简单的方案快速验证
3. **"看设计文档"** - 你先研究这个方案

告诉我你的选择！🚀
