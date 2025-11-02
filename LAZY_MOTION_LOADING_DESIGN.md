# 动态加载 Motion 数据设计方案

## 🎯 问题描述

**现状**：
- 需要训练 13,000 个 motion 文件
- 当前实现：一次性加载所有 motion 到 GPU 显存
- **问题**：显存爆炸，OOM (Out of Memory)

**目标**：
- 实现按需加载（Lazy Loading）
- 只在 GPU 上保留当前使用的 motion
- 不常用的 motion 保留在 CPU 或磁盘

---

## 📊 对比分析

### TWIST-master 的实现

#### ✅ 优点：
1. **全量加载到 GPU**
   ```python
   # 在 __init__ 时一次性加载所有 motion
   self.gts = torch.cat([m.global_translation for m in motions], dim=0).to(device)
   self.grs = torch.cat([m.global_rotation for m in motions], dim=0).to(device)
   ```

2. **使用 PKL 缓存加速**
   ```python
   # 第一次加载后序列化到 .pkl
   self.serialize_motions(pkl_file)
   # 之后直接读取 pkl
   self.deserialize_motions(pkl_file)
   ```

3. **访问速度极快**
   - 所有数据在 GPU，无延迟
   - 适合 motion 数量较少的场景（<1000）

#### ❌ 缺点：
- **显存占用巨大**：对于 13,000 个 motion，显存不够

---

### HDMI 当前实现

#### ✅ 优点：
1. **灵活的数据格式**
   - 支持 GMR 和 HDMI 两种格式
   - 支持 YAML 配置文件

2. **统一的 TensorClass**
   ```python
   data = TwistMotionData(
       body_pos_w=body_pos_w,  # [total_length, N_bodies, 3]
       ...
   )
   ```

#### ❌ 缺点：
- **同样全量加载**：所有 motion 拼接成一个大 tensor
- **显存占用巨大**：13,000 个 motion 会爆显存

---

## 🚀 解决方案：Lazy Loading

### 核心思想

**分层存储**：
```
┌────────────────────────────────────────┐
│ GPU (Hot Cache)                        │
│ - 正在使用的 motion (≈1000 个)         │
│ - 最近使用的 motion (LRU)              │
└────────────────────────────────────────┘
           ↕ 动态换入换出
┌────────────────────────────────────────┐
│ CPU Memory                             │
│ - 元数据（motion 路径、长度、难度）     │
│ - 索引信息                              │
└────────────────────────────────────────┘
           ↕ 按需加载
┌────────────────────────────────────────┐
│ Disk (.pkl files)                     │
│ - 所有 13,000 个 motion 文件            │
└────────────────────────────────────────┘
```

---

## 📐 设计方案

### 方案 1: Motion 池 + LRU 缓存 (推荐)

#### 架构

```python
class LazyTwistMotionDataset:
    def __init__(self, motion_paths, max_gpu_motions=1000):
        # 1. CPU 上的轻量级元数据
        self.motion_paths = motion_paths  # List[Path]
        self.motion_lengths = []  # List[int]
        self.motion_difficulty = torch.ones(len(motion_paths))  # [N]

        # 2. GPU 缓存（LRU）
        self.max_gpu_motions = max_gpu_motions
        self.gpu_cache = {}  # {motion_id: TwistMotionData}
        self.access_order = []  # LRU 队列

        # 3. 只加载元数据，不加载实际 motion 数据
        self._load_metadata()

    def _load_metadata(self):
        """快速扫描所有 motion 文件，只加载元数据"""
        for path in tqdm(self.motion_paths, desc="Loading metadata"):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                fps = data['fps']
                length = data['body_pos_w'].shape[0]
                self.motion_lengths.append(length)

    def get_motion(self, motion_id):
        """获取指定 motion（自动缓存管理）"""
        # 如果在缓存中，直接返回
        if motion_id in self.gpu_cache:
            self._update_lru(motion_id)
            return self.gpu_cache[motion_id]

        # 如果缓存满了，移除最久未使用的
        if len(self.gpu_cache) >= self.max_gpu_motions:
            oldest_id = self.access_order.pop(0)
            del self.gpu_cache[oldest_id]

        # 从磁盘加载到 GPU
        motion_data = self._load_motion_from_disk(motion_id)
        self.gpu_cache[motion_id] = motion_data
        self._update_lru(motion_id)

        return motion_data

    def get_slice(self, motion_ids, time_steps, steps):
        """获取批量 motion 切片"""
        results = []
        for i, motion_id in enumerate(motion_ids):
            motion = self.get_motion(motion_id)  # 触发缓存逻辑
            t = time_steps[i]
            # 切片逻辑...
            results.append(...)
        return TwistMotionData.stack(results)
```

#### 优点
- ✅ 显存可控：最多只有 `max_gpu_motions` 个 motion 在 GPU
- ✅ 自动管理：LRU 策略自动淘汰不常用 motion
- ✅ 透明加载：使用方式与原来完全相同

#### 缺点
- ⚠️ 首次访问有延迟：需要从磁盘加载
- ⚠️ 缓存未命中时性能下降

---

### 方案 2: 分批预加载

#### 架构

```python
class BatchedTwistMotionDataset:
    def __init__(self, motion_paths, batch_size=1000):
        self.motion_paths = motion_paths
        self.batch_size = batch_size
        self.num_batches = len(motion_paths) // batch_size

        # 当前加载的批次
        self.current_batch_id = 0
        self.current_batch_motions = {}

        # 加载第一批
        self._load_batch(0)

    def _load_batch(self, batch_id):
        """加载指定批次到 GPU"""
        start = batch_id * self.batch_size
        end = min((batch_id + 1) * self.batch_size, len(self.motion_paths))

        # 清空当前批次
        self.current_batch_motions.clear()
        torch.cuda.empty_cache()

        # 加载新批次
        for i in range(start, end):
            motion_data = self._load_motion_from_disk(i)
            self.current_batch_motions[i] = motion_data

        self.current_batch_id = batch_id

    def get_motion(self, motion_id):
        """获取 motion（自动切换批次）"""
        batch_id = motion_id // self.batch_size

        # 如果不在当前批次，切换批次
        if batch_id != self.current_batch_id:
            self._load_batch(batch_id)

        return self.current_batch_motions[motion_id]
```

#### 优点
- ✅ 实现简单
- ✅ 显存可控

#### 缺点
- ❌ 批次切换代价大（需要重新加载整个批次）
- ❌ 如果采样跨批次，性能很差

---

### 方案 3: 混合方案（最优）

结合方案 1 和方案 2：

```python
class HybridTwistMotionDataset:
    def __init__(self, motion_paths, cache_size=1000, prefetch_size=100):
        # LRU 缓存
        self.cache_size = cache_size
        self.gpu_cache = {}

        # 预加载队列（根据 curriculum 预测下一批 motion）
        self.prefetch_size = prefetch_size
        self.prefetch_queue = []

        # Curriculum 提示
        self.motion_difficulty = torch.ones(len(motion_paths))

    def predict_next_motions(self):
        """根据 curriculum 预测下一批可能采样的 motion"""
        mean_difficulty = self.motion_difficulty.mean()
        valid_mask = (self.motion_difficulty <= mean_difficulty).float()

        # 计算采样概率
        weights = valid_mask / valid_mask.sum()

        # 预测最可能被采样的 motion
        top_k = torch.topk(weights, k=self.prefetch_size)
        return top_k.indices.tolist()

    def prefetch(self):
        """后台预加载"""
        predicted_ids = self.predict_next_motions()
        for motion_id in predicted_ids:
            if motion_id not in self.gpu_cache:
                # 异步加载到 GPU
                self._async_load(motion_id)
```

---

## 🛠️ 实现步骤

### Step 1: 添加元数据快速扫描

```python
# active_adaptation/utils/twist_motion.py

@classmethod
def create_from_path_lazy(cls, root_path, max_gpu_cache=1000):
    """创建 lazy loading 数据集"""
    # 快速扫描元数据
    motion_paths = ...

    metadata = []
    for path in tqdm(motion_paths, desc="Scanning metadata"):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            metadata.append({
                'path': path,
                'length': data['body_pos_w'].shape[0],
                'fps': data['fps'],
            })

    return LazyTwistMotionDataset(metadata, max_gpu_cache)
```

### Step 2: 实现 LRU 缓存

```python
from collections import OrderedDict

class LazyTwistMotionDataset:
    def __init__(self, metadata, max_cache_size=1000):
        self.metadata = metadata
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()  # 自带 LRU 功能

    def get_motion(self, motion_id):
        if motion_id in self.cache:
            # 移到最后（最近使用）
            self.cache.move_to_end(motion_id)
            return self.cache[motion_id]

        # 加载到缓存
        motion_data = self._load_from_disk(motion_id)

        # 如果缓存满，移除最久未使用
        if len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False)  # FIFO，移除第一个

        self.cache[motion_id] = motion_data
        return motion_data
```

### Step 3: 修改 command.py 使用 lazy dataset

```python
# active_adaptation/envs/mdp/commands/twist/command.py

def __init__(self, ...):
    # 使用 lazy loading
    self.dataset = TwistMotionDataset.create_from_path_lazy(
        data_path,
        max_gpu_cache=1000,  # 最多缓存 1000 个 motion
        isaac_joint_names=self.asset.joint_names,
        target_fps=int(1/self.env.step_dt)
    )
```

### Step 4: 添加预加载优化

```python
def _update_motion_difficulty(self, env_ids):
    # 原有逻辑...

    # 预测下一批可能采样的 motion
    predicted_ids = self._predict_next_sample()

    # 后台预加载
    self.dataset.prefetch(predicted_ids)
```

---

## 📊 性能优化

### 1. 并行加载

```python
from concurrent.futures import ThreadPoolExecutor

class LazyTwistMotionDataset:
    def __init__(self, ...):
        self.loader_pool = ThreadPoolExecutor(max_workers=4)

    def _async_load(self, motion_id):
        """后台异步加载"""
        future = self.loader_pool.submit(self._load_from_disk, motion_id)
        return future
```

### 2. 内存映射（Memory Mapped）

```python
# 使用 tensordict 的 MemoryMappedTensor
from tensordict import MemoryMappedTensor

# 数据存储在磁盘，只有访问时才加载到内存
data = MemoryMappedTensor.empty(total_length, ...)
```

### 3. 压缩存储

```python
# 使用更小的数据类型
body_pos_w = torch.empty(..., dtype=torch.float16)  # 半精度
```

---

## 🎯 推荐方案

### 方案选择：**混合方案（LRU + 预加载）**

**理由**：
1. ✅ Curriculum learning 使得采样是有规律的
   - 训练初期：只采样简单 motion（difficulty=1）
   - 训练后期：采样所有 motion
2. ✅ 预加载可以隐藏加载延迟
3. ✅ LRU 确保常用 motion 始终在缓存

**配置建议**：
```yaml
# cfg/task/G1/twist/0927_twist_teacher_new.yaml
command:
  _target_: active_adaptation.envs.mdp.commands.twist.command.TwistMotionTracking
  data_path: .../twist_dataset_13000.yaml

  # Lazy loading 配置
  lazy_loading: true          # 启用 lazy loading
  max_gpu_cache: 1000         # GPU 最多缓存 1000 个 motion
  prefetch_enabled: true      # 启用预加载
  prefetch_size: 200          # 预加载 200 个 motion
```

**预期效果**：
- 显存占用：从 ~50GB 降低到 ~5GB
- 性能损失：<5%（由于预加载）
- 训练时间：几乎不变

---

## 🔧 实现优先级

### Phase 1: 基础 Lazy Loading（必需）
- [ ] 实现 LazyTwistMotionDataset
- [ ] 实现 LRU 缓存
- [ ] 修改 create_from_path 支持 lazy=True

### Phase 2: 性能优化（推荐）
- [ ] 实现预加载
- [ ] 根据 curriculum 预测采样
- [ ] 后台异步加载

### Phase 3: 高级优化（可选）
- [ ] 内存映射
- [ ] 半精度存储
- [ ] 压缩缓存

---

## 📝 测试方法

```python
# test_lazy_loading.py
def test_lazy_loading():
    dataset = TwistMotionDataset.create_from_path_lazy(
        "cfg/task/G1/twist/twist_dataset_13000.yaml",
        max_gpu_cache=100
    )

    # 测试缓存命中率
    motion_ids = torch.randint(0, 13000, (1000,))
    for motion_id in motion_ids:
        motion = dataset.get_motion(motion_id)

    print(f"Cache hits: {dataset.cache_hits}")
    print(f"Cache misses: {dataset.cache_misses}")
    print(f"Hit rate: {dataset.cache_hits / (dataset.cache_hits + dataset.cache_misses):.2%}")
```

---

## 🎉 总结

**核心改动**：
1. 不再一次性加载所有 motion 到 GPU
2. 使用 LRU 缓存动态管理 GPU 上的 motion
3. 根据 curriculum 预加载下一批可能采样的 motion

**效果**：
- 支持 13,000+ motion 训练
- 显存占用可控（用户可配置）
- 性能损失极小（<5%）

**下一步**：
开始实现 Phase 1 的基础 Lazy Loading 功能。
