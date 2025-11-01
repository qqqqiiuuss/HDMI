# HDMI WandB Episode_Len 计算详解

## 问题
WandB曲线中的 `train/state/episode_len` 是如何计算的？

## 答案总结

**`train/state/episode_len` = 所有完成episode的平均长度（步数）**

具体来说：
- 每个environment独立计数步数
- 当某个environment的episode结束（done=True）时，记录其episode长度
- 定期（每log_interval）计算所有完成episode的平均长度
- 上报到WandB

---

## 详细计算流程

### 1. Episode长度计数器初始化

```python
# active_adaptation/envs/base.py:263
self.episode_length_buf = torch.zeros(self.num_envs, dtype=int, device=self.device)
```

**说明**:
- `episode_length_buf`: shape `(num_envs,)` - 每个环境一个计数器
- 初始值: 全0
- 例如: 4096个环境 → `episode_length_buf.shape = (4096,)`

---

### 2. 每步自增

```python
# active_adaptation/envs/base.py:790 (在_step()方法中)
self.episode_length_buf.add_(1)
```

**说明**:
- 每次调用`env.step()`，所有环境的计数器+1
- 例如: 执行10步后，所有环境的计数器都是10

---

### 3. Episode结束时记录

```python
# active_adaptation/envs/base.py:716 (在_compute_reward()方法中)
self.stats["episode_len"][:] = self.episode_length_buf.unsqueeze(1)
```

**说明**:
- 在每一步计算reward时，将当前的`episode_length_buf`写入`stats["episode_len"]`
- `stats["episode_len"]`: shape `(num_envs, 1)` - 每个环境的当前步数

---

### 4. Reset时清零

```python
# active_adaptation/envs/base.py:634 (在reset()方法中)
self.episode_length_buf[env_ids] = 0
```

**说明**:
- 当某些环境重置时（done=True），将对应的计数器清零
- 例如: 环境0和100结束 → `episode_length_buf[[0, 100]] = 0`

---

### 5. 累积完成episode的统计

```python
# scripts/helpers.py:95-103 (EpisodeStats.add()方法)
def add(self, tensordict: TensorDictBase) -> TensorDictBase:
    next_tensordict = tensordict["next"]
    done = next_tensordict["done"]  # shape: (num_envs, 1)

    if done.any():
        done = done.squeeze(-1)  # shape: (num_envs,)

        # 只选择done=True的环境
        next_tensordict = next_tensordict.select(*self.in_keys)

        # 累积这些环境的stats
        self._stats = self._stats + next_tensordict[done].sum(dim=0)
        # self._stats["episode_len"] += next_tensordict[done]["stats", "episode_len"].sum()

        # 计数完成的episode数量
        self._episodes += done.sum()

    return len(self)
```

**示例**:
假设在某一步：
- 环境0: done=True, episode_len=85
- 环境1: done=False, episode_len=42
- 环境2: done=True, episode_len=120
- 环境3: done=False, episode_len=30

那么：
```python
self._stats["episode_len"] += 85 + 120  # = 205
self._episodes += 2  # 两个episode完成
```

---

### 6. 计算平均值并上报

```python
# scripts/train.py:232-235
if i % log_interval == 0 and len(episode_stats):
    for k, v in sorted(episode_stats.pop().items(True, True)):
        key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
        info[key] = torch.mean(v.float()).item()

# scripts/helpers.py:105-109 (EpisodeStats.pop()方法)
def pop(self):
    stats = self._stats / self._episodes  # 计算平均值
    self._stats.zero_()  # 清零累积值
    self._episodes.zero_()  # 清零计数
    return stats.cpu()
```

**示例**:
延续上面的例子，如果在log_interval期间累积了：
```python
self._stats["episode_len"] = 85 + 120 + 95 + 110 + ... = 12800 (总和)
self._episodes = 2 + 1 + 1 + ... = 160 (完成的episode数量)
```

那么：
```python
average_episode_len = 12800 / 160 = 80.0
info["train/state/episode_len"] = 80.0
```

这个值被上报到WandB。

---

### 7. Log频率

```python
# scripts/train.py:99
log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
```

**示例**:
- `max_episode_length = 500`
- `train_every = 32`
- `log_interval = 500 // 32 + 1 = 16`

意味着每16个训练迭代（16 × 32 = 512步）记录一次统计。

---

## 完整示例

### 场景设置
- 4096个并行环境
- `train_every = 32` (每32步训练一次)
- `log_interval = 16` (每16次训练记录一次)
- 记录周期 = 16 × 32 = 512步

### 时间线

#### Step 0-31 (第1个train_every)
```
Environment 0: step 1, 2, 3, ..., 31
Environment 1: step 1, 2, 3, ..., 31
...
Environment 4095: step 1, 2, 3, ..., 31

完成的episode: 0个
```

#### Step 32-63 (第2个train_every)
```
Environment 0: step 32, 33, ..., 63
Environment 50: step 32, 33, ..., 85 [DONE at 85] → reset → step 1, 2, ...
Environment 100: step 32, 33, ..., 78 [DONE at 78] → reset → step 1, 2, ...
...

完成的episode: 假设120个
累积: episode_len_sum = 85 + 78 + ... = 9600
      episode_count = 120
```

#### Step 480-511 (第16个train_every)
```
累积到log_interval:
  episode_len_sum = 85 + 78 + 95 + ... = 12800 (总和)
  episode_count = 160 (完成的episode数)

计算平均值:
  average_episode_len = 12800 / 160 = 80.0

上报到WandB:
  train/state/episode_len = 80.0
```

---

## 与TWIST-MAIN的对比

### TWIST-MAIN
```python
# Isaac Gym中的实现
self.extras["episode"] = {}
for key in self.episode_sums.keys():
    self.extras["episode"][key] = torch.mean(
        self.episode_sums[key][env_ids] / motion_length
    )
```

**特点**:
- 只在环境reset时记录（不是每步记录）
- 可能normalize by motion_length

### HDMI
```python
# 每步记录当前长度
self.stats["episode_len"][:] = self.episode_length_buf

# Episode结束时累积
if done.any():
    self._stats += next_tensordict[done].sum(dim=0)
    self._episodes += done.sum()

# 定期计算平均值
stats = self._stats / self._episodes
```

**特点**:
- 每步都记录到stats
- Episode结束时才累积到统计
- 定期计算所有完成episode的平均值

---

## 为什么HDMI的episode_len是80而TWIST是400？

### 可能原因1: 提前终止

如果HDMI在80步时触发了终止条件：
```python
# active_adaptation/envs/base.py:876
truncated = (self.episode_length_buf >= self.max_episode_length).unsqueeze(1)
# 或者其他终止条件
terminated = self._compute_termination()  # cum_body_pos_error等
```

那么：
- Episode在80步结束
- `episode_len = 80`被记录
- 平均值也接近80

### 可能原因2: 训练阶段不同

TWIST的episode_len会随训练提升：
- 训练初期：tracking差，80步就超限 → episode_len ≈ 80
- 训练后期：tracking好，能跑完整个motion → episode_len ≈ 400

### 可能原因3: Motion长度不同

如果用的motion不同：
- HDMI的motion: 1.6秒 × 50Hz = 80步
- TWIST的motion: 8秒 × 50Hz = 400步

但你说用的是同一个motion，所以这个不太可能。

---

## 诊断建议

### 1. 打印详细信息

在`base.py:716`后添加：

```python
# base.py:716
self.stats["episode_len"][:] = self.episode_length_buf.unsqueeze(1)

# 添加调试代码
if self.timestamp % 1000 == 0:  # 每1000步打印一次
    done_envs = (self.episode_length_buf >= 1).sum()
    print(f"[Step {self.timestamp}]")
    print(f"  Active envs: {done_envs}/{self.num_envs}")
    print(f"  Episode lengths: min={self.episode_length_buf.min()}, "
          f"mean={self.episode_length_buf.float().mean():.1f}, "
          f"max={self.episode_length_buf.max()}")
```

### 2. 检查终止统计

在reset()时打印：

```python
# base.py:634
self.episode_length_buf[env_ids] = 0

# 添加
if len(env_ids) > 0:
    lengths = self.episode_length_buf[env_ids].cpu()
    print(f"[Reset] {len(env_ids)} envs, "
          f"lengths: min={lengths.min()}, mean={lengths.float().mean():.1f}, max={lengths.max()}")
```

### 3. 对比阈值效果

使用修改后的配置（threshold: 1.0）训练，观察：
- `train/state/episode_len` 是否从80增长
- `train/stats/episode_len` vs `train/state/episode_len` 是否相同

---

## 总结

**`train/state/episode_len` 的计算**:
1. 每个环境独立计数步数（`episode_length_buf`）
2. Episode结束时记录长度
3. 定期计算所有完成episode的平均长度
4. 上报到WandB

**HDMI的episode_len=80原因**:
- 最可能：某个终止条件在80步左右触发（如`cum_body_pos_error_local`）
- 需要验证：用threshold=1.0重新训练，看是否改善

**关键区别**:
- HDMI: 记录**完成episode的平均长度**
- TWIST: 可能normalize by motion_length或记录方式不同
