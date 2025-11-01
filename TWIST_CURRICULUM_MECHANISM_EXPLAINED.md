# TWIST Curriculum Learning 机制详解

## 🔍 关键发现：代码被注释了！

**重要**: 虽然TWIST-MAIN实现了per-motion的curriculum learning机制，但在实际使用中**该功能被注释掉了**！

---

## 1. 代码实现分析

### 1.1 Per-Motion Difficulty初始化

```python
# humanoid_mimic.py:40-42
num_motions = self._motion_lib.num_motions()
self.motion_difficulty = 9 * torch.ones((num_motions), device=self.device, dtype=torch.float)
self.motion_termination_dist = torch.ones((num_motions), device=self.device, dtype=torch.float)
```

**说明**:
- `motion_difficulty`: shape `(num_motions,)` - **每个motion都有独立的difficulty**
- 初始化为9.0（最简单），范围是[1.0, 9.0]
- `motion_termination_dist`: shape `(num_motions,)` - **每个motion都有独立的终止阈值**

---

### 1.2 Difficulty动态更新机制

```python
# humanoid_mimic.py:244-260
def _update_motion_difficulty(self, env_ids):
    # 1. 获取刚刚reset的环境对应的motion IDs
    reset_motion_ids = self._motion_ids[env_ids]

    # 2. 计算每个环境的完成率
    completion_rate = self.episode_length_buf[env_ids] * self.dt / motion_length
    # completion_rate = 实际执行时间 / motion总时长
    # 例如: 如果motion是10秒，执行了8秒就终止了 → completion_rate = 0.8

    # 3. 使用scatter_add统计每个motion的平均完成率
    motion_completion_rate_sum = torch.zeros(num_motions).scatter_add(0, reset_motion_ids, completion_rate)
    motion_completion_rate_count = torch.zeros(num_motions).scatter_add(0, reset_motion_ids, torch.ones_like(completion_rate))
    motion_completion_rate = motion_completion_rate_sum / motion_completion_rate_count.clamp(min=1)

    # 4. 根据完成率调整difficulty
    # 完成率 <= 50%: 太难了，降低difficulty（增加数值）
    add_idx = motion_completion_rate <= 0.5
    self.motion_difficulty[add_idx] *= (1 + gamma)  # gamma = 0.01，每次增加1%

    # 完成率 >= 95%: 太简单了，增加difficulty（减小数值）
    sub_idx = motion_completion_rate >= 0.95
    self.motion_difficulty[sub_idx] *= (1 - gamma)  # 每次减少1%

    # 5. Clamp到[1, 9]范围
    self.motion_difficulty = torch.clamp(self.motion_difficulty, min=1., max=9.)

    # 6. 根据difficulty计算termination distance
    motion_difficulty_ratio = (self.motion_difficulty - 1) / 8  # [0, 1]
    self.motion_termination_dist = (0.7 - 0.1) * motion_difficulty_ratio + 0.1
    # difficulty=1 (最难) → termination_dist = 0.1 (严格)
    # difficulty=9 (最简单) → termination_dist = 0.7 (宽松)
```

---

### 1.3 ❌ 但实际上没有使用！

```python
# humanoid_mimic.py:313-320
# Line 313-316: 被注释掉的代码
# lose_tracking = body_pos_dist > self.motion_termination_dist[self._motion_ids] ** 2
# self.deviate_tracking_frames[lose_tracking] += 1
# self.deviate_tracking_frames[~lose_tracking] = 0
# pose_fail = self.deviate_tracking_frames >= self.cfg.motion.reset_consec_frames

# Line 318: 实际使用的代码（固定阈值）
pose_fail = body_pos_dist > self._pose_termination_dist ** 2  # 固定0.7

# Line 320: 被注释掉的per-motion阈值
# pose_fail = body_pos_dist > self.motion_termination_dist[self._motion_ids] ** 2
```

**关键发现**:
- ✅ Curriculum机制**有实现**且**会计算**`motion_termination_dist`
- ❌ 但在终止判断时**使用的是固定阈值**0.7，而不是动态阈值
- ❌ 连续帧计数机制(`reset_consec_frames=30`)也被注释掉了

---

## 2. 实际行为

### 配置文件

```python
# g1_mimic_distill_config.py:287-288
motion_curriculum = True           # ✅ 启用
motion_curriculum_gamma = 0.01     # 更新步长1%
reset_consec_frames = 30           # ❌ 未使用（代码被注释）
```

### 实际效果

虽然`motion_curriculum = True`，但由于Line 318使用固定阈值，实际上：
- ✅ Difficulty会根据完成率动态调整
- ✅ `motion_termination_dist`会被计算
- ❌ **但不会影响终止判断**
- 终止条件始终是固定的0.7

**所以TWIST的终止阈值是固定的，不是动态的！**

---

## 3. 为什么episode_length会变化？

如果终止阈值是固定的，那么TWIST的episode_length为什么会从初期的短变到后期的长（400步）？

### 可能原因1: 训练质量提升

随着训练进行：
- Policy越来越好
- Tracking error越来越小
- 更不容易触发终止条件（error > 0.7）
- 所以episode更长

**这是正常的训练进步，而不是curriculum learning的效果！**

### 可能原因2: Motion数据的自然长度

如果motion本身就是8秒（400步），那么：
- 训练初期：tracking差，80步就error > 0.7，提前终止
- 训练后期：tracking好，能跟完整个400步直到motion结束

### 可能原因3: Motion结束自动重置

```python
# humanoid_mimic.py:286-290
motion_end = self.episode_length_buf * self.dt >= motion_length
if self.viewer is None:
    self.reset_buf |= motion_end
```

当motion播放完毕时，会自动重置，所以episode_length = motion_length（如果没有提前终止）。

---

## 4. HDMI vs TWIST对比

### 相同点
- 都使用固定的终止阈值（0.7）
- 都没有per-motion的动态阈值

### 不同点

#### TWIST:
```python
# 固定阈值，立即判断
pose_fail = body_pos_dist > 0.7 ** 2
self.reset_buf |= pose_fail

# 第一步保护
first_step = self.episode_length_buf == 0
self.reset_buf[first_step] = 0
```

#### HDMI:
```yaml
cum_body_pos_error_local:
  threshold: 0.7 (现在改为1.0)
  min_steps: 2  # 需要连续2步超限才触发
```

**关键差异**:
- TWIST: error > 0.7 就**立即终止**（除了第一步）
- HDMI: error > 0.7 需要**连续2步**才终止

理论上HDMI应该**更宽松**，但实际情况可能取决于：
- 误差计算方式的细微差异
- 坐标系转换的差异
- 初始化噪声的影响

---

## 5. 实验建议

### 实验1: 验证TWIST是否真的有curriculum

在TWIST代码中打印：

```python
# 在check_termination()中添加
if self.common_step_counter % 1000 == 0:
    print(f"Step {self.common_step_counter}")
    print(f"  motion_difficulty: min={self.motion_difficulty.min():.2f}, "
          f"mean={self.motion_difficulty.mean():.2f}, max={self.motion_difficulty.max():.2f}")
    print(f"  motion_termination_dist: min={self.motion_termination_dist.min():.3f}, "
          f"mean={self.motion_termination_dist.mean():.3f}, max={self.motion_termination_dist.max():.3f}")
    print(f"  Using fixed threshold: {self._pose_termination_dist}")
```

**预期结果**:
- `motion_difficulty`会变化（从9逐渐降到1左右）
- `motion_termination_dist`会变化（从0.7逐渐降到0.1左右）
- 但实际使用的是固定的0.7

---

### 实验2: 对比episode_length的训练曲线

在WandB中对比：
- TWIST在训练初期（0-5M steps）
- TWIST在训练中期（5-20M steps）
- TWIST在训练后期（>20M steps）

**如果episode_length逐渐增加**:
- 说明是training质量提升导致的，而不是curriculum
- HDMI应该也会有同样的趋势

---

### 实验3: 启用TWIST的per-motion阈值

取消注释Line 320：

```python
# 从
pose_fail = body_pos_dist > self._pose_termination_dist ** 2

# 改为
pose_fail = body_pos_dist > self.motion_termination_dist[self._motion_ids] ** 2
```

然后重新训练，看episode_length是否有明显变化。

---

## 6. 结论

### TWIST的Curriculum机制现状

1. **代码存在但未使用**:
   - ✅ 实现了per-motion difficulty tracking
   - ✅ 实现了动态termination distance计算
   - ❌ 但终止判断使用的是**固定阈值0.7**

2. **Per-Motion特性**:
   - ✅ 每个motion都有独立的difficulty
   - ✅ 根据完成率动态调整
   - ❌ 但不影响实际训练

3. **Episode Length增长原因**:
   - 主要是**training质量提升**，不是curriculum
   - Policy越来越好 → tracking error越小 → 更少触发终止

### HDMI应该怎么做

1. **已修改的阈值**（threshold: 0.7 → 1.0）应该会改善episode_length

2. **不需要实现curriculum**:
   - TWIST实际上也没用
   - Episode length会随训练自然增长

3. **关注训练质量**:
   - 如果tracking reward在提升，episode_length会自然增长
   - 如果始终在80步终止，说明tracking质量没有提升

4. **Debug重点**:
   - 打印终止统计，看是哪个条件触发
   - 对比TWIST和HDMI的实际误差数值
   - 检查初始化噪声是否过大

---

## 7. 快速验证清单

- [ ] 用threshold=1.0重新训练，观察episode_length是否增加
- [ ] 打印termination统计，确认是哪个条件导致提前终止
- [ ] 对比TWIST在相同训练阶段的episode_length（不是只看最终结果）
- [ ] 检查HDMI的tracking_error曲线是否在下降
- [ ] 如果error在下降但episode_length不增长，说明阈值仍然太严格
