# 诊断HDMI提前终止问题

## 现象
- **相同的motion数据**
- **每步tracking reward相似** (tracking质量相同)
- **Episode length差异巨大**: HDMI ~80步 vs TWIST ~400步

## 结论
HDMI在80步左右被某个**终止条件**提前终止了！

---

## 诊断步骤

### 步骤1: 打印终止原因统计

在训练代码中添加logging，找出是哪个终止条件触发的：

```python
# 在环境的step()或reset()中添加
if self.reset_buf.any():
    # 打印哪些环境被终止了
    terminated_envs = self.reset_buf.nonzero(as_tuple=False).flatten()

    # 检查每个终止条件
    for term_name, term_fn in self.termination_manager.items():
        if hasattr(term_fn, 'error'):
            # 检查这个终止条件是否触发
            triggered = term_fn().squeeze(-1)
            if triggered.any():
                num_triggered = triggered.sum().item()
                avg_error = term_fn.error[triggered].mean().item()
                print(f"[Termination] {term_name}: {num_triggered} envs triggered, avg_error={avg_error:.3f}, threshold={term_fn.threshold}")
```

**预期输出**:
```
[Termination] cum_body_pos_error_local: 2048 envs triggered, avg_error=0.75, threshold=0.7
[Termination] cum_body_z_error: 12 envs triggered, avg_error=0.22, threshold=0.2
```

这会告诉你：**是哪个终止条件导致80%以上的episode在80步终止**

---

### 步骤2: 对比阈值设置

#### HDMI当前配置：
```yaml
cum_body_pos_error_local:
  threshold: 0.7  # 关键点位置误差阈值
  min_steps: 2

cum_body_z_error:
  threshold: 0.2  # 根节点高度误差阈值
  min_steps: 2
```

#### TWIST-MAIN配置：
```python
# g1_mimic_distill_config.py:44
pose_termination_dist = 0.7  # ✅ 相同

# g1_mimic_distill_config.py:242
root_height_diff_threshold = 0.2  # ✅ 相同
```

**阈值看起来相同，但还有隐藏差异...**

---

### 步骤3: 检查TWIST的Curriculum Learning

**TWIST-MAIN有动态阈值调整**！

```python
# humanoid_mimic.py:260
motion_difficulty_ratio = (self.motion_difficulty - 1) / 8
self.motion_termination_dist = (0.7 - 0.1) * motion_difficulty_ratio + 0.1
# 训练初期：difficulty=1 → termination_dist=0.1（严格）
# 训练后期：difficulty=9 → termination_dist=0.7（宽松）
```

**关键发现**：
- TWIST在**训练后期**，termination distance会从0.1逐渐增加到0.7
- 意味着越训练越难触发终止，episode越来越长
- HDMI**没有这个机制**，threshold固定为0.7

**你看到的400步可能是TWIST训练后期的结果，而HDMI一直用固定阈值！**

---

### 步骤4: 对比误差累积机制

#### TWIST的误差计算方式：

```python
# humanoid_mimic.py:310-318
body_pos_dist = torch.sum(body_pos_diff ** 2, dim=-1)  # 每个body的平方和
body_pos_dist = torch.max(body_pos_dist, dim=-1)[0]    # 取最大的body
pose_fail = body_pos_dist > self._pose_termination_dist ** 2

# 关键：直接判断，不累积！
```

#### HDMI的误差累积机制：

```python
# terminations.py:26-35
def update(self):
    self.__exceeded = self.error >= self.threshold
    self.__cum_steps[self.__exceeded] += 1  # 累积超限步数
    self.__cum_steps[~self.__exceeded] = 0   # 未超限时清零

def __call__(self):
    return (self.__cum_steps >= self.min_steps).unsqueeze(-1)  # 连续超限min_steps才触发
```

**差异**：
- TWIST：误差 > 阈值就**立即终止**
- HDMI：误差 > 阈值需要连续**min_steps步**才终止

**理论上HDMI应该更宽松**，但实际却更容易终止？

---

### 步骤5: 检查误差计算的实际数值

在HDMI训练中添加logging：

```python
# 在terminations.py的update()中添加
if hasattr(self, 'error'):
    if (self.env.episode_length_buf % 10 == 0).any():  # 每10步打印一次
        print(f"[{self.__class__.__name__}] Step {self.env.episode_length_buf[0].item()}, "
              f"Error: min={self.error.min():.3f}, mean={self.error.mean():.3f}, max={self.error.max():.3f}, "
              f"threshold={self.threshold}, cum_steps_max={self.__cum_steps.max()}")
```

**预期输出**：
```
[cum_body_pos_error_local] Step 70, Error: min=0.12, mean=0.45, max=0.68, threshold=0.7, cum_steps_max=0
[cum_body_pos_error_local] Step 80, Error: min=0.15, mean=0.52, max=0.72, threshold=0.7, cum_steps_max=2
→ 第80步时，某些环境的error突破0.7，连续2步后触发终止
```

---

## 可能的原因和解决方案

### 原因1: Curriculum Learning缺失（最可能）

**TWIST训练后期**使用宽松的阈值（接近0.7），而**HDMI始终**使用0.7。

**但是**，TWIST在训练**初期**也用严格阈值（0.1），那时episode应该更短才对...

**除非**：你对比的是TWIST训练后期 vs HDMI训练初期/中期？

**解决方案**：
1. 对比训练的**同一阶段**（如都在1M steps时）
2. 或者在HDMI中也实现curriculum learning

---

### 原因2: 误差计算方式的细微差异

虽然公式看起来等价，但可能存在：
- 坐标系转换的差异
- 数值精度问题
- Body选择不同

**验证方法**：打印两边的实际误差数值对比

---

### 原因3: 动作延迟/平滑的差异

HDMI的action配置：
```yaml
action:
  min_delay: 2
  max_delay: 6
  alpha: [0.8, 1.0]
```

如果TWIST的延迟/平滑参数不同，会影响tracking质量，从而影响误差累积。

---

### 原因4: 初始化噪声导致累积误差

```yaml
init_joint_pos_noise: 0.1
init_joint_vel_noise: 0.1
```

如果噪声太大，初始误差就大，后续即使tracking良好，累积误差也可能超限。

**测试方法**：临时设为0，看episode_length是否增加

---

## 最简单的临时解决方案

**增大阈值，让HDMI更难触发终止**：

```yaml
cum_body_pos_error_local:
  threshold: 1.0  # 从0.7增大到1.0（+43%）
  min_steps: 2

cum_body_z_error:
  threshold: 0.3  # 从0.2增大到0.3（+50%）
  min_steps: 2
```

如果episode_length增加到200+，说明确实是阈值问题。

---

## 推荐行动顺序

1. **立即尝试**：增大threshold，看是否改善
2. **然后执行**：步骤1（打印终止原因统计），确认是哪个条件触发
3. **深入分析**：步骤5（打印误差数值），看实际误差分布
4. **对比训练阶段**：确认TWIST的400步是在哪个训练阶段观察到的

---

## 关键检查清单

- [ ] TWIST的400步是训练**初期、中期还是后期**观察到的？
- [ ] HDMI的80步是在相同训练阶段吗？
- [ ] 终止统计显示是哪个条件导致提前终止？
- [ ] 误差数值是否接近threshold？
- [ ] 增大threshold是否改善episode_length？
