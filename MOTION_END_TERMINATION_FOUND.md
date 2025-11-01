# HDMI确实有Motion End终止条件！

## 结论

**HDMI已经实现了motion_end终止条件**，通过`TwistMotionTracking`的`finished`属性实现。

---

## 实现位置

### 1. Command Manager中的finished属性

```python
# active_adaptation/envs/mdp/commands/twist/command.py:335-345
@property
def finished(self):
    """
    检查运动是否完全结束

    Returns:
        torch.Tensor: 形状为[num_envs, 1]的布尔张量，表示每个环境是否结束
    """
    if self.replay_motion:
        return torch.ones(self.num_envs, 1, dtype=bool, device=self.device)
    return (self.t >= self.motion_len).unsqueeze(1)
```

**实现逻辑**:
- `self.t`: 当前motion的时间步（每步+1，见line 417）
- `self.motion_len`: 每个环境的motion长度（从dataset获取，见line 206）
- 当 `t >= motion_len` 时，返回True

**特殊情况**:
- `replay_motion=True`时永不结束（循环播放）

---

### 2. Base Environment中的使用

```python
# active_adaptation/envs/base.py:876-880
truncated = (self.episode_length_buf >= self.max_episode_length).unsqueeze(1)

# 如果命令管理器有完成标志，也考虑截断
if hasattr(self.command_manager, "finished"):
    truncated = truncated | self.command_manager.finished
```

**逻辑**:
- `truncated`包含两种情况：
  1. Episode长度超过`max_episode_length` (500步)
  2. Motion播放完毕（`command_manager.finished == True`）

**注意**: 这是**truncated**（截断），不是**terminated**（失败终止）
- `done = terminated | truncated` (line 885)
- WandB记录的`episode_len`包含所有done的episode（无论是terminated还是truncated）

---

## 时间推进机制

```python
# active_adaptation/envs/mdp/commands/twist/command.py:417
self.t += 1  # 每步调用update()时，时间+1
```

**流程**:
1. Episode开始：`self.t[env_ids] = start_t` (line 232，通常是0)
2. 每步：`self.t += 1` (line 417)
3. 当 `self.t >= self.motion_len` 时：`finished = True`
4. 触发truncated，episode结束

---

## 验证：为什么Episode Length不是Motion Length？

### 预期行为

如果motion长度是8秒（400步@50Hz）：
- 理论上：`episode_len`应该接近400（如果tracking良好，没有提前terminated）
- 实际上：HDMI的`episode_len ≈ 80`

### 原因分析

Episode可能因为以下原因在80步提前终止：

1. **Pose误差终止** (最可能)
   ```yaml
   cum_body_pos_error_local:
     threshold: 0.7  # 刚修改回原始值
     min_steps: 2
   ```
   - 如果关键点位置误差 > 0.7m，连续2步后触发`terminated`
   - Episode在80步就terminated，不会等到motion结束

2. **高度误差终止**
   ```yaml
   cum_body_z_error:
     threshold: 0.2  # 刚修改回原始值
     min_steps: 2
   ```
   - 如果根节点高度误差 > 0.2m，连续2步后触发`terminated`

3. **Motion End终止**
   ```python
   finished = (self.t >= self.motion_len)
   ```
   - 如果没有提前terminated，会在motion结束时truncated

**关键差异**:
- TWIST训练后期：tracking质量好 → 不触发terminated → episode_len接近motion_len (400)
- HDMI训练初期：tracking质量差 → 80步就触发terminated → episode_len = 80

---

## 回答你的问题："如果取消terminate，episode_len是不是就是400？"

### 答案：理论上是，但取决于motion的实际长度

#### 场景1: 如果禁用所有terminated条件

```yaml
termination:
  cum_body_pos_error_local:
    enabled: false  # 禁用pose误差终止
  cum_body_z_error:
    enabled: false  # 禁用高度误差终止
```

**结果**:
- Episode不会因tracking误差提前终止
- **只会在motion结束时truncated**
- `episode_len = motion_len`（如果motion是400步，则episode_len=400）

#### 场景2: Motion长度验证

需要验证motion的实际长度：

```python
# 在command.py:206打印
print(f"Motion lengths: min={self.motion_len.min()}, "
      f"mean={self.motion_len.mean():.1f}, "
      f"max={self.motion_len.max()}")
```

**可能的情况**:
- 如果motion是8秒@50Hz → motion_len = 400
- 如果motion是1.6秒@50Hz → motion_len = 80

#### 场景3: 如果motion_len就是80

那么即使禁用terminated，`episode_len`也会是80（因为motion播放完毕）。

---

## 对比TWIST-MAIN

### TWIST的Motion End实现

```python
# TWIST-master/humanoid_mimic.py:286-290
motion_end = self.episode_length_buf * self.dt >= self._motion_lib.get_motion_length(self._motion_ids)
if self.viewer is None:  # 训练模式
    self.reset_buf |= motion_end
```

**差异**:
- TWIST: 使用`episode_length_buf * dt`计算当前时间
- HDMI: 使用`self.t`（在command manager中维护）

**结果相同**: 都会在motion结束时触发重置

### 为什么TWIST的Episode Length是400？

**可能原因**:
1. **Motion长度确实是8秒**（400步）
2. **训练后期tracking质量好**，不会在80步触发terminated
3. **Episode运行到motion结束**（400步），被motion_end触发truncated
4. **WandB记录的episode_len = 400**

**HDMI为什么是80？**
1. **Motion长度可能相同**（400步）
2. **训练初期/中期tracking质量差**，在80步就触发terminated
3. **Episode在80步提前结束**，没有等到motion结束
4. **WandB记录的episode_len = 80**

---

## 验证方案

### 方案1: 禁用所有terminated条件

修改配置：

```yaml
termination:
  cum_body_pos_error_local:
    enabled: false
  cum_body_z_error:
    enabled: false
```

**运行训练**，观察`train/state/episode_len`：
- 如果变成400 → Motion长度确实是400，之前是因为terminated提前结束
- 如果还是80 → Motion长度就是80

### 方案2: 打印Motion长度

在`command.py:206`后添加：

```python
if len(env_ids) == self.num_envs:  # 第一次采样
    print(f"[Motion Lengths] min={motion_len.min()}, "
          f"mean={motion_len.mean():.1f}, "
          f"max={motion_len.max()}")
    print(f"Corresponding time @ 50Hz: {motion_len.mean() * 0.02:.2f}s")
```

### 方案3: 打印终止统计

在`base.py:885`后添加：

```python
if done.any() and self.timestamp % 100 == 0:
    done_mask = done.squeeze(-1)
    terminated_mask = terminated.squeeze(-1)
    truncated_mask = truncated.squeeze(-1)

    print(f"[Episode End Stats @ step {self.timestamp}]")
    print(f"  Done: {done_mask.sum()} envs")
    print(f"  - Terminated (failed): {terminated_mask.sum()} envs")
    print(f"  - Truncated (timeout/motion_end): {truncated_mask.sum()} envs")
    print(f"  Episode lengths: min={self.episode_length_buf[done_mask].min()}, "
          f"mean={self.episode_length_buf[done_mask].float().mean():.1f}, "
          f"max={self.episode_length_buf[done_mask].max()}")
```

---

## 总结

### 关键发现

1. ✅ **HDMI已经有motion_end终止条件**
   - 通过`command_manager.finished`属性实现
   - 在`base.py:880`中被使用

2. ✅ **实现方式与TWIST-MAIN等价**
   - TWIST: `episode_length * dt >= motion_length`
   - HDMI: `self.t >= self.motion_len`
   - 效果相同

3. ⚠️ **Episode Length差异的真正原因**
   - 不是缺少motion_end
   - 而是HDMI在motion结束前就被terminated（tracking误差超限）
   - TWIST的400可能是训练后期的结果，tracking质量好才能跑完整个motion

4. ✅ **Termination配置已对齐**
   - threshold: 0.7, 0.2（已恢复到TWIST原始值）
   - min_steps: 2（模拟first_step保护）

### 下一步行动

1. **优先验证motion长度**：运行方案2，打印motion_len的实际值
2. **测试禁用terminated**：运行方案1，看episode_len是否增长到400
3. **对比训练阶段**：确认TWIST的400是训练哪个阶段观察到的

**如果方案1测试后episode_len变成400，说明问题解决！**
