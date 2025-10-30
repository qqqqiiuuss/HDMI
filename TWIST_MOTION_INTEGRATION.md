# TWIST Motion Dataset Integration Guide

本文档说明如何在 HDMI 框架中使用 TWIST 的 motion data。

## 概述

`TwistMotionDataset` 类提供了与 HDMI `MotionDataset` 完全兼容的接口，同时保留了 TWIST 的所有预处理逻辑：

✅ **保留 TWIST 特性**：
- 19 点 box filter 平滑
- 有限差分速度计算
- 四元数差分角速度
- 单一张量拼接存储
- 带权重的 motion 采样

✅ **兼容 HDMI 接口**：
- `get_slice(motion_ids, starts, steps)`
- `find_joints()` / `find_bodies()`
- `to(device)` 设备迁移
- 与 `RobotTracking` 完全兼容

---

## 1. 准备 TWIST Motion Data

### 1.1 数据格式要求

TWIST motion data 应该是 **pickle (.pkl) 文件**，包含以下字段：

```python
motion_data = {
    "fps": 50,                          # 帧率
    "root_pos": np.ndarray,             # (T, 3) 根位置
    "root_rot": np.ndarray,             # (T, 4) 根旋转 [w, x, y, z]
    "dof_pos": np.ndarray,              # (T, num_dof) DOF 位置
    "local_body_pos": np.ndarray,       # (T, num_bodies, 3) 局部关键点位置
    "link_body_list": List[str],        # 身体链接名称列表
    "joint_names": List[str],           # 关节名称列表
}
```

### 1.2 创建 YAML 配置文件

在 HDMI 项目根目录下创建 `twist_motions.yaml`：

```yaml
# twist_motions.yaml
root_path: "/path/to/twist/motions"  # TWIST motion 数据目录

motions:
  - file: "walk_forward.pkl"
    weight: 1.0

  - file: "run.pkl"
    weight: 2.0  # 更高权重，更可能被采样

  - file: "jump.pkl"
    weight: 0.5

  - file: "crawl.pkl"
    weight: 1.0
```

**说明**：
- `root_path`: motion pkl 文件所在的目录
- `file`: 相对于 `root_path` 的文件路径
- `weight`: 采样权重（可选，默认 1.0）

---

## 2. 在 HDMI 中使用

### 2.1 方法 A: 直接在 Command 中使用

修改 `active_adaptation/envs/mdp/commands/hdmi/command.py`：

```python
from active_adaptation.utils.twist_motion import TwistMotionDataset

class RobotTracking(Command):
    def __init__(
        self, env,
        data_path: str,  # 改为指向 YAML 文件
        use_twist_motion: bool = True,  # 新增参数
        **kwargs
    ):
        super().__init__(env)

        if use_twist_motion:
            # 使用 TWIST motion dataset
            self.dataset = TwistMotionDataset.create_from_yaml(
                yaml_path=data_path,
                device=self.device,
                smooth_window=19  # TWIST 默认平滑窗口
            ).to(self.device)
        else:
            # 使用 HDMI 原生 motion dataset
            self.dataset = MotionDataset.create_from_path(
                root_path=data_path,
                isaac_joint_names=self.asset.joint_names,
                target_fps=int(1/self.env.step_dt)
            ).to(self.device)

        # 其余代码保持不变...
```

**配置文件修改** (`cfg/task/G1/hdmi/your_task.yaml`)：

```yaml
command:
  _target_: active_adaptation.envs.mdp.commands.hdmi.command.RobotTracking
  data_path: "twist_motions.yaml"  # 指向 YAML 配置
  use_twist_motion: true            # 启用 TWIST motion

  tracking_keypoint_names: [".*ankle.*", ".*wrist.*"]
  tracking_joint_names: [".*"]
  # ... 其余配置保持不变
```

---

### 2.2 方法 B: 创建专用的 TWIST Command

创建新的 command 类 `active_adaptation/envs/mdp/commands/hdmi/twist_command.py`：

```python
from active_adaptation.envs.mdp.commands.hdmi.command import RobotTracking
from active_adaptation.utils.twist_motion import TwistMotionDataset

class TwistRobotTracking(RobotTracking):
    """
    使用 TWIST motion 的机器人跟踪命令

    与 RobotTracking 完全相同，但强制使用 TwistMotionDataset
    """

    def __init__(
        self, env,
        data_path: str,  # YAML 配置文件路径
        smooth_window: int = 19,  # TWIST 平滑窗口
        **kwargs
    ):
        # 临时存储参数
        self._data_path = data_path
        self._smooth_window = smooth_window

        # 调用父类，跳过 dataset 初始化
        super().__init__(env, data_path="", **kwargs)

    def _init_dataset(self):
        """重写 dataset 初始化方法"""
        self.dataset = TwistMotionDataset.create_from_yaml(
            yaml_path=self._data_path,
            device=self.device,
            smooth_window=self._smooth_window
        ).to(self.device)
```

**注意**: 这需要在 `RobotTracking.__init__` 中将 dataset 初始化提取为 `_init_dataset()` 方法。

---

## 3. 接口对比

### 3.1 核心接口对比

| 功能 | HDMI MotionDataset | TwistMotionDataset |
|-----|-------------------|-------------------|
| **加载方式** | `create_from_path(root_path)` | `create_from_yaml(yaml_path)` |
| **数据结构** | MotionData | TwistMotionData |
| **存储方式** | 独立 npz 文件 + 元数据 | 单一拼接张量 |
| **速度计算** | 直接从 npz 读取 | 有限差分 + 19 点平滑 |
| **采样方式** | 均匀随机 `torch.randint` | 带权重 `torch.multinomial` |

### 3.2 数据属性对比

| 属性 | HDMI | TWIST | 说明 |
|-----|------|-------|------|
| `body_pos_w` | ✅ | ❌ | HDMI: 世界坐标系 |
| `root_pos` | ❌ | ✅ | TWIST: 根位置 |
| `body_quat_w` | ✅ | ❌ | HDMI: 世界坐标系 |
| `root_rot` | ❌ | ✅ | TWIST: 根旋转（四元数） |
| `joint_pos` | ✅ | ❌ | HDMI: 关节位置 |
| `dof_pos` | ❌ | ✅ | TWIST: DOF 位置 |
| `body_lin_vel_w` | ✅ | ❌ | HDMI: 从 npz 读取 |
| `root_vel` | ❌ | ✅ | TWIST: 计算 + 平滑 |
| `body_ang_vel_w` | ✅ | ❌ | HDMI: 从 npz 读取 |
| `root_ang_vel` | ❌ | ✅ | TWIST: 四元数差分 + 平滑 |
| `local_key_body_pos` | ❌ | ✅ | TWIST 特有 |

---

## 4. 适配器模式（推荐）

如果需要完全兼容现有代码，可以创建一个适配器类：

创建 `active_adaptation/utils/twist_motion_adapter.py`：

```python
from active_adaptation.utils.twist_motion import TwistMotionDataset, TwistMotionData
from active_adaptation.utils.motion import MotionData
import torch

class TwistToHDMIAdapter:
    """
    将 TwistMotionData 适配为 HDMI MotionData

    自动转换坐标系和数据格式
    """

    def __init__(self, twist_dataset: TwistMotionDataset):
        self.twist_dataset = twist_dataset

    def get_slice(self, motion_ids, starts, steps):
        """获取切片并转换为 HDMI 格式"""
        twist_data = self.twist_dataset.get_slice(motion_ids, starts, steps)

        # 转换为 HDMI 格式
        # 假设 root = body[0]
        body_pos_w = torch.zeros(
            (*twist_data.root_pos.shape[:-1], len(self.twist_dataset.body_names), 3),
            device=twist_data.root_pos.device
        )
        body_pos_w[..., 0, :] = twist_data.root_pos

        body_quat_w = torch.zeros(
            (*twist_data.root_rot.shape[:-1], len(self.twist_dataset.body_names), 4),
            device=twist_data.root_rot.device
        )
        body_quat_w[..., 0, :] = twist_data.root_rot

        body_lin_vel_w = torch.zeros_like(body_pos_w)
        body_lin_vel_w[..., 0, :] = twist_data.root_vel

        body_ang_vel_w = torch.zeros_like(body_pos_w)
        body_ang_vel_w[..., 0, :] = twist_data.root_ang_vel

        # 创建 HDMI MotionData
        return MotionData(
            motion_id=twist_data.motion_id,
            step=twist_data.step,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            joint_pos=twist_data.dof_pos,
            joint_vel=twist_data.dof_vel,
            batch_size=twist_data.batch_size
        )

    def __getattr__(self, name):
        """代理其他方法到 twist_dataset"""
        return getattr(self.twist_dataset, name)
```

**使用方式**：

```python
from active_adaptation.utils.twist_motion import TwistMotionDataset
from active_adaptation.utils.twist_motion_adapter import TwistToHDMIAdapter

# 加载 TWIST dataset
twist_dataset = TwistMotionDataset.create_from_yaml("twist_motions.yaml", device="cuda")

# 包装为 HDMI 兼容接口
self.dataset = TwistToHDMIAdapter(twist_dataset)

# 现在可以像使用 HDMI MotionDataset 一样使用
motion_data = self.dataset.get_slice(motion_ids, starts, steps)
```

---

## 5. 完整使用示例

### 5.1 准备数据

```bash
# 假设你的 TWIST motion 数据在:
/data/twist_motions/
├── walk_forward.pkl
├── run.pkl
└── jump.pkl
```

### 5.2 创建 YAML 配置

```yaml
# config/twist_motions.yaml
root_path: "/data/twist_motions"
motions:
  - file: "walk_forward.pkl"
    weight: 1.0
  - file: "run.pkl"
    weight: 2.0
  - file: "jump.pkl"
    weight: 1.5
```

### 5.3 修改任务配置

```yaml
# cfg/task/G1/hdmi/twist_test.yaml
defaults:
  - /task/G1/hdmi/base/hdmi-base

command:
  _target_: active_adaptation.envs.mdp.commands.hdmi.command.RobotTracking
  data_path: "config/twist_motions.yaml"

  tracking_keypoint_names: ["pelvis", ".*ankle.*", ".*wrist.*"]
  tracking_joint_names: [".*"]

  root_body_name: "pelvis"
  reset_range: null  # 随机起始时间

  future_steps: [1, 2, 8, 16]
  sample_motion: true  # 启用采样
```

### 5.4 修改 Command 类

在 `active_adaptation/envs/mdp/commands/hdmi/command.py` 的 `__init__` 中：

```python
from active_adaptation.utils.twist_motion import TwistMotionDataset

def __init__(self, env, data_path: str, ...):
    super().__init__(env)

    # 检测是否为 YAML 文件
    if data_path.endswith('.yaml'):
        # 使用 TWIST motion
        self.dataset = TwistMotionDataset.create_from_yaml(
            yaml_path=data_path,
            device=self.device,
            smooth_window=19
        ).to(self.device)
    else:
        # 使用 HDMI motion
        self.dataset = MotionDataset.create_from_path(
            root_path=data_path,
            isaac_joint_names=self.asset.joint_names,
            target_fps=int(1/self.env.step_dt)
        ).to(self.device)

    # 其余代码保持不变...
```

### 5.5 运行训练

```bash
python scripts/train.py \
    algo=ppo_roa_train \
    task=G1/hdmi/twist_test
```

---

## 6. 注意事项

### 6.1 坐标系差异

⚠️ **重要**: TWIST 和 HDMI 使用不同的坐标系：

| 项目 | TWIST | HDMI |
|-----|-------|------|
| 根状态 | `root_pos`, `root_rot` | `body_pos_w[0]`, `body_quat_w[0]` |
| 关节 | `dof_pos` (DOF) | `joint_pos` (关节角度) |
| 关键点 | `local_key_body_pos` (局部) | `body_pos_w` (世界) |

如果需要在奖励函数中使用世界坐标系的关键点位置，需要从局部坐标转换：

```python
# 在 Command.update() 中
world_key_body_pos = self._local_to_world(
    self.dataset.data.local_key_body_pos,
    self.dataset.data.root_pos,
    self.dataset.data.root_rot
)
```

### 6.2 速度计算

TWIST 的速度是通过有限差分 + 19 点平滑计算的，与 HDMI 从 npz 直接读取的速度可能有差异。如果对速度精度要求高，可以考虑：

1. 在 motion 预处理时统一计算速度
2. 调整 `smooth_window` 参数（默认 19）
3. 使用中心差分代替前向差分

### 6.3 性能考虑

- **内存**: TWIST 将所有 motion 拼接成单一张量，可能占用更多显存
- **加载时间**: 首次加载需要计算速度和平滑，比 HDMI 稍慢
- **运行时**: 两者性能相近，TWIST 可能因为张量拼接在采样时略快

---

## 7. 故障排查

### 问题 1: 找不到 `twist_motion` 模块

**解决方案**: 确保 `twist_motion.py` 在正确路径：
```bash
ls active_adaptation/utils/twist_motion.py
```

### 问题 2: pkl 文件格式错误

**解决方案**: 检查 pkl 文件是否包含所有必需字段：
```python
import pickle
with open("motion.pkl", "rb") as f:
    data = pickle.load(f)
    print(data.keys())  # 应包含 fps, root_pos, root_rot, dof_pos, local_body_pos
```

### 问题 3: 设备不匹配

**解决方案**: 确保在创建后调用 `.to(device)`:
```python
dataset = TwistMotionDataset.create_from_yaml(...).to(self.device)
```

### 问题 4: 关节名称不匹配

**解决方案**: 在 pkl 文件中添加 `joint_names` 字段，或者在 `find_joints` 时使用正则表达式：
```python
self.tracking_joint_names = self.asset.find_joints(".*hip.*|.*knee.*")[1]
```

---

## 8. 扩展功能

### 8.1 添加课程学习

修改 `sample_motions` 方法支持难度等级：

```python
class TwistMotionDataset:
    def __init__(self, ..., difficulties=None):
        self.difficulties = difficulties or torch.ones(len(starts))

    def sample_motions(self, n: int, max_difficulty: float = None):
        if max_difficulty is not None:
            # 只采样难度低于阈值的 motion
            valid_mask = self.difficulties <= max_difficulty
            valid_weights = self.weights * valid_mask.float()
            valid_weights /= valid_weights.sum()
            return torch.multinomial(valid_weights, n, replacement=True)
        return torch.multinomial(self.weights, n, replacement=True)
```

### 8.2 添加文本描述

在 pkl 文件中添加描述，支持语言条件：

```python
motion_data["description"] = "A human walking forward slowly"
```

---

## 9. 性能优化建议

1. **预计算缓存**: 将处理后的数据保存为新的 pkl，避免每次重新计算
2. **调整平滑窗口**: 根据 motion 质量调整 `smooth_window`（默认 19）
3. **使用 FP16**: 在显存受限时使用半精度：
   ```python
   data = data.to(dtype=torch.float16)
   ```

---

## 10. 总结

✅ **优势**:
- 保留 TWIST 的所有预处理逻辑
- 完全兼容 HDMI 接口
- 支持带权重的 motion 采样
- 数据拼接提升采样效率

⚠️ **限制**:
- 需要手动转换坐标系（如果使用世界坐标）
- 速度计算方式与 HDMI 不同
- 首次加载时间较长

📝 **推荐使用场景**:
- 已有 TWIST 训练的 motion data
- 需要带权重的 motion 采样
- 对速度平滑有较高要求
