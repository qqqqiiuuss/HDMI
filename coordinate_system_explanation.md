# GMR local_body_pos vs LocoMujoco body_pos_w 坐标系转换详解

## 核心区别

### GMR `local_body_pos`
- **坐标系**: 相对于根部(pelvis)的局部坐标系
- **根部位置**: 始终为 `[0, 0, 0]` 或接近零
- **含义**: 描述机器人的"形状"和关节配置
- **特点**: 与机器人在世界中的位置无关
- **计算方式**: `forward_kinematics(root_pos=0, root_rot=identity, dof_pos)`

### LocoMujoco `body_pos_w`
- **坐标系**: 世界坐标系 (world coordinates)
- **根部位置**: 机器人在世界中的实际位置
- **含义**: 机器人各部位在世界中的绝对位置
- **特点**: 包含机器人的世界位置和姿态
- **用途**: 直接用于物理仿真和可视化

## 数据格式对比

```python
# GMR PKL 格式
{
    "root_pos": [T, 3],           # 根部的世界位置
    "root_rot": [T, 4],           # 根部的世界旋转 (xyzw)
    "dof_pos": [T, N_joints],     # 关节角度
    "local_body_pos": [T, N_bodies, 3],  # 相对于根部的局部位置
}

# LocoMujoco NPZ 格式
{
    "body_pos_w": [T, N_bodies, 3],      # 世界坐标系中的绝对位置
    "body_quat_w": [T, N_bodies, 4],     # 世界坐标系中的绝对旋转 (wxyz)
    "joint_pos": [T, N_joints],          # 关节角度
    # ... 其他数据
}
```

## 转换公式

### 位置转换
```python
body_pos_w[t] = root_rotation[t].apply(local_body_pos[t]) + root_pos[t]
```

### 详细步骤
1. **旋转变换**: 将局部坐标通过根部旋转变换到世界坐标系方向
2. **平移变换**: 加上根部在世界坐标系中的位置

## 代码实现

```python
from scipy.spatial.transform import Rotation

def convert_gmr_to_locomujoco_positions(root_pos, root_rot, local_body_pos):
    """
    将GMR的local_body_pos转换为LocoMujoco的body_pos_w
    
    Args:
        root_pos: [T, 3] 根部世界位置
        root_rot: [T, 4] 根部世界旋转 (xyzw格式)
        local_body_pos: [T, N_bodies, 3] 局部身体位置
    
    Returns:
        body_pos_w: [T, N_bodies, 3] 世界坐标系身体位置
    """
    T, N_bodies, _ = local_body_pos.shape
    body_pos_w = np.zeros_like(local_body_pos)
    
    # 创建旋转对象
    root_rotations = Rotation.from_quat(root_rot)  # xyzw格式
    
    for t in range(T):
        # 应用旋转和平移变换
        rotated_pos = root_rotations[t].apply(local_body_pos[t])
        body_pos_w[t] = rotated_pos + root_pos[t]
    
    return body_pos_w
```

## 验证方法

### 1. 根部位置验证
```python
# GMR中根部在局部坐标系中应该是原点
assert np.allclose(local_body_pos[:, 0], 0, atol=1e-3)

# 转换后根部应该等于root_pos
assert np.allclose(body_pos_w[:, 0], root_pos, atol=1e-3)
```

### 2. 相对距离保持
```python
# 任意两个身体部位之间的距离在转换前后应该保持不变
for t in range(T):
    local_dist = np.linalg.norm(local_body_pos[t, i] - local_body_pos[t, j])
    world_dist = np.linalg.norm(body_pos_w[t, i] - body_pos_w[t, j])
    assert np.isclose(local_dist, world_dist, atol=1e-6)
```

## 常见问题和解决方案

### 1. 根部位置不匹配
**问题**: 转换后的根部位置与期望的root_pos不一致
**原因**: GMR的local_body_pos中根部不是严格的[0,0,0]
**解决**: 检查并调整local_body_pos的根部位置

```python
# 确保根部位置为零
local_body_pos[:, 0] = 0
```

### 2. 旋转方向错误
**问题**: 转换后的身体部位位置看起来"镜像"了
**原因**: 四元数格式不匹配或旋转方向理解错误
**解决**: 检查四元数格式(xyzw vs wxyz)和旋转方向

### 3. 尺度不匹配
**问题**: 转换后的机器人看起来太大或太小
**原因**: GMR和LocoMujoco使用不同的单位系统
**解决**: 应用适当的尺度因子

## 实际应用示例

### 转换G1机器人数据
```python
# 加载GMR数据
with open("robot_motion.pkl", "rb") as f:
    data = pickle.load(f)

root_pos = data["root_pos"]
root_rot = data["root_rot"]  # xyzw格式
local_body_pos = data["local_body_pos"]

# 转换为LocoMujoco格式
body_pos_w = convert_gmr_to_locomujoco_positions(
    root_pos, root_rot, local_body_pos
)

# 保存为NPZ格式
np.savez_compressed(
    "motion.npz",
    body_pos_w=body_pos_w,
    # ... 其他数据
)
```

## 可视化对比

转换前后的数据应该在可视化中表现一致：

1. **GMR可视化**: 使用`root_pos`, `root_rot`, `dof_pos`
2. **LocoMujoco可视化**: 使用转换后的`body_pos_w`, `body_quat_w`

两种可视化应该显示完全相同的机器人运动。

## 总结

GMR的`local_body_pos`和LocoMujoco的`body_pos_w`的主要区别在于坐标系：

- **GMR**: 局部坐标系，描述机器人形状
- **LocoMujoco**: 世界坐标系，描述绝对位置

转换的核心是通过根部的旋转和平移将局部坐标变换到世界坐标。正确的转换确保了数据在不同系统间的一致性和可用性。







