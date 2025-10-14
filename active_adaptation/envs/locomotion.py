"""
机器人运动学习环境的核心实现文件
负责在Isaac Sim中设置和配置机器人运动学习环境
支持机器人运动控制、对象操作、接触传感器等功能的集成
"""

import os
import json
import torch
from isaaclab.utils import configclass

import active_adaptation
from active_adaptation.envs.base import _Env

class SimpleEnv(_Env):
    """
    机器人运动学习环境类
    继承自_Env基类，专门用于处理机器人运动控制任务
    支持Isaac Sim和MuJoCo两种仿真后端
    """
    def __init__(self, cfg):
        """
        初始化环境
        
        Args:
            cfg: 环境配置对象，包含机器人、任务、传感器等配置信息
        """
        super().__init__(cfg)
        # 获取场景中的机器人对象引用
        self.robot = self.scene.articulations["robot"]
        
        # 如果使用Isaac Sim后端且有GUI，设置可视化界面
        if self.backend == "isaac" and self.sim.has_gui():
            from isaaclab.envs.ui import BaseEnvWindow, ViewportCameraController
            from isaaclab.envs import ViewerCfg
            
            # 计算最接近观察点的环境索引（用于GUI显示）
            # hacks to make IsaacLab happy. we don't use them.
            self.lookat_env_i = (
                self.scene._default_env_origins.cpu() 
                - torch.tensor(self.cfg.viewer.lookat)
            ).norm(dim=-1).argmin().item()
            self.cfg.viewer.env_index = self.lookat_env_i
            self.manager_visualizers = {}
            
            # 创建GUI窗口和摄像头控制器
            self.window = BaseEnvWindow(self, window_name="IsaacLab")
            self.viewport_camera_controller = ViewportCameraController(
                self,
                ViewerCfg(self.cfg.viewer.eye, self.cfg.viewer.lookat, origin_type="env")
            )

            # 设置摄像头视角
            look_at_env_id = self.lookat_env_i
            self.sim.set_camera_view(
                eye=self.scene.env_origins[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.eye),
                target=self.scene.env_origins[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.lookat)
            )

        # 初始化动作缓冲区
        self.action_buf: torch.Tensor = self.action_manager.action_buf
        self.last_action: torch.Tensor = self.action_manager.applied_action

    def setup_scene(self):
        """
        设置仿真场景
        根据配置动态构建Isaac Sim或MuJoCo仿真环境
        包括机器人、对象、传感器、地形等所有场景元素
        """
        import active_adaptation.envs.scene as scene

        # Isaac Sim后端配置
        if active_adaptation.get_backend() == "isaac":
            import isaaclab.sim as sim_utils
            from isaaclab.scene import InteractiveSceneCfg
            from isaaclab.assets import AssetBaseCfg, ArticulationCfg
            from isaaclab.sensors import ContactSensorCfg
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
            from active_adaptation.assets import ROBOTS, OBJECTS, get_asset_meta
            from active_adaptation.envs.terrain import TERRAINS
            
            # 创建交互式场景配置
            env_spacing = self.cfg.viewer.get("env_spacing", 2.0)  # 环境间距
            scene_cfg = InteractiveSceneCfg(
                num_envs=self.cfg.num_envs, 
                env_spacing=env_spacing, 
                replicate_physics=False
            )
            
            # 配置天空光照
            scene_cfg.sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
            )
            
            # 配置机器人
            scene_cfg.robot: ArticulationCfg = ROBOTS[self.cfg.robot.name]
            
            # 如果配置中有机器人参数覆盖，应用覆盖参数
            if hasattr(self.cfg.robot, 'override_params'):
                from active_adaptation.utils import update_class_from_dict
                update_class_from_dict(scene_cfg.robot, self.cfg.robot.override_params, _ns="")
            
            # 设置机器人路径和类型
            scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"  # ENV_REGEX_NS = "/World/envs/env_.*"
            robot_type = self.cfg.robot.get("robot_type", self.cfg.robot.name)
            scene_cfg.robot.spawn.usd_path = scene_cfg.robot.spawn.usd_path.format(ROBOT_TYPE=robot_type)

            # 对象配置：如果任务需要操作对象（如suitcase、box等）
            if "object_asset_name" in self.cfg.command:
                # 添加额外对象（如果配置中指定了额外对象）
                extra_object_names = self.cfg.command.get("extra_object_names", [])
                for extra_obj_name in extra_object_names:
                    extra_obj_cfg = OBJECTS[extra_obj_name]
                    extra_obj_cfg.prim_path = "{ENV_REGEX_NS}/" + extra_obj_name
                    setattr(scene_cfg, extra_obj_name, extra_obj_cfg)

                # 获取主要操作对象配置
                obj_name = self.cfg.command.object_asset_name  # 对象名称，如"suitcase"
                obj_contact_body_name = self.cfg.command.object_body_name  # 对象接触体名称

                # 配置主要对象
                obj_cfg = OBJECTS[obj_name]  # 从OBJECTS字典获取对象配置
                obj_cfg.prim_path = "{ENV_REGEX_NS}/" + obj_name  # 设置对象路径
                obj_type = self.cfg.command.get("object_type", obj_name)  # 获取对象类型
                obj_cfg.spawn.usd_path = obj_cfg.spawn.usd_path.format(OBJECT_TYPE=obj_type)
                print(f"Using object type {obj_type} with asset {obj_cfg.spawn.usd_path}")
                setattr(scene_cfg, obj_name, obj_cfg)  # 动态设置对象属性

                # 为对象操作任务添加接触传感器
                eef_names = self.cfg.command.get("contact_eef_body_name", [])  # 获取末端执行器名称列表
                contact_geom_prim_path = "{ENV_REGEX_NS}/" + obj_name + "/" + obj_contact_body_name  # 构建对象几何体路径
                # ENV_REGEX_NS = "/World/envs/env_.*" 用于多环境并行仿真

                # 为每个末端执行器创建专用接触传感器
                for eef_name in eef_names:
                    contact_sensor_name = f"{eef_name}_{obj_name}_contact_forces"  # 传感器名称，如"left_wrist_yaw_link_suitcase_contact_forces"
                    eef_prim_path = "{ENV_REGEX_NS}/Robot/" + eef_name  # 末端执行器路径
                    setattr(scene_cfg, contact_sensor_name, ContactSensorCfg(  # 对象专用传感器配置
                        prim_path=eef_prim_path,  # 传感器安装位置（机器人末端执行器）
                        history_length=0,  # 不保存历史数据（接触检测只需要当前时刻）
                        track_air_time=False,  # 不跟踪空中时间
                        filter_prim_paths_expr=[contact_geom_prim_path],  # 只监测与特定对象的接触
                    ))
                    
            # 随机化配置：为训练添加随机化以提高模型泛化能力
            body_scale_rand = self.cfg.randomization.get("body_scale", None)
            if body_scale_rand is not None:
                from active_adaptation.assets.spawn import clone
                asset = getattr(scene_cfg, body_scale_rand.name)  # 获取要随机化的资产
                spawn_func = asset.spawn.func.__wrapped__  # 获取原始生成函数
                asset.spawn.func = clone(spawn_func)  # 使用克隆函数包装
                asset.spawn.scale_range = tuple(body_scale_rand.scale_range)  # 设置缩放范围
                asset.spawn.homogeneous_scale = body_scale_rand.get("homogeneous_scale", False)  # 是否均匀缩放
                print(f"Randomized {body_scale_rand.name} scale to {asset.spawn.scale_range}")

            # 配置地形
            scene_cfg.terrain = TERRAINS[self.cfg.terrain]
            
            # 配置全局接触传感器（用于步态分析和平衡控制）
            scene_cfg.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*(ankle_roll|wrist_.*)_link",  # 监测脚踝和手腕链接
                history_length=3,  # 保存3帧历史数据
                track_air_time=True  # 跟踪空中时间（用于步态分析）
            )
            # 注意：全局传感器使用固定名称"contact_forces"，对象专用传感器使用动态名称 
            # 摄像头配置：可选配置摄像头用于视觉观察
            if self.cfg.get("enable_cameras", False):
                from isaaclab.sensors import TiledCameraCfg
                
                # 配置摄像头参数
                camera_spawn_cfg = sim_utils.PinholeCameraCfg(
                    focal_length=7.6,  # 焦距
                    focus_distance=400.0,  # 对焦距离
                    horizontal_aperture=20.0,  # 水平光圈
                    clipping_range=(0.1, 1.0e5),  # 裁剪范围
                )
                
                # 配置平铺摄像头（支持多环境）
                tiled_camera: TiledCameraCfg = TiledCameraCfg(
                    prim_path="/World/envs/env_.*/Robot/d435_link/front_cam",  # 摄像头安装路径
                    spawn=camera_spawn_cfg,
                    offset=TiledCameraCfg.OffsetCfg(
                        pos=(0.0, 0.0, 0.0),  # 位置偏移
                        rot=(0.5, -0.5, 0.5, -0.5),  # 旋转偏移（四元数）
                        convention="ros"  # 坐标系约定
                    ),
                    # 数据类型：RGB、深度、距离图像
                    data_types=["rgb", "depth", "distance_to_image_plane"],
                    update_latest_camera_pose=True,  # 更新最新摄像头姿态
                    update_period=0.02,  # 更新周期
                    width=self.cfg.camera_width,  # 图像宽度
                    height=self.cfg.camera_height,  # 图像高度
                )
                scene_cfg.tiled_camera = tiled_camera
            
            # 光线投射器配置（可选，用于激光雷达等传感器）
            # if self.cfg.get("enable_raycaster", False):
            #     from isaaclab.sensors import RayCasterCfg
            #     raycaster = RayCasterCfg(
            #         prim_path="/World/envs/env_.*/Robot/d435_link/front_cam",
            #         update_period=0.02,
            #         offset=RayCasterCfg.OffsetCfg(
            #             pos=(0.0, 0.0, 0.0),
            #             rot=(0.5, -0.5, 0.5, -0.5),
            #         ),
            #     )
            #     scene_cfg.raycaster = raycaster
            
            # 配置物理仿真参数
            sim_cfg = sim_utils.SimulationCfg(
                dt=self.cfg.sim.isaac_physics_dt,  # 物理时间步长
                render=sim_utils.RenderCfg(
                    rendering_mode="quality",  # 渲染质量
                    # antialiasing_mode="FXAA",  # 抗锯齿模式
                    # enable_global_illumination=True,  # 全局光照
                    # enable_reflections=True,  # 反射效果
                ),
                device=f"cuda:{active_adaptation.get_local_rank()}"  # GPU设备
            )
            
            # GPU内存优化配置
            # slightly reduces GPU memory usage
            # sim_cfg.physx.gpu_max_rigid_contact_count = 2**21
            # sim_cfg.physx.gpu_max_rigid_patch_count = 2**21
            sim_cfg.physx.gpu_found_lost_pairs_capacity = 2538320  # 2**20 找到/丢失对容量
            sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 61999079  # 2**26 聚合对容量
            sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2**23  # 总聚合对容量
            sim_cfg.physx.enable_stabilization = False  # 禁用稳定化（提高性能）
            # sim_cfg.physx.gpu_collision_stack_size = 2**25
            # sim_cfg.physx.gpu_heap_capacity = 2**24
            
            # 创建Isaac Lab仿真和场景
            self.sim, self.scene = scene.create_isaaclab_sim_and_scene(sim_cfg, scene_cfg)

            # 设置摄像头视角
            self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
            
            # 配置渲染产品（用于数据收集和可视化）
            try:
                import omni.replicator.core as rep
                # 创建渲染产品
                self._render_product = rep.create.render_product(
                    "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
                )
                # 创建RGB注释器，用于从渲染产品读取数据
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
                # 可选：实例分割注释器
                # self._seg_annotator = rep.AnnotatorRegistry.get_annotator(
                #     "instance_id_segmentation_fast", 
                #     device="cpu",
                # )
                # self._seg_annotator.attach([self._render_product])
                # for _ in range(4):
                #     self.sim.render()
            except ModuleNotFoundError:
                print("Set app.enable_cameras=true to use cameras.")
            
            # 初始化调试绘制工具
            try:
                from active_adaptation.utils.debug import DebugDraw
                self.debug_draw = DebugDraw()
                print("[INFO] Debug Draw API enabled.")
            except ModuleNotFoundError:
                print()
            
            # 保存资产元数据到JSON文件
            asset_meta = get_asset_meta(self.scene["robot"])
            path = os.path.join(os.getcwd(), "asset_meta.json")
            print(f"Saving asset meta to {path}")
            with open(path, "w") as f:
                json.dump(asset_meta, f, indent=4)
                
        else:
            # MuJoCo后端配置（备选仿真环境）
            from active_adaptation.envs.mujoco import MJScene, MJSim
            from active_adaptation.assets_mjcf import ROBOTS

            @configclass
            class SceneCfg:
                robot = ROBOTS[self.cfg.robot.name]
                contact_forces = "robot"
            
            self.scene = MJScene(SceneCfg())
            self.sim = MJSim(self.scene)

        
    def _reset_idx(self, env_ids: torch.Tensor):
        """
        重置指定环境中的机器人状态
        
        Args:
            env_ids: 要重置的环境ID张量
        """
        # 从命令管理器获取初始状态
        init_root_state = self.command_manager.sample_init(env_ids)
        if init_root_state is not None and not self.robot.is_fixed_base:
            # 将初始状态写入仿真
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        # 重置统计信息
        self.stats[env_ids] = 0.

    def render(self, mode: str="human"):
        """
        渲染环境
        
        Args:
            mode: 渲染模式，默认为"human"
            
        Returns:
            渲染结果
        """
        # 可选：动态调整摄像头视角跟随机器人
        # look_at_env_id = self.lookat_env_i
        # self.sim.set_camera_view(
        #     eye=self.robot.data.root_pos_w[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.eye),
        #     target=self.robot.data.root_pos_w[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.lookat)
        # )
        return super().render(mode)

