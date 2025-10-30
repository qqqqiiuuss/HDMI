#!/usr/bin/env python3
"""
生成 AMASS 数据集的 YAML 配置文件

该脚本扫描指定目录下的所有 PKL 文件，并生成符合 TWIST 格式的 YAML 配置。

用法:
    python scripts/generate_amass_yaml.py

输出:
    cfg/task/G1/twist/amass_dataset_generated.yaml
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict


def find_all_pkl_files(root_dir: str) -> List[str]:
    """
    递归查找目录下所有 PKL 文件

    Args:
        root_dir: 根目录路径

    Returns:
        相对于根目录的 PKL 文件路径列表
    """
    root_path = Path(root_dir)
    pkl_files = []

    # 递归查找所有 .pkl 文件
    for pkl_file in root_path.rglob("*.pkl"):
        # 计算相对路径
        relative_path = pkl_file.relative_to(root_path)
        pkl_files.append(str(relative_path))

    return sorted(pkl_files)


def normalize_path(path: str) -> str:
    """
    规范化路径，将反斜杠转换为正斜杠

    Args:
        path: 原始路径

    Returns:
        规范化后的路径
    """
    return path.replace("\\", "/")


def get_motion_category(file_path: str) -> str:
    """
    根据文件路径推断运动类别

    Args:
        file_path: 文件路径

    Returns:
        运动类别描述
    """
    path_lower = file_path.lower()

    # 定义类别映射
    category_keywords = {
        "walk": "walking",
        "run": "running",
        "jump": "jumping",
        "dance": "dancing",
        "martial": "martial arts",
        "punch": "punching",
        "kick": "kicking",
        "gesture": "gestures",
        "crouch": "crouching",
        "hop": "hopping",
        "skip": "skipping",
        "turn": "turning",
        "stand": "standing transitions",
        "sit": "sitting",
        "lie": "lying down",
        "climb": "climbing",
        "crawl": "crawling",
    }

    # 检查路径中是否包含关键词
    for keyword, category in category_keywords.items():
        if keyword in path_lower:
            return category

    # 默认类别
    return "general movement"


def generate_yaml_config(
    root_path: str,
    pkl_files: List[str],
    output_file: str,
    default_weight: float = 1.0
) -> None:
    """
    生成 YAML 配置文件

    Args:
        root_path: 数据集根目录
        pkl_files: PKL 文件列表（相对路径）
        output_file: 输出 YAML 文件路径
        default_weight: 默认权重
    """
    # 构建配置数据结构
    config = {
        "root_path": root_path,
        "motions": []
    }

    # 为每个 PKL 文件创建条目
    for pkl_file in pkl_files:
        # 规范化路径
        normalized_path = normalize_path(pkl_file)

        # 推断运动类别
        category = get_motion_category(normalized_path)

        # 创建条目
        motion_entry = {
            "description": category,
            "file": normalized_path,
            "weight": default_weight
        }

        config["motions"].append(motion_entry)

    # 写入 YAML 文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入头部注释
        f.write("# Auto-generated AMASS dataset configuration\n")
        f.write(f"# Total motions: {len(pkl_files)}\n")
        f.write(f"# Generated from: {root_path}\n\n")

        # 写入 root_path
        f.write(f"root_path: {root_path}\n")
        f.write("motions:\n\n")

        # 写入每个运动条目
        for motion in config["motions"]:
            f.write(f"- description: {motion['description']}\n")
            f.write(f"  file: {motion['file']}\n")
            f.write(f"  weight: {motion['weight']}\n")

    print(f"✅ 成功生成 YAML 配置文件: {output_file}")
    print(f"   包含 {len(pkl_files)} 个运动文件")


def print_statistics(pkl_files: List[str]) -> None:
    """
    打印数据集统计信息

    Args:
        pkl_files: PKL 文件列表
    """
    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)

    # 按类别统计
    category_counts = {}
    for pkl_file in pkl_files:
        category = get_motion_category(pkl_file)
        category_counts[category] = category_counts.get(category, 0) + 1

    # 按数量排序
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n总文件数: {len(pkl_files)}")
    print(f"\n按类别统计:")
    for category, count in sorted_categories:
        print(f"  {category:25s}: {count:4d} files")

    # 按顶层目录统计
    top_dir_counts = {}
    for pkl_file in pkl_files:
        top_dir = pkl_file.split(os.sep)[0] if os.sep in pkl_file else pkl_file
        top_dir_counts[top_dir] = top_dir_counts.get(top_dir, 0) + 1

    print(f"\n按顶层目录统计:")
    sorted_dirs = sorted(top_dir_counts.items(), key=lambda x: x[1], reverse=True)
    for dir_name, count in sorted_dirs[:10]:  # 只显示前10个
        print(f"  {dir_name:30s}: {count:4d} files")

    print("="*60 + "\n")


def main():
    """主函数"""
    # 配置参数
    AMASS_ROOT = "/home/ubuntu/DATA2/workspace/AMASS_G1"
    OUTPUT_FILE = "/home/ubuntu/DATA2/workspace/xmh/HDMI-main/cfg/task/G1/twist/amass_dataset_generated.yaml"

    print("="*60)
    print("AMASS 数据集 YAML 配置生成器")
    print("="*60)
    print(f"\n数据集目录: {AMASS_ROOT}")
    print(f"输出文件:   {OUTPUT_FILE}")

    # 检查目录是否存在
    if not os.path.exists(AMASS_ROOT):
        print(f"\n❌ 错误: 数据集目录不存在: {AMASS_ROOT}")
        return

    # 查找所有 PKL 文件
    print(f"\n正在扫描目录...")
    pkl_files = find_all_pkl_files(AMASS_ROOT)

    if not pkl_files:
        print(f"\n❌ 错误: 未找到任何 PKL 文件")
        return

    print(f"找到 {len(pkl_files)} 个 PKL 文件")

    # 打印统计信息
    print_statistics(pkl_files)

    # 生成 YAML 配置
    print("正在生成 YAML 配置文件...")
    generate_yaml_config(
        root_path=AMASS_ROOT,
        pkl_files=pkl_files,
        output_file=OUTPUT_FILE,
        default_weight=1.0
    )

    # 显示前几个示例
    print(f"\n前 5 个运动文件示例:")
    for i, pkl_file in enumerate(pkl_files[:5], 1):
        category = get_motion_category(pkl_file)
        print(f"  {i}. {pkl_file}")
        print(f"     类别: {category}")

    print(f"\n✨ 完成！可以在以下位置查看生成的配置文件:")
    print(f"   {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
