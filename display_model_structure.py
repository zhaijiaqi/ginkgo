#!/usr/bin/env python3
"""
显示模型结构的脚本
读取保存的模型文件并显示网络结构
"""

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.ppo_agent_factory import CGPPOAgent
import pfrl


def load_and_display_model(model_path: str):
    """
    加载模型并显示网络结构

    Args:
        model_path: 模型文件路径（不包含.pt后缀）
    """
    print("=" * 60)
    print(f"加载模型: {model_path}")
    print("=" * 60)

    # 加载元数据
    metadata_path = f"{model_path}.pt"
    if not os.path.exists(metadata_path):
        print(f"错误：找不到元数据文件 {metadata_path}")
        return

    print(f"加载元数据文件: {metadata_path}")
    metadata = torch.load(metadata_path)
    print(f"模型元数据:")
    print(f"  - num_tiles: {metadata['num_tiles']}")
    print(f"  - tile_state_dim: {metadata['tile_state_dim']}")
    print(f"  - action_size: {metadata['action_size']}")
    print(f"  - shared_agent_path: {metadata['shared_agent_path']}")

    # 创建对应的代理实例
    print(f"\n创建 CGPPOAgent 实例...")
    agent = CGPPOAgent(
        num_tiles=metadata['num_tiles'],
        tile_state_dim=metadata['tile_state_dim'],
        action_size=metadata['action_size'],
        # 其他参数使用默认值，因为我们只是为了查看结构
    )

    # 加载共享代理
    print(f"元数据中的路径: {metadata['shared_agent_path']}")

    # 直接使用指定的正确路径
    shared_agent_path = '/home/bingxing2/home/scx7axu/program/rlcg/log/Muu_20251203_223703_w3=5/final_model.pt_shared'

    if not os.path.exists(shared_agent_path):
        print(f"错误：找不到共享代理文件 {shared_agent_path}")
        print("请检查模型文件是否完整")
        return

    print(f"加载共享代理文件: {shared_agent_path}")
    agent.tile_agents[0].load(shared_agent_path)

    # 显示模型结构
    print(f"\n{'='*60}")
    print("网络结构详情:")
    print(f"{'='*60}")

    # 获取模型（所有tile代理共享相同模型）
    model = agent.tile_agents[0].model

    print(f"模型类型: {type(model)}")
    print(f"模型结构:")
    print(model)

    # 详细显示每个分支
    print(f"\n{'='*40}")
    print("策略网络 (Policy Network):")
    print(f"{'='*40}")
    policy_branch = model.child_modules[0]  # Branched 的第一个分支是策略网络
    print(policy_branch)

    print(f"\n{'='*40}")
    print("价值网络 (Value Network):")
    print(f"{'='*40}")
    value_branch = model.child_modules[1]  # Branched 的第二个分支是价值网络
    print(value_branch)

    # 显示模型参数统计
    print(f"\n{'='*40}")
    print("模型参数统计:")
    print(f"{'='*40}")

    def count_parameters(model, name=""):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}总参数数量: {total_params:,}")
        print(f"{name}可训练参数数量: {trainable_params:,}")
        return total_params, trainable_params

    # 策略网络参数
    print("策略网络:")
    policy_total, policy_trainable = count_parameters(policy_branch, "  ")

    # 价值网络参数
    print("价值网络:")
    value_total, value_trainable = count_parameters(value_branch, "  ")

    print(f"\n总计:")
    print(f"  所有参数: {policy_total + value_total:,}")
    print(f"  可训练参数: {policy_trainable + value_trainable:,}")

    # 显示具体的层信息
    print(f"\n{'='*40}")
    print("网络层详情:")
    print(f"{'='*40}")

    def print_layer_info(module, name="", indent=0):
        prefix = "  " * indent
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child_module, torch.nn.Sequential):
                print(f"{prefix}{full_name}: {type(child_module).__name__}")
                print_layer_info(child_module, full_name, indent + 1)
            elif isinstance(child_module, torch.nn.Linear):
                in_features, out_features = child_module.in_features, child_module.out_features
                print(f"{prefix}{full_name}: Linear({in_features} -> {out_features})")
            elif isinstance(child_module, torch.nn.ReLU):
                print(f"{prefix}{full_name}: ReLU")
            else:
                print(f"{prefix}{full_name}: {type(child_module).__name__}")

    print("策略网络结构:")
    print_layer_info(policy_branch)

    print("\n价值网络结构:")
    print_layer_info(value_branch)

    print(f"\n{'='*60}")
    print("模型加载和显示完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # 默认模型路径
    default_model_path = "./log/inference_time_test/final_model"

    # 如果提供了命令行参数，使用它作为模型路径
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = default_model_path

    # 检查模型路径是否存在
    if not os.path.exists(f"{model_path}.pt"):
        print(f"错误：找不到模型文件 {model_path}.pt")
        print(f"请确保模型文件存在，或提供正确的路径作为命令行参数")
        print(f"用法: python {sys.argv[0]} [model_path]")
        sys.exit(1)

    load_and_display_model(model_path)
