#!/usr/bin/env python3
"""
推理时间测试和模型结构显示脚本
支持：
1. 显示模型结构详情
2. 测量模型推理时间
"""

import torch
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.ppo_agent_factory import CGPPOAgent
import pfrl


def create_mock_input(num_tiles: int, tile_state_dim: int):
    """
    创建模拟输入数据

    Args:
        num_tiles: tile数量
        tile_state_dim: 每个tile的状态维度

    Returns:
        模拟的状态向量
    """
    # 每个tile的状态包括：tile_size + 1（迭代索引）
    # 这里我们创建一个合理的模拟数据

    mock_state = []

    for tile_idx in range(num_tiles):
        # 模拟tile的状态特征（例如：残差、当前精度等）
        tile_features = np.random.rand(tile_state_dim - 1) * 2 - 1  # 归一化到[-1, 1]

        # 迭代索引（假设在CG算法的不同迭代阶段）
        iteration_idx = np.random.randint(0, 100)  # 迭代步数

        # 组合tile状态
        tile_state = np.concatenate([tile_features, [iteration_idx]])
        mock_state.extend(tile_state)

    return np.array(mock_state, dtype=np.float32)


def load_model_weights(model_path: str):
    """
    加载模型权重文件并推断参数
    
    Args:
        model_path: 模型权重文件路径（不包含.pt后缀）
    
    Returns:
        tuple: (weights, num_tiles, tile_state_dim, action_size, agent)
    """
    # 加载模型权重文件
    weight_path = f"{model_path}"
    if not os.path.exists(weight_path):
        print(f"错误：找不到模型权重文件 {weight_path}")
        return None, None, None, None, None

    print(f"加载模型权重文件: {weight_path}")
    weights = torch.load(weight_path, weights_only=False)
    
    # 检查权重文件格式
    if not isinstance(weights, dict) or not any('child_modules' in k for k in weights.keys()):
        print("错误：这不是有效的模型权重文件格式")
        return None, None, None, None, None
    
    # 从权重形状推断参数
    print("\n从权重形状推断模型参数...")
    
    # child_modules.0.0.0.weight 是策略网络第一层，输入维度是 tile_state_dim
    # child_modules.0.0.4.weight 是策略网络最后一层，输出维度是 action_size
    policy_first_key = 'child_modules.0.0.0.weight'
    policy_last_key = 'child_modules.0.0.4.weight'
    
    tile_state_dim = None
    action_size = None
    
    if policy_first_key in weights:
        tile_state_dim = weights[policy_first_key].shape[1]  # 输入维度
        print(f"从权重推断: tile_state_dim = {tile_state_dim}")
    else:
        print("错误：无法从权重文件中找到策略网络第一层")
        return None, None, None, None, None
    
    if policy_last_key in weights:
        action_size = weights[policy_last_key].shape[0]  # 输出维度
        print(f"从权重推断: action_size = {action_size}")
    else:
        print("错误：无法从权重文件中找到策略网络最后一层")
        return None, None, None, None, None
    
    # 从配置文件读取 num_tiles
    num_tiles = None
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'default.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            cg_config = config.get('cg', {})
            spmv_config = config.get('spmv', {})
            
            matrix_size = cg_config.get('matrix_size', 512)
            tilesize = spmv_config.get('tilesize', 32)
            num_tiles = (matrix_size + tilesize - 1) // tilesize  # 向上取整
            print(f"从配置文件读取: num_tiles = {num_tiles}")
    except Exception as e:
        print(f"从配置文件读取失败: {e}")
    
    # 如果无法从配置文件读取，使用默认值
    if num_tiles is None:
        print("警告：无法从配置文件确定 num_tiles，使用默认值 16")
        num_tiles = 100
    
    print(f"\n模型参数:")
    print(f"  - num_tiles: {num_tiles}")
    print(f"  - tile_state_dim: {tile_state_dim}")
    print(f"  - action_size: {action_size}")
    print(f"  - total_state_dim: {num_tiles * tile_state_dim}")

    # 创建代理实例
    print(f"\n创建 CGPPOAgent 实例...")
    agent = CGPPOAgent(
        num_tiles=num_tiles,
        tile_state_dim=tile_state_dim,
        action_size=action_size
    )

    # 直接加载权重到模型
    print(f"加载权重到模型...")
    model = agent.tile_agents[0].model
    model.load_state_dict(weights)
    model = torch.compile(model)
    print("权重加载完成")
    
    return weights, num_tiles, tile_state_dim, action_size, agent


def display_model_structure(agent):
    """
    显示模型结构详情
    
    Args:
        agent: CGPPOAgent 实例
    """
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
    print("模型结构显示完成!")
    print(f"{'='*60}")


def measure_inference_time(model_path: str, num_runs: int = 100, show_structure: bool = False):
    """
    测量模型推理时间

    Args:
        model_path: 模型权重文件路径（不包含.pt后缀）
        num_runs: 运行次数，用于统计平均时间
        show_structure: 是否显示模型结构
    """
    print("=" * 60)
    print(f"推理时间测试: {model_path}")
    print("=" * 60)

    # 加载模型
    weights, num_tiles, tile_state_dim, action_size, agent = load_model_weights(model_path)
    if agent is None:
        return None

    # 如果设置了显示结构，先显示模型结构
    if show_structure:
        display_model_structure(agent)
    
    # 设置为评估模式
    agent.eval_mode()

    # 创建模拟输入
    print("\n创建模拟输入数据...")
    test_input = create_mock_input(num_tiles, tile_state_dim)
    print(f"输入形状: {test_input.shape}")
    print(f"输入范围: [{test_input.min():.3f}, {test_input.max():.3f}]")

    # 预热运行
    print("\n预热运行...")
    for _ in range(5):
        _ = agent.act(test_input)

    # 正式测量推理时间
    print("\n开始测量推理时间...")
    print(f"运行 {num_runs} 次推理测试...")

    inference_times = []

    for i in range(num_runs):
        start_time = time.perf_counter()
        actions = agent.act(test_input)
        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000  # 转换为毫秒
        inference_times.append(inference_time)

    # 统计结果
    inference_times = np.array(inference_times)
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    median_time = np.median(inference_times)

    print("\n" + "="*60)
    print("推理时间统计结果 (毫秒):")
    print("="*60)
    print(f"平均推理时间: {mean_time:.3f} ms")
    print(f"标准差: {std_time:.3f} ms")
    print(f"最小时间: {min_time:.3f} ms")
    print(f"最大时间: {max_time:.3f} ms")
    print(f"中位数时间: {median_time:.3f} ms")
    print(f"95%置信区间: [{np.percentile(inference_times, 2.5):.3f}, {np.percentile(inference_times, 97.5):.3f}] ms")

    # 显示一次推理的详细结果
    print("\n" + "="*40)
    print("单次推理结果示例:")
    print("="*40)

    # 再次运行一次以显示详细结果
    actions = agent.act(test_input)

    print(f"输入状态维度: {len(test_input)}")
    print(f"输出动作数量: {len(actions)}")
    print(f"每个tile的动作: {actions[:10]}..." if len(actions) > 10 else f"每个tile的动作: {actions}")

    # 显示动作分布统计
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1

    print(f"动作分布: {action_counts}")
    print(f"最常用动作: {max(action_counts.items(), key=lambda x: x[1])}")

    print("\n" + "="*60)
    print("推理时间测试完成!")
    print("="*60)

    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'median_time': median_time,
        'inference_times': inference_times
    }


if __name__ == "__main__":
    # 默认模型路径
    default_model_path = "./log/inference_time_test/final_model"

    # 解析命令行参数
    model_path = default_model_path
    num_runs = 100
    show_structure = False
    structure_only = False

    if len(sys.argv) > 1:
        # 检查是否是帮助或特殊选项
        if sys.argv[1] in ['-h', '--help']:
            print("用法: python inference_time_demo.py [model_path] [num_runs] [--structure] [--structure-only]")
            print("\n参数:")
            print("  model_path      模型权重文件路径（不包含.pt后缀）")
            print("  num_runs        推理测试运行次数（默认: 100）")
            print("  --structure     显示模型结构（同时进行推理测试）")
            print("  --structure-only 仅显示模型结构（不进行推理测试）")
            sys.exit(0)
        
        # 解析选项
        args = sys.argv[1:]
        if '--structure-only' in args:
            structure_only = True
            args.remove('--structure-only')
        if '--structure' in args:
            show_structure = True
            args.remove('--structure')
        
        # 解析位置参数
        if len(args) > 0:
            model_path = args[0]
        if len(args) > 1:
            try:
                num_runs = int(args[1])
            except ValueError:
                print("错误：num_runs参数必须是整数")
                sys.exit(1)

    # 检查模型权重文件是否存在
    if not os.path.exists(f"{model_path}"):
        print(f"错误：找不到模型权重文件 {model_path}")
        print(f"请确保模型权重文件存在，或提供正确的路径作为命令行参数")
        print(f"用法: python {sys.argv[0]} [model_path] [num_runs] [--structure] [--structure-only]")
        sys.exit(1)

    # 如果只显示结构
    if structure_only:
        print("=" * 60)
        print(f"显示模型结构: {model_path}")
        print("=" * 60)
        weights, num_tiles, tile_state_dim, action_size, agent = load_model_weights(model_path)
        if agent is not None:
            display_model_structure(agent)
    else:
        # 进行推理时间测试（可选择是否显示结构）
        measure_inference_time(model_path, num_runs, show_structure=show_structure)
