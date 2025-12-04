#!/usr/bin/env python3
"""
推理时间演示脚本
模拟数据并展示模型的一次推理时间
"""

import torch
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.ppo_agent_factory import CGPPOAgent


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


def measure_inference_time(model_path: str, num_runs: int = 100):
    """
    测量模型推理时间

    Args:
        model_path: 模型路径（不包含.pt后缀）
        num_runs: 运行次数，用于统计平均时间
    """
    print("=" * 60)
    print(f"推理时间测试: {model_path}")
    print("=" * 60)

    # 加载元数据
    metadata_path = f"{model_path}.pt"
    if not os.path.exists(metadata_path):
        print(f"错误：找不到元数据文件 {metadata_path}")
        return

    print(f"加载元数据文件: {metadata_path}")
    metadata = torch.load(metadata_path)
    print(f"模型配置:")
    print(f"  - num_tiles: {metadata['num_tiles']}")
    print(f"  - tile_state_dim: {metadata['tile_state_dim']}")
    print(f"  - action_size: {metadata['action_size']}")
    print(f"  - total_state_dim: {metadata['num_tiles'] * metadata['tile_state_dim']}")

    # 创建代理实例
    print(f"\n创建 CGPPOAgent 实例...")
    agent = CGPPOAgent(
        num_tiles=metadata['num_tiles'],
        tile_state_dim=metadata['tile_state_dim'],
        action_size=metadata['action_size']
    )

    # 加载模型
    shared_agent_path = '/home/bingxing2/home/scx7axu/program/rlcg/log/Muu_20251203_223703_w3=5/final_model.pt_shared'
    if not os.path.exists(shared_agent_path):
        print(f"错误：找不到共享代理文件 {shared_agent_path}")
        return

    print(f"加载共享代理文件: {shared_agent_path}")
    agent.tile_agents[0].load(shared_agent_path)

    # 设置为评估模式
    agent.eval_mode()

    # 创建模拟输入
    print("\n创建模拟输入数据...")
    test_input = create_mock_input(metadata['num_tiles'], metadata['tile_state_dim'])
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

    # 如果提供了命令行参数，使用它作为模型路径
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = default_model_path

    # 检查模型路径是否存在
    if not os.path.exists(f"{model_path}.pt"):
        print(f"错误：找不到模型文件 {model_path}.pt")
        print(f"请确保模型文件存在，或提供正确的路径作为命令行参数")
        print(f"用法: python {sys.argv[0]} [model_path] [num_runs]")
        sys.exit(1)

    # 获取运行次数参数
    num_runs = 100
    if len(sys.argv) > 2:
        try:
            num_runs = int(sys.argv[2])
        except ValueError:
            print("错误：num_runs参数必须是整数")
            sys.exit(1)

    measure_inference_time(model_path, num_runs)
