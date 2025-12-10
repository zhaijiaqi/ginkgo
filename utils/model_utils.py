"""
Model Utils - 模型加载相关工具函数
"""

import os
import torch
from typing import Dict, Optional, Tuple

from env.cg_env import CGEnvironment
from agent.ppo_agent_factory import PPOAgentFactory


def _infer_tile_state_dim_from_weights(weights: dict) -> Optional[int]:
    """
    从权重文件中推断tile_state_dim
    
    Args:
        weights: 模型权重字典
        
    Returns:
        tile_state_dim 或 None
    """
    # child_modules.0.0.0.weight 是策略网络第一层，输入维度是 tile_state_dim
    policy_first_key = 'child_modules.0.0.0.weight'
    if policy_first_key in weights:
        tile_state_dim = weights[policy_first_key].shape[1]  # 输入维度
        return tile_state_dim
    return None


def _load_model_metadata(model_path: str) -> Optional[Dict]:
    """
    尝试加载模型元数据
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        元数据字典或None
    """
    try:
        data = torch.load(model_path, map_location='cpu', weights_only=False)
        # 检查是否包含元数据（新格式）
        if isinstance(data, dict):
            # 检查是否包含元数据键
            if 'num_tiles' in data and 'tile_state_dim' in data and 'action_size' in data:
                return data
    except Exception:
        pass
    return None


def detect_model_config(model_path: str) -> Optional[Dict]:
    """
    检测模型训练时的配置信息（不加载模型）
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        包含 tilesize, tile_state_dim, num_tiles, action_size 的字典，或None
    """
    original_path = model_path
    if not os.path.exists(model_path):
        if os.path.exists(f"{model_path}.pt"):
            model_path = f"{model_path}.pt"
        else:
            return None
    
    # 尝试加载元数据
    metadata = _load_model_metadata(model_path)
    if metadata:
        model_tile_state_dim = metadata['tile_state_dim']
        model_tilesize = model_tile_state_dim - 1
        return {
            'tilesize': model_tilesize,
            'tile_state_dim': model_tile_state_dim,
            'num_tiles': metadata.get('num_tiles'),
            'action_size': metadata.get('action_size'),
            'hidden_sizes': metadata.get('hidden_sizes', [64, 64])  # 如果存在则返回，否则使用默认值
        }
    
    # 尝试从权重文件推断
    try:
        weights = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(weights, dict):
            has_model_keys = any('child_modules' in str(k) or 'weight' in str(k) for k in weights.keys())
            if has_model_keys:
                model_tile_state_dim = _infer_tile_state_dim_from_weights(weights)
                if model_tile_state_dim:
                    model_tilesize = model_tile_state_dim - 1
                    return {
                        'tilesize': model_tilesize,
                        'tile_state_dim': model_tile_state_dim,
                        'num_tiles': None,
                        'action_size': None
                    }
    except Exception:
        pass
    
    return None


def load_model_weights(model_path: str, config: Dict, env: CGEnvironment):
    """
    加载模型权重文件并创建代理
    会自动从模型文件中推断训练时的配置（tilesize等），确保模型和环境配置匹配

    Args:
        model_path: 模型权重文件路径（可以是.pt文件或目录，会自动处理）
        config: 训练配置（会被更新以匹配模型）
        env: CG环境（如果模型配置不匹配，会提示用户）

    Returns:
        CGPPOAgent 实例
    """
    # 检查模型文件是否存在
    original_path = model_path
    if not os.path.exists(model_path):
        # 尝试添加.pt后缀
        if os.path.exists(f"{model_path}.pt"):
            model_path = f"{model_path}.pt"
        else:
            raise FileNotFoundError(f"找不到模型权重文件: {original_path} 或 {original_path}.pt")

    print(f"加载模型权重文件: {model_path}")
    
    # 首先尝试加载元数据（新格式）
    metadata = _load_model_metadata(model_path)
    
    if metadata:
        # 新格式：从元数据中读取配置
        print("检测到新格式模型文件（包含元数据）")
        model_tile_state_dim = metadata['tile_state_dim']
        model_num_tiles = metadata['num_tiles']
        model_action_size = metadata['action_size']
        
        # 计算模型训练时的tilesize
        model_tilesize = model_tile_state_dim - 1  # tile_state_dim = tilesize + 1
        
        # 获取hidden_sizes（如果存在）
        model_hidden_sizes = metadata.get('hidden_sizes', [64, 64])
        if not isinstance(model_hidden_sizes, list):
            model_hidden_sizes = [64, 64]  # 默认值
        
        print(f"模型训练配置:")
        print(f"  - tile_state_dim: {model_tile_state_dim}")
        print(f"  - tilesize: {model_tilesize}")
        print(f"  - num_tiles: {model_num_tiles}")
        print(f"  - action_size: {model_action_size}")
        print(f"  - hidden_sizes: {model_hidden_sizes}")
        
        # 检查当前环境配置是否匹配
        current_tilesize = env.spmv_sim.tilesize
        current_tile_state_dim = current_tilesize + 1
        
        if current_tile_state_dim != model_tile_state_dim:
            print(f"\n⚠️  警告: 环境配置与模型不匹配!")
            print(f"  模型训练时的 tilesize: {model_tilesize} (tile_state_dim={model_tile_state_dim})")
            print(f"  当前环境的 tilesize: {current_tilesize} (tile_state_dim={current_tile_state_dim})")
            print(f"\n正在更新配置以匹配模型...")
            
            # 更新配置以匹配模型
            if 'spmv' not in config:
                config['spmv'] = {}
            config['spmv']['tilesize'] = model_tilesize
            
            # 重新创建环境以匹配模型配置
            from .env_utils import create_env_config
            env_config = create_env_config(config)
            env = CGEnvironment(env_config)
            print(f"已更新环境配置: tilesize={model_tilesize}")
        
        # 使用模型元数据创建代理
        from agent.ppo_agent_factory import CGPPOAgent
        cg_agent = CGPPOAgent(
            num_tiles=model_num_tiles,
            tile_state_dim=model_tile_state_dim,
            action_size=model_action_size,
            hidden_sizes=model_hidden_sizes  # 使用模型保存的hidden_sizes
        )
        
        # 加载共享模型权重
        if 'shared_agent_path' in metadata:
            shared_path = metadata['shared_agent_path']
            if os.path.exists(shared_path):
                cg_agent.tile_agents[0].load(shared_path)
                print("从共享模型文件加载权重成功")
            else:
                raise FileNotFoundError(f"找不到共享模型文件: {shared_path}")
        else:
            raise ValueError("元数据中缺少 shared_agent_path")
            
    else:
        # 旧格式：直接加载state_dict，从权重形状推断配置
        print("检测到旧格式模型文件（直接state_dict）")
        weights = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if not isinstance(weights, dict):
            raise ValueError(f"权重文件不是字典格式: {model_path}")
        
        # 检查是否包含模型权重键
        has_model_keys = any('child_modules' in str(k) or 'weight' in str(k) for k in weights.keys())
        if not has_model_keys:
            raise ValueError(f"无法识别的权重文件格式: {model_path}")
        
        # 从权重形状推断tile_state_dim
        model_tile_state_dim = _infer_tile_state_dim_from_weights(weights)
        if model_tile_state_dim is None:
            raise ValueError("无法从权重文件中推断tile_state_dim")
        
        # 计算模型训练时的tilesize
        model_tilesize = model_tile_state_dim - 1
        
        print(f"从权重推断的模型配置:")
        print(f"  - tile_state_dim: {model_tile_state_dim}")
        print(f"  - tilesize: {model_tilesize}")
        
        # 检查当前环境配置是否匹配
        current_tilesize = env.spmv_sim.tilesize
        current_tile_state_dim = current_tilesize + 1
        
        if current_tile_state_dim != model_tile_state_dim:
            print(f"\n⚠️  警告: 环境配置与模型不匹配!")
            print(f"  模型训练时的 tilesize: {model_tilesize} (tile_state_dim={model_tile_state_dim})")
            print(f"  当前环境的 tilesize: {current_tilesize} (tile_state_dim={current_tile_state_dim})")
            print(f"\n正在更新配置以匹配模型...")
            
            # 更新配置以匹配模型
            if 'spmv' not in config:
                config['spmv'] = {}
            config['spmv']['tilesize'] = model_tilesize
            
            # 重新创建环境以匹配模型配置
            from .env_utils import create_env_config
            env_config = create_env_config(config)
            env = CGEnvironment(env_config)
            print(f"已更新环境配置: tilesize={model_tilesize}")
        
        # 创建代理（使用更新后的环境）
        state_dim = env.get_state_dim()
        action_size = env.get_action_space_size()
        cg_agent = PPOAgentFactory(config).create_agent(state_dim, action_size)
        
        # 直接加载权重到模型
        model = cg_agent.tile_agents[0].model
        model.load_state_dict(weights)
        print("直接加载权重成功")
    
    # 设置为评估模式
    cg_agent.eval_mode()
    for tile_agent in cg_agent.tile_agents:
        if hasattr(tile_agent, 'eval'):
            tile_agent.eval()
        if hasattr(tile_agent, 'training'):
            tile_agent.training = False

    print("模型权重加载完成")
    return cg_agent

