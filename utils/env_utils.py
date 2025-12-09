"""
Environment Utils - 环境配置相关工具函数
"""

from typing import Dict, Optional


def create_env_config(config: Dict, matrix_name: Optional[str] = None, 
                     matrix_size: Optional[int] = None) -> Dict:
    """
    从训练配置创建环境配置

    Args:
        config: 训练配置字典
        matrix_name: 矩阵名称（如果指定，会覆盖配置中的值）
        matrix_size: 矩阵大小（如果指定且matrix_name为None，会覆盖配置中的值）

    Returns:
        环境配置字典
    """
    # 确定matrix_name
    final_matrix_name = matrix_name if matrix_name is not None else config.get('cg', {}).get('matrix_name', 'None')
    
    # 确定matrix_size
    if matrix_size is not None and final_matrix_name == 'None':
        final_matrix_size = matrix_size
    elif matrix_size is not None:
        final_matrix_size = matrix_size
    else:
        final_matrix_size = config.get('cg', {}).get('matrix_size', 1024)
    
    env_config = {
        'max_iter': config.get('cg', {}).get('max_iter', 100),
        'stop_tol': config.get('cg', {}).get('stop_tol', 1e-10),
        'matrix_size': final_matrix_size,
        'matrix_name': final_matrix_name,
        'matrix_data_dir': config.get('cg', {}).get('matrix_data_dir', '~/data/matrix'),
        'matrix_set_csv': config.get('cg', {}).get('matrix_set_csv', 'matrix_set.csv'),
        'tilesize': config.get('spmv', {}).get('tilesize', 32),
        'precision_cost_table': config.get('spmv', {}).get('precision_cost_table', {
            'fp64': 1.0, 'fp32': 0.7, 'tf32': 0.55,
            'fp16': 0.35, 'bf16': 0.33, 'fp8': 0.15
        }),
        'reward': config.get('reward'),
        'normalize_state': config.get('env', {}).get('normalize_state', True),
        'random_seed': config.get('random_seed', 42)
    }
    return env_config

