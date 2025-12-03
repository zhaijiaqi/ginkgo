"""
Data Utils - 数据处理和序列化工具函数
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Union


def convert_to_serializable(obj: Any) -> Any:
    """
    将numpy数据类型转换为JSON可序列化的Python原生类型

    Args:
        obj: 要转换的对象

    Returns:
        转换后的对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj
