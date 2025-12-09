"""
Plot Utils - 绘图相关工具函数
"""

import matplotlib.pyplot as plt


def configure_matplotlib_chinese():
    """
    配置matplotlib支持中文显示
    应在导入matplotlib后、绘图前调用
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

