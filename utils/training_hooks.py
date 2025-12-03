"""
Training Hooks - 训练过程中的钩子函数
"""

from typing import Any, Dict, List


class TrainingStatsHook:
    """训练统计钩子，用于记录详细的训练统计"""

    def __init__(self, training_logger, env, eval_interval_steps):
        self.training_logger = training_logger
        self.env = env
        self.eval_interval_steps = eval_interval_steps
        self.step_count = 0
        self.episode_count = 0
        self.current_episode_actions = []  # 当前episode的所有tile动作历史
        self.support_train_agent = True  # pfrl 需要的属性

    def __call__(self, env, agent, step):
        self.step_count += 1

        # 记录步骤奖励和详细统计信息
        if hasattr(env, 'last_reward') and hasattr(env, 'last_info'):
            self.training_logger.log_step(self.step_count, env.last_reward, env.last_info)

            # 收集tile动作历史用于episode记录
            if 'tile_actions' in env.last_info:
                self.current_episode_actions.append(env.last_info['tile_actions'])


class EvalHook:
    """评估钩子，用于记录episode信息"""

    def __init__(self, training_hook):
        self.training_hook = training_hook
        self.support_train_agent = True  # pfrl 需要的属性

    def __call__(self, env, agent, step, eval_stats, **kwargs):
        """评估钩子 - 记录episode统计"""
        print(f"评估结果 (Step {step}): {eval_stats}")

        # 记录详细的episode统计
        self.training_hook.episode_count += 1

        # 从环境中获取完整的episode信息
        episode_info = env.get_episode_info()

        # 添加评估统计
        episode_stats = dict(eval_stats)  # 复制eval_stats
        episode_stats.update(episode_info)

        # 添加tile动作历史
        detailed_info = {
            'residual_history': episode_info.get('residual_history', []),
            'tile_actions_history': self.training_hook.current_episode_actions.copy(),
            'performance_stats': episode_info.get('performance_stats', {})
        }

        # 记录episode统计
        self.training_hook.training_logger.log_episode(
            self.training_hook.episode_count,
            episode_stats,
            detailed_info
        )

        # 重置当前episode的动作历史
        self.training_hook.current_episode_actions = []
