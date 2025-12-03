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
        self.support_train_agent = True  # pfrl 需要的属性

    def __call__(self, env, agent, step):
        self.step_count += 1

        # 记录步骤奖励（需要从环境中获取）
        if hasattr(env, 'last_reward'):
            self.training_logger.log_step(self.step_count, env.last_reward)


class EvalHook:
    """评估钩子，用于记录episode信息"""

    def __init__(self, training_hook):
        self.training_hook = training_hook
        self.support_train_agent = True  # pfrl 需要的属性

    def __call__(self, env, agent, step, eval_stats, **kwargs):
        """评估钩子 - 记录episode统计"""
        print(f"评估结果 (Step {step}): {eval_stats}")
