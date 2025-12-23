"""
PPO Agent Factory - 使用 pfrl 实现 PPO 代理
提供策略网络和价值网络，用于 CG 混合精度控制
"""

import torch
import torch.nn as nn
import pfrl
from pfrl.agents import PPO
import numpy as np
from typing import Dict, Any, Optional, List
import os
import logging
import time
from collections import deque

class CGPPOAgent:
    """
    为CG环境定制的PPO代理
    使用多个独立的PPO子代理，每个tile一个
    """

    def __init__(self, num_tiles: int, tile_state_dim: int, action_size: int, hidden_sizes=None, max_grad_norm=None, **ppo_kwargs):
        self.num_tiles = num_tiles
        self.tile_state_dim = tile_state_dim
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [64, 64]  # 默认隐藏层大小
        # 统计窗口：我们做 batch 推理时不走 pfrl.PPO.act()，因此需要自己维护 entropy/value 统计
        # 用 deque 做滑动窗口，避免 get_statistics() 里出现空均值导致的 NaN
        self._stats_window = int(ppo_kwargs.get("stats_window", 100))
        self._entropy_window = deque(maxlen=self._stats_window)
        self._value_window = deque(maxlen=self._stats_window)

        # 创建共享的模型（单个tile的状态维度）
        shared_model = self._create_single_tile_model(tile_state_dim, action_size, self.hidden_sizes)

        # 创建多个代理实例，但共享相同的模型参数
        self.tile_agents = []
        for _ in range(num_tiles):
            # 创建PPO代理的参数
            agent_kwargs = ppo_kwargs.copy()
            agent_kwargs['model'] = shared_model  # 所有代理共享同一个模型
            agent_kwargs['optimizer'] = torch.optim.Adam(shared_model.parameters(), lr=ppo_kwargs.get('lr', 1e-4))
            agent_kwargs.pop('lr', None)

            # 添加梯度裁剪参数
            if max_grad_norm is not None:
                agent_kwargs['max_grad_norm'] = max_grad_norm

            # 创建代理实例，使用带有梯度裁剪的 PPO
            agent = PPO(**agent_kwargs)
            self.tile_agents.append(agent)

    def _create_single_tile_model(self, tile_state_dim: int, action_size: int, hidden_sizes: list):
        """创建单个tile的模型
        
        Args:
            tile_state_dim: tile状态维度
            action_size: 动作空间大小
            hidden_sizes: 隐藏层大小列表，例如 [64, 64] 表示两层，每层64个神经元
        """
        def make_policy_network():
            layers = []
            prev_size = tile_state_dim

            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                ])
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, action_size))
            policy_net = nn.Sequential(*layers)

            # 数值稳定性改进：最后一层使用合理的初始化
            if layers and isinstance(layers[-1], nn.Linear):
                with torch.no_grad():
                    # 最后一层使用标准初始化，避免过小
                    nn.init.orthogonal_(layers[-1].weight, gain=0.1)  # 增大增益
                    layers[-1].bias.data.fill_(1.0)
                    # 关键：让初始策略倾向 action0（假设 action0=高精度）
                    # 这里直接硬编码一个正偏置（logit 加成）；值越大越偏向 action0。
                    if action_size > 0:
                        layers[-1].bias.data[0] = 10.0

            # 初始化权重 - 为策略网络使用更合适的增益
            for layer in policy_net:
                if isinstance(layer, nn.Linear):
                    if layer == layers[-1]:  # 最后一层已经初始化过了
                        continue
                    # 隐藏层使用标准初始化
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

            return policy_net

        def make_value_network():
            layers = []
            prev_size = tile_state_dim

            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                ])
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, 1))
            value_net = nn.Sequential(*layers)

            # 初始化权重
            for layer in value_net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

            return value_net

        policy = make_policy_network()
        value = make_value_network()

        # 使用 SoftmaxCategoricalHead 将策略输出转换为分类分布
        policy_with_dist = nn.Sequential(policy, pfrl.policies.SoftmaxCategoricalHead())

        model = pfrl.nn.Branched(policy_with_dist, value)
        return model

    def act(self, obs):
        """为每个tile独立选择动作（参数共享的代理）- 优化批量推理"""
        # 直接将连续的obs数组重塑为批量形式，避免切分/合并开销
        obs_array = np.asarray(obs, dtype=np.float32)
        batch_obs = obs_array.reshape(self.num_tiles, self.tile_state_dim)
        batch_obs = torch.from_numpy(batch_obs)

        # 使用共享模型进行推理（所有代理共享相同模型）
        model = self.tile_agents[0].model
        device = next(model.parameters()).device
        batch_obs = batch_obs.to(device)

        # 前向传播获取策略分布与 value（用于统计）
        with torch.no_grad():
            policy_dist, values = model(batch_obs)

            # 数值稳定性：如果分布参数出现 NaN/Inf，fallback 到均匀分布
            # 注意：policy_dist 通常已经是 torch.distributions.Distribution（来自 SoftmaxCategoricalHead）
            if hasattr(policy_dist, "logits"):
                logits = policy_dist.logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(
                        f"警告: 检测到无效的logits值 - NaN: {torch.isnan(logits).any()}, Inf: {torch.isinf(logits).any()}"
                    )
                    print(f"logits范围: min={logits.min().item():.6f}, max={logits.max().item():.6f}")
                    policy_dist = torch.distributions.Categorical(logits=torch.zeros_like(logits))
            elif hasattr(policy_dist, "probs"):
                probs = policy_dist.probs
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    print(
                        f"警告: 检测到无效的probs值 - NaN: {torch.isnan(probs).any()}, Inf: {torch.isinf(probs).any()}"
                    )
                    # 用均匀 probs 作为 fallback
                    uniform = torch.ones_like(probs) / probs.size(-1)
                    policy_dist = torch.distributions.Categorical(probs=uniform)

            # 从分布采样动作
            actions_t = policy_dist.sample()

            # 维护统计：batch 平均 entropy / value（不依赖 pfrl PPO 内部统计）
            try:
                ent = policy_dist.entropy()
                self._entropy_window.append(float(ent.mean().detach().cpu().item()))
            except Exception:
                # 某些分布可能不支持 entropy（理论上不该发生），保持窗口不更新
                pass

            try:
                # values 形状通常为 (B, 1) 或 (B,)
                v = values
                if v is not None:
                    self._value_window.append(float(v.mean().detach().cpu().item()))
            except Exception:
                pass

            actions = actions_t.detach().cpu().numpy()

        # 为每个代理设置状态（模拟act方法的行为）
        for tile_idx in range(self.num_tiles):
            agent = self.tile_agents[tile_idx]

            # 初始化batch变量（如果还没有初始化）
            if agent.batch_last_episode is None:
                agent._initialize_batch_variables(1)

            # 设置上一次的状态和动作（模拟pfrl act方法的行为）
            # 从原始obs中提取对应tile的观测
            start_idx = tile_idx * self.tile_state_dim
            end_idx = start_idx + self.tile_state_dim
            tile_obs = obs[start_idx:end_idx]
            agent.batch_last_state = [tile_obs]
            agent.batch_last_action = [actions[tile_idx]]

        return actions.tolist()  # 转换为列表以保持接口一致性

    def get_statistics(self):
        """返回自定义统计信息（用于 batch act 的情况）"""
        avg_value = float(np.mean(self._value_window)) if len(self._value_window) > 0 else np.nan
        avg_entropy = float(np.mean(self._entropy_window)) if len(self._entropy_window) > 0 else np.nan
        # 保持 pfrl 的接口风格：list[tuple[str, scalar]]
        return [
            ("average_value", np.float32(avg_value)),
            ("average_entropy", np.float32(avg_entropy)),
        ]

    def _get_custom_stats_dict(self) -> Dict[str, np.float32]:
        """以 dict 形式返回 batch 统计，便于与 PPO 原生统计合并。"""
        stats = self.get_statistics()
        out: Dict[str, np.float32] = {}
        for k, v in stats:
            out[k] = v
        return out
    
    # def act(self, obs):
    #     """为每个tile独立选择动作（参数共享的代理）- 优化推理"""
    #     actions = []
    #     for tile_idx in range(self.num_tiles):
    #         start_idx = tile_idx * self.tile_state_dim
    #         end_idx = start_idx + self.tile_state_dim
    #         tile_obs = obs[start_idx:end_idx]
    #         action = self.tile_agents[tile_idx].act(tile_obs)
    #         actions.append(action)
    #     return actions

    def observe(self, obs, reward, done, reset):
        """观察多tile环境的转换（参数共享的代理）"""
        # 为每个代理调用observe，每个代理观察对应tile的状态
        for tile_idx in range(self.num_tiles):
            # 提取当前tile的状态
            start_idx = tile_idx * self.tile_state_dim
            end_idx = start_idx + self.tile_state_dim
            tile_obs = obs[start_idx:end_idx] if len(obs) > 0 else obs

            # 为对应代理调用observe
            self.tile_agents[tile_idx].observe(tile_obs, reward, done, reset)

    def save(self, path: str):
        """保存参数共享的代理模型"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 保存参数共享的代理
        saved_data = {
            'num_tiles': self.num_tiles,
            'tile_state_dim': self.tile_state_dim,
            'action_size': self.action_size,
            'hidden_sizes': self.hidden_sizes,  # 保存隐藏层配置
        }

        # 保存第一个代理（所有代理共享相同模型，所以只需要保存一个）
        self.tile_agents[0].save(path + "_shared")
        saved_data['shared_agent_path'] = path + "_shared"

        # 保存元数据
        import torch
        torch.save(saved_data, path)

    def load(self, path: str):
        """加载参数共享的代理模型"""
        import torch
        saved_data = torch.load(path)

        # 验证参数匹配
        assert saved_data['num_tiles'] == self.num_tiles
        assert saved_data['tile_state_dim'] == self.tile_state_dim
        assert saved_data['action_size'] == self.action_size
        
        # 如果保存的数据中包含hidden_sizes，验证是否匹配
        if 'hidden_sizes' in saved_data:
            if saved_data['hidden_sizes'] != self.hidden_sizes:
                print(f"⚠️  警告: 保存的模型hidden_sizes ({saved_data['hidden_sizes']}) 与当前配置 ({self.hidden_sizes}) 不匹配")
                print(f"   使用保存的模型配置: {saved_data['hidden_sizes']}")
                # 注意：这里不更新self.hidden_sizes，因为模型结构已经创建好了
                # 如果结构不匹配，会在加载权重时失败

        # 加载共享模型（加载到一个代理，然后所有代理都会共享相同的参数）
        agent_path = saved_data['shared_agent_path']
        self.tile_agents[0].load(agent_path)

    def eval_mode(self):
        """切换到评估模式"""
        for tile_agent in self.tile_agents:
            # print(f"Before mode switch: tile_agent.training: {tile_agent.training}")
            tile_agent.eval_mode()
            # print(f"After mode switch: tile_agent.training: {tile_agent.training}")

    @property
    def training(self):
        """是否处于训练模式"""
        # 返回第一个代理的训练状态（所有代理状态应该一致）
        return self.tile_agents[0].training if self.tile_agents else False




class PPOAgentFactory:
    """
    PPO Agent 工厂类
    创建和配置用于 CG 混合精度控制的 PPO 代理
    """

    def __init__(self, yaml_config: Dict):
        """
        初始化工厂

        Args:
            config: PPO 配置参数
        """
        # self.config = yaml_config.get('ppo', {})
        self.config = yaml_config
        self.ppo_config = self.config.get('ppo', {})

    def create_agent(self, state_dim: int, action_size: int) -> CGPPOAgent:
        """
        创建为CG环境定制的PPO代理

        Args:
            state_dim: 状态维度
            action_size: 动作空间大小

        Returns:
            配置好的 CGPPOAgent 代理
        """
        # 计算tile数量
        # 假设每个 tile 的状态维度 = tilesize（与 CGEnvironment.get_state_features 对齐）
        tilesize = self.config.get('spmv', {}).get('tilesize', 32)  # 从配置中读取tilesize
        tile_state_dim = tilesize
        num_tiles = state_dim // tile_state_dim
        if num_tiles <= 0:
            raise ValueError(
                f"无法创建 CGPPOAgent：计算得到 num_tiles={num_tiles} (state_dim={state_dim}, tilesize={tilesize}). "
                f"请检查环境的观测维度是否为 num_tiles*tilesize，以及配置中的 spmv.tilesize 是否与环境一致。"
            )

        # 网络参数（传递给CGPPOAgent的子代理）

        # PPO 参数（传递给子代理）
        
        # print("self.update_interval: ", self.ppo_config.get('update_interval', 2048))

        # 创建 CG PPO 代理
        # 参数解释:
        # num_tiles: tile 的数量，每个 tile 拥有独立的子 PPO 代理
        # tile_state_dim: 每个 tile 的状态维度（通常 = tilesize）
        # action_size: 动作空间大小，表示支持多少种混合精度动作
        # lr: 学习率 (learning rate)，用于优化代理神经网络
        # gpu: 使用的 GPU 设备编号（int），为 None 时使用 CPU
        # gamma: 折扣因子 (discount factor)，控制未来奖励的当前价值
        # lambd: GAE (Generalized Advantage Estimation) 的 lambda 值，调节 bias-variance
        # phi: 状态预处理函数，通常用于将状态转换为 np.float32 类型
        # value_func_coef: value function（值函数）损失的权重
        # entropy_coef: 熵奖励的权重，鼓励策略探索
        # update_interval: 模型参数更新的步数间隔（收集多少步经验再更新一次网络）
        # minibatch_size: 每个更新周期中用于梯度下降的小批量样本数
        # epochs: 每个 update_interval 下，所有数据被训练的轮数
        # clip_eps: PPO 策略损失裁剪参数 ε，限制策略更新幅度
        # clip_eps_vf: PPO value function 损失的裁剪参数（一般为None）
        # standardize_advantages: 是否对优势函数（Advantage）标准化
        # act_deterministically: 选择动作时是否用确定性策略（测试时常用）

        # 从配置中读取隐藏层大小
        hidden_sizes = self.ppo_config.get('hidden_sizes', [64, 64])
        if isinstance(hidden_sizes, (list, tuple)):
            hidden_sizes = list(hidden_sizes)
        else:
            # 如果配置不是列表，转换为列表
            hidden_sizes = [hidden_sizes] if isinstance(hidden_sizes, int) else [64, 64]
        
        agent = CGPPOAgent(
            num_tiles=num_tiles,
            tile_state_dim=tile_state_dim,
            action_size=action_size,
            hidden_sizes=hidden_sizes,                                         # 隐藏层大小
            lr=float(self.ppo_config.get('learning_rate', 1e-4)),              # 学习率
            gpu=self.ppo_config.get('gpu', 0),                                 # GPU设备编号
            gamma=self.ppo_config.get('gamma', 0.99),                          # 折扣因子
            lambd=self.ppo_config.get('lambda', 0.95),                         # GAE lambda
            phi=lambda x: np.asarray(x, dtype=np.float32),                     # 状态预处理
            value_func_coef=self.ppo_config.get('value_coef', 0.5),            # 值函数损失系数
            entropy_coef=self.ppo_config.get('entropy_coef', 0.01),            # 熵奖励系数
            update_interval=self.ppo_config.get('update_interval', 1),         # 参数更新间隔
            # 兼容旧配置键：有些配置文件使用 batch_size 表示 minibatch_size
            minibatch_size=self.ppo_config.get('minibatch_size', self.ppo_config.get('batch_size', 64)),  # 小批量样本数
            epochs=self.ppo_config.get('n_epochs', 10),                        # 每 update 的 epoch 数
            clip_eps=self.ppo_config.get('clip_eps', 0.2),                     # 策略裁剪参数
            clip_eps_vf=self.ppo_config.get('clip_eps_vf', None),              # 值函数裁剪参数
            standardize_advantages=self.ppo_config.get('standardize_advantages', True),  # 优势归一化
            act_deterministically=self.ppo_config.get('act_deterministically', False),   # 行为是否确定性
            max_grad_norm=self.ppo_config.get('max_grad_norm', 0.5),           # 梯度裁剪
        )
        return agent


class PfrlCompatibleCGPPOAgent:
    """
    将 CGPPOAgent 适配为 pfrl 兼容的 agent 接口

    这个适配器使得多 tile 的 CGPPOAgent 能够与 pfrl.experiments.train_agent_with_evaluation 一起使用
    """

    def __init__(self, cg_ppo_agent: CGPPOAgent):
        self.cg_ppo_agent = cg_ppo_agent
        self.logger = logging.getLogger(__name__)

    def _unwrap_cg_agent(self):
        """
        训练脚本里经常会传入 DoublePrecisionWrapperAgent，它的 .ppo_agent 才是 CGPPOAgent。
        为了拿到 batch act 维护的 entropy/value 统计，这里统一做一次 unwrap。
        """
        inner = getattr(self.cg_ppo_agent, "ppo_agent", None)
        return inner if inner is not None else self.cg_ppo_agent

    def act(self, obs):
        """选择动作"""
        return self.cg_ppo_agent.act(obs)

    def batch_act(self, batch_obs):
        """批量选择动作"""
        actions = []
        for obs in batch_obs:
            action = self.act(obs)
            actions.append(action)
        return actions

    def observe(self, obs, reward, done, reset):
        """观察环境转换"""
        return self.cg_ppo_agent.observe(obs, reward, done, reset)

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        """批量观察环境转换"""
        for obs, reward, done, reset in zip(batch_obs, batch_reward, batch_done, batch_reset):
            self.observe(obs, reward, done, reset)

    def save(self, filename):
        """保存模型"""
        # CGPPOAgent 的 save 方法
        return self.cg_ppo_agent.save(filename)

    def load(self, filename):
        """加载模型"""
        # CGPPOAgent 的 load 方法
        return self.cg_ppo_agent.load(filename)

    def get_statistics(self):
        """获取统计信息"""
        # 目标：日志里既要有 PPO 原生的 loss/n_updates/explained_variance，
        # 也要保证 average_value/average_entropy 在 batch act 场景下不为 NaN。
        base_agent = self._unwrap_cg_agent()
        custom = {}
        try:
            custom = base_agent._get_custom_stats_dict()
        except Exception:
            custom = {}

        try:
            # PPO 原生统计来自底层 tile_agents（无论是否 wrapper，都应能取到）
            tile_agents = getattr(base_agent, "tile_agents", None)
            if not tile_agents:
                tile_agents = getattr(self.cg_ppo_agent, "tile_agents", None)
            base_stats = tile_agents[0].get_statistics()

            # pfrl PPO 通常返回 list[tuple[str, scalar]]；我们按原顺序输出，
            # 但若遇到 average_value/average_entropy 就用 batch 统计覆盖。
            if isinstance(base_stats, list):
                merged_list = []
                for k, v in base_stats:
                    if k in custom:
                        merged_list.append((k, custom[k]))
                    else:
                        merged_list.append((k, v))

                # 如果 base 里缺少这两项（极少见），补到最前面以匹配你期望的日志格式
                base_keys = {k for k, _ in merged_list}
                for wanted in ("average_value", "average_entropy"):
                    if wanted in custom and wanted not in base_keys:
                        merged_list.insert(0 if wanted == "average_value" else 1, (wanted, custom[wanted]))

                merged_stats = {}
                for j, stat in enumerate(merged_list):
                    merged_stats[f"shared_stat_{j}"] = stat
                return merged_stats

            # 如果 PPO 返回 dict（不常见），直接覆盖并返回
            if isinstance(base_stats, dict):
                merged = dict(base_stats)
                merged.update(custom)
                return merged

            # 其他类型兜底
            return {"shared_stats": base_stats, **custom}
        except Exception as e:
            # 最后兜底：只返回 batch 自己的统计，至少不 NaN
            self.logger.warning(f"无法获取共享代理的统计信息: {e}")
            merged_stats = {}
            try:
                stats = base_agent.get_statistics()
                for j, stat in enumerate(stats):
                    merged_stats[f"shared_stat_{j}"] = stat
            except Exception:
                pass
            return merged_stats

    def eval_mode(self):
        """切换到评估模式，返回上下文管理器"""
        # pfrl期望eval_mode返回一个上下文管理器
        from contextlib import contextmanager

        @contextmanager
        def eval_context():
            # 切换到评估模式
            for tile_agent in self.cg_ppo_agent.tile_agents:
                tile_agent.eval_mode()

            try:
                yield
            finally:
                # pfrl会在需要时自动处理训练模式的恢复
                pass

        return eval_context()

    @property
    def training(self):
        """是否处于训练模式"""
        # 返回第一个代理的训练状态（所有代理状态应该一致）
        return self.cg_ppo_agent.tile_agents[0].training if self.cg_ppo_agent.tile_agents else False

    @property
    def saved_attributes(self):
        """需要保存的属性"""
        # 由于我们使用自定义的 save/load 方法，返回空列表
        return []
