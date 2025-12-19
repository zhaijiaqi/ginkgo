import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from eval.evaluator import run_episode_with_agent
from eval.validate_lightweight_selector import LightweightSelectorAgent
from eval.lightweight_precision_selector import extract_sub_p_features, LightweightPrecisionSelector

class LearnableLightweightPrecisionSelector:
    """
    可学习轻量级精度选择器

    将原来硬编码的参数（归一化因子、特征权重、精度阈值、迭代偏差）变成可学习的参数
    通过训练数据学习最优的参数组合
    """

    def __init__(self,
                 early_iter_threshold: int = 70,
                 use_iteration_factor: bool = True,
                 learning_rate: float = 0.001,
                 device: str = 'cpu'):
        """
        初始化可学习精度选择器

        Args:
            early_iter_threshold: 早期迭代阈值
            use_iteration_factor: 是否使用迭代次数因子
            learning_rate: 学习率
            device: 计算设备
        """
        self.early_iter_threshold = early_iter_threshold
        self.use_iteration_factor = use_iteration_factor
        self.device = device

        # 精度代码映射：0=fp64, 1=fp32, 2=tf32, 3=fp16, 4=bf16, 5=fp8
        self.precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']

        # 定义可学习参数
        self.learnable_params = nn.ParameterDict({
            # 归一化因子（用于特征score计算）
            'l2_norm_factor': nn.Parameter(torch.tensor(0.5, dtype=torch.float32)),
            'max_abs_factor': nn.Parameter(torch.tensor(0.2, dtype=torch.float32)),
            'l1_norm_factor': nn.Parameter(torch.tensor(5.0, dtype=torch.float32)),

            # 特征权重（l2, max_abs, l1的权重）
            'feature_weights': nn.Parameter(torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32)),

            # 精度选择阈值（对应fp8, bf16, fp16, tf32, fp32的分界点）
            'precision_thresholds': nn.Parameter(torch.tensor([1.2, 0.8, 0.4, 0.0, -0.3], dtype=torch.float32)),

            # 迭代偏差参数（不同阶段的偏差值）
            'iteration_biases': nn.Parameter(torch.tensor([0.8, 0.6, 0.2, -0.3], dtype=torch.float32)),

            # 迭代阶段阈值（30, early_iter_threshold, 120）
            'iteration_thresholds': nn.Parameter(torch.tensor([30.0, float(early_iter_threshold), 120.0], dtype=torch.float32))
        }).to(device)

        # 确保权重参数为正且和为1
        with torch.no_grad():
            self.learnable_params['feature_weights'].data = torch.softmax(self.learnable_params['feature_weights'], dim=0)
            # 确保阈值有序（从高到低）
            sorted_thresholds, _ = torch.sort(self.learnable_params['precision_thresholds'], descending=True)
            self.learnable_params['precision_thresholds'].data = sorted_thresholds

        self.optimizer = optim.Adam(self.learnable_params.parameters(), lr=learning_rate)

    def _normalize_features(self, l2_norm: float, max_abs: float, l1_norm: float) -> torch.Tensor:
        """归一化特征到0-1范围"""
        l2_norm_tensor = torch.tensor(l2_norm, dtype=torch.float32, device=self.device)
        max_abs_tensor = torch.tensor(max_abs, dtype=torch.float32, device=self.device)
        l1_norm_tensor = torch.tensor(l1_norm, dtype=torch.float32, device=self.device)

        l2_score = torch.clamp(l2_norm_tensor / self.learnable_params['l2_norm_factor'], max=1.0)
        max_abs_score = torch.clamp(max_abs_tensor / self.learnable_params['max_abs_factor'], max=1.0)
        l1_score = torch.clamp(l1_norm_tensor / self.learnable_params['l1_norm_factor'], max=1.0)

        return torch.stack([l2_score, max_abs_score, l1_score])

    def _compute_iteration_bias(self, iteration: int) -> torch.Tensor:
        """计算迭代偏差"""
        if not self.use_iteration_factor:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        iteration_tensor = torch.tensor(float(iteration), dtype=torch.float32, device=self.device)
        thresholds = self.learnable_params['iteration_thresholds']
        biases = self.learnable_params['iteration_biases']

        # 根据迭代次数选择对应的偏差
        if iteration_tensor < thresholds[0]:
            return biases[0]  # 非常早期
        elif iteration_tensor < thresholds[1]:
            return biases[1]  # 早期
        elif iteration_tensor < thresholds[2]:
            return biases[2]  # 中期
        else:
            return biases[3]  # 后期

    def _select_precision_from_score(self, final_score: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """根据最终分数选择精度（推理时使用）"""
        thresholds = self.learnable_params['precision_thresholds']

        if final_score > thresholds[0]:
            return 5, final_score  # fp8
        elif final_score > thresholds[1]:
            return 4, final_score  # bf16
        elif final_score > thresholds[2]:
            return 3, final_score  # fp16
        elif final_score > thresholds[3]:
            return 2, final_score  # tf32
        elif final_score > thresholds[4]:
            return 1, final_score  # fp32
        else:
            return 0, final_score  # fp64

    def select_precision(self, sub_p: np.ndarray, iteration: int) -> Tuple[int, float]:
        """
        为给定的 sub_p 选择精度

        Args:
            sub_p: 子向量
            iteration: 当前迭代次数

        Returns:
            (精度代码, 最终分数)
        """
        # 提取特征
        features = extract_sub_p_features(sub_p)
        l2_norm = features['l2_norm']
        max_abs = features['max_abs']
        l1_norm = features['l1_norm']

        # 归一化特征
        feature_scores = self._normalize_features(l2_norm, max_abs, l1_norm)

        # 计算综合特征分数
        weights = torch.softmax(self.learnable_params['feature_weights'], dim=0)
        feature_score = torch.sum(feature_scores * weights)

        # 计算迭代偏差
        iteration_bias = self._compute_iteration_bias(iteration)

        # 最终分数
        final_score = feature_score + iteration_bias

        # 确保分数在合理范围内
        final_score = torch.clamp(final_score, min=-1.0, max=2.0)

        # 选择精度
        precision, score = self._select_precision_from_score(final_score)

        return precision, score.item()

    def select_precisions(self, p_vector: np.ndarray, iteration: int, tilesize: int) -> List[int]:
        """
        为整个 p 向量的所有 tiles 选择精度

        Args:
            p_vector: 完整的 p 向量
            iteration: 当前迭代次数
            tilesize: tile 大小

        Returns:
            精度代码列表
        """
        matrix_size = len(p_vector)
        num_tiles = (matrix_size + tilesize - 1) // tilesize

        actions = []
        scores = []
        for tile_idx in range(num_tiles):
            start_idx = tile_idx * tilesize
            end_idx = min(start_idx + tilesize, matrix_size)
            sub_p = p_vector[start_idx:end_idx]

            # 如果最后一个 tile 不足 tilesize，补0
            if len(sub_p) < tilesize:
                sub_p = np.concatenate([sub_p, np.zeros(tilesize - len(sub_p))])

            precision, score = self.select_precision(sub_p, iteration)
            actions.append(precision)
            scores.append(score)

        # print(f"scores: {[round(s, 2) for s in scores]}")
        return actions

    def _compute_precision_logits(self, final_score: torch.Tensor) -> torch.Tensor:
        """计算精度分类的logits（用于交叉熵损失）"""
        thresholds = self.learnable_params['precision_thresholds']

        # 计算每个精度级别的得分
        # fp8 (5): score > threshold[0]
        # bf16 (4): threshold[1] < score <= threshold[0]
        # fp16 (3): threshold[2] < score <= threshold[1]
        # tf32 (2): threshold[3] < score <= threshold[2]
        # fp32 (1): threshold[4] < score <= threshold[3]
        # fp64 (0): score <= threshold[4]

        logits = torch.zeros(6, dtype=torch.float32, device=self.device)

        # 为每个精度级别计算得分
        logits[5] = torch.relu(final_score - thresholds[0])  # fp8
        logits[4] = torch.relu(final_score - thresholds[1]) * torch.relu(thresholds[0] - final_score + 1e-6)  # bf16
        logits[3] = torch.relu(final_score - thresholds[2]) * torch.relu(thresholds[1] - final_score + 1e-6)  # fp16
        logits[2] = torch.relu(final_score - thresholds[3]) * torch.relu(thresholds[2] - final_score + 1e-6)  # tf32
        logits[1] = torch.relu(final_score - thresholds[4]) * torch.relu(thresholds[3] - final_score + 1e-6)  # fp32
        logits[0] = torch.relu(thresholds[4] - final_score + 1e-6)  # fp64

        return logits

    def update_parameters(self, training_data: List[Tuple[np.ndarray, int, int]]):
        """
        使用训练数据更新可学习参数

        Args:
            training_data: 训练数据列表，每个元素为 (sub_p, iteration, target_precision)
        """
        self.optimizer.zero_grad()

        total_loss = 0.0
        batch_size = len(training_data)

        for (sub_p, iteration, target_precision) in training_data:
            # 前向传播
            features = extract_sub_p_features(sub_p)
            l2_norm = features['l2_norm']
            max_abs = features['max_abs']
            l1_norm = features['l1_norm']

            feature_scores = self._normalize_features(l2_norm, max_abs, l1_norm)
            weights = torch.softmax(self.learnable_params['feature_weights'], dim=0)
            feature_score = torch.sum(feature_scores * weights)

            iteration_bias = self._compute_iteration_bias(iteration)
            final_score = feature_score + iteration_bias
            final_score = torch.clamp(final_score, min=-1.0, max=2.0)

            # 计算预测精度的logits
            logits = self._compute_precision_logits(final_score)

            # 目标精度（one-hot编码）
            target_tensor = torch.zeros(6, dtype=torch.float32, device=self.device)
            target_tensor[target_precision] = 1.0

            # 计算损失（交叉熵损失）
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target_tensor.unsqueeze(0))
            total_loss += loss

        # 约束权重参数为正且和为1
        with torch.no_grad():
            self.learnable_params['feature_weights'].data = torch.softmax(self.learnable_params['feature_weights'], dim=0)
            # 确保阈值有序
            sorted_thresholds, _ = torch.sort(self.learnable_params['precision_thresholds'], descending=True)
            self.learnable_params['precision_thresholds'].data = sorted_thresholds

        avg_loss = total_loss / batch_size
        avg_loss.backward()
        self.optimizer.step()

        return avg_loss.item()

    def save_parameters(self, filepath: str):
        """保存学习到的参数"""
        torch.save({
            'learnable_params': self.learnable_params.state_dict(),
            'early_iter_threshold': self.early_iter_threshold,
            'use_iteration_factor': self.use_iteration_factor
        }, filepath)

    def load_parameters(self, filepath: str):
        """加载学习到的参数"""
        checkpoint = torch.load(filepath)
        self.learnable_params.load_state_dict(checkpoint['learnable_params'])
        self.early_iter_threshold = checkpoint['early_iter_threshold']
        self.use_iteration_factor = checkpoint['use_iteration_factor']

    def get_parameters_summary(self) -> Dict[str, float]:
        """获取当前参数的摘要"""
        summary = {}
        with torch.no_grad():
            summary['l2_norm_factor'] = self.learnable_params['l2_norm_factor'].item()
            summary['max_abs_factor'] = self.learnable_params['max_abs_factor'].item()
            summary['l1_norm_factor'] = self.learnable_params['l1_norm_factor'].item()

            weights = torch.softmax(self.learnable_params['feature_weights'], dim=0)
            summary['l2_weight'] = weights[0].item()
            summary['max_abs_weight'] = weights[1].item()
            summary['l1_weight'] = weights[2].item()

            thresholds = self.learnable_params['precision_thresholds']
            for i, name in enumerate(['fp8', 'bf16', 'fp16', 'tf32', 'fp32']):
                summary[f'{name}_threshold'] = thresholds[i].item()

            biases = self.learnable_params['iteration_biases']
            summary['very_early_bias'] = biases[0].item()
            summary['early_bias'] = biases[1].item()
            summary['mid_bias'] = biases[2].item()
            summary['late_bias'] = biases[3].item()

            iter_thresholds = self.learnable_params['iteration_thresholds']
            summary['very_early_threshold'] = iter_thresholds[0].item()
            summary['early_threshold'] = iter_thresholds[1].item()
            summary['mid_threshold'] = iter_thresholds[2].item()

        return summary


# 使用示例
if __name__ == "__main__":
    import numpy as np
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import yaml
    import torch
    import random
    import time
    from collections import defaultdict

    # 导入项目模块
    from env.cg_env import CGEnvironment
    from agent.ppo_agent_factory import PfrlCompatibleCGPPOAgent

    # 显式导入本地utils模块
    import importlib.util
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    utils_spec = importlib.util.spec_from_file_location("utils", os.path.join(project_root, "utils", "__init__.py"))
    utils = importlib.util.module_from_spec(utils_spec)
    sys.modules["utils"] = utils
    utils_spec.loader.exec_module(utils)

    from utils import create_env_config, load_model_weights
    from eval.validate_lightweight_selector import run_episode_with_selector

    def collect_training_data_from_model(model_path: str, config_path: str,
                                       matrix_name: str = "Muu", matrix_size: int = 512,
                                       num_episodes: int = 5, random_seed: int = 42):
        """
        从训练好的模型运行中收集训练数据

        Args:
            model_path: 模型路径
            config_path: 配置文件路径
            matrix_name: 矩阵名称
            matrix_size: 矩阵大小
            num_episodes: 收集的episode数量
            random_seed: 随机种子

        Returns:
            训练数据列表: [(sub_p, iteration, target_precision), ...]
        """
        print(f"从模型 {model_path} 收集训练数据...")

        # 加载配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 创建环境配置
        env_config = create_env_config(config, matrix_name=matrix_name, matrix_size=matrix_size)
        env_config['random_seed'] = random_seed

        # 创建环境
        env = CGEnvironment(env_config)
        tilesize = env.spmv_sim.tilesize

        # 加载模型
        cg_agent = load_model_weights(model_path, config, env)
        pfrl_agent = PfrlCompatibleCGPPOAgent(cg_agent)
        pfrl_agent.eval_mode()

        training_data = []

        for episode in range(num_episodes):
            print(f"收集第 {episode + 1}/{num_episodes} 个episode...")

            # 重置环境
            random.seed(random_seed + episode)
            np.random.seed(random_seed + episode)
            torch.manual_seed(random_seed + episode)

            obs = env.reset(seed=random_seed + episode)

            done = False
            step_count = 0

            while not done and step_count < 200:  # 防止无限循环
                # 获取当前状态
                iteration = env.current_iteration
                p_vector = env.p

                # 模型决策
                actions = pfrl_agent.act(obs)

                # 收集训练数据：对每个tile收集 (sub_p, iteration, target_precision)
                tile_idx = 0
                for action in actions:
                    start_idx = tile_idx * tilesize
                    end_idx = min(start_idx + tilesize, env.matrix_size)
                    sub_p = p_vector[start_idx:end_idx]

                    # 如果最后一个tile不足tilesize，补0
                    if len(sub_p) < tilesize:
                        sub_p = np.concatenate([sub_p, np.zeros(tilesize - len(sub_p))])

                    # action 就是目标精度 (0-5)
                    training_data.append((sub_p.copy(), iteration, int(action)))

                    tile_idx += 1

                # 执行动作
                obs, reward, done, info = env.step(actions)
                step_count += 1

                # 如果收敛了就停止
                if env.current_iteration >= env.max_iter:
                    break

            print(f"  Episode {episode + 1}: 收集了 {len(actions)} 个tile的决策数据")

        print(f"总共收集了 {len(training_data)} 个训练样本")
        return training_data

    def evaluate_learned_selector(selector, model_path: str, config_path: str,
                                matrix_name: str = "Muu", matrix_size: int = 512,
                                num_test_episodes: int = 3, random_seed: int = 42):
        """
        评估学习到的选择器性能，与全精度基准和初始参数选择器进行比较

        测试四种方法：
        1. full_precision: 全精度基准（所有tile都使用fp64）
        2. model: RL模型推理结果
        3. initial_lightweight: 使用初始参数的轻量级选择器
        4. learned_lightweight: 使用训练参数的轻量级选择器
        """
        print(f"\n评估学习到的选择器...")

        # 加载配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 创建环境配置
        env_config = create_env_config(config, matrix_name=matrix_name, matrix_size=matrix_size)
        env_config['random_seed'] = random_seed
        # 保存初始 b 向量（用于确保所有策略使用相同的问题）
        env_test = CGEnvironment(env_config)
        env_test.reset(seed=random_seed)
        saved_b = env_test.b.copy()
        
        results = {}
        results['full_precision'] = []
        results['model'] = []
        results['initial_lightweight'] = []
        results['learned_lightweight'] = []
        
        for episode in range(num_test_episodes):
            print(f"测试第 {episode + 1}/{num_test_episodes} 个episode...")
            
            # 测试全精度
            print("\n\n 测试全精度...")
            full_env = CGEnvironment(env_config)
            class FullPrecisionAgent:
                """全精度代理，总是选择fp64"""
                def __init__(self, env):
                    self.env = env

                def act(self, obs):
                    # 总是返回0（fp64）
                    tilesize = self.env.spmv_sim.tilesize
                    matrix_size = self.env.matrix_size
                    num_tiles = (matrix_size + tilesize - 1) // tilesize
                    return [0] * num_tiles

                def observe(self, obs, reward, done, done2):
                    pass

                def eval_mode(self):
                    pass

            full_agent = FullPrecisionAgent(full_env)
            full_result = run_episode_with_selector(full_env, full_agent, 
                                                    seed=random_seed, fixed_b=saved_b)
        
            # 测试模型
            print("\n\n 测试模型...")
            env_model = CGEnvironment(env_config)
            cg_agent = load_model_weights(model_path, config, env_model)
            pfrl_agent = PfrlCompatibleCGPPOAgent(cg_agent)
            pfrl_agent.eval_mode()
            model_result = run_episode_with_agent(env_model, pfrl_agent, 
                                                seed=random_seed, fixed_b=saved_b)
            
            # 测试初始参数的轻量级选择器
            print("\n\n 测试初始参数的轻量级选择器...")
            env_lw = CGEnvironment(env_config)
            lw_selector = LightweightPrecisionSelector(
                early_iter_threshold=70,
                use_iteration_factor=True
            )
            initial_agent = LightweightSelectorAgent(lw_selector, env_lw)
            initial_result = run_episode_with_selector(env_lw, initial_agent, 
                                                seed=random_seed, fixed_b=saved_b)
            
            # 测试学习参数的轻量级选择器
            print("\n\n 测试学习参数的轻量级选择器...")
            env_learned = CGEnvironment(env_config)
            learned_selector = LearnableLightweightPrecisionSelector(
                early_iter_threshold=70,
                use_iteration_factor=True
            )
            learned_agent = LightweightSelectorAgent(learned_selector, env_learned)
            learned_result = run_episode_with_selector(env_learned, learned_agent, 
                                                seed=random_seed, fixed_b=saved_b)

            # 记录结果
            results['full_precision'].append({
                'iterations': full_result['iterations'],
                'final_residual': full_result['final_residual'],
                'action history': full_result['precision_history'],
                'compute_cost': full_result['compute_cost']
            })

            results['model'].append({
                'iterations': model_result['iterations'],
                'final_residual': model_result['final_residual'],
                'action history': model_result['precision_history'],
                'compute_cost': model_result['compute_cost']
            })

            results['initial_lightweight'].append({
                'iterations': initial_result['iterations'],
                'final_residual': initial_result['final_residual'],
                'action history': initial_result['precision_history'],
                'compute_cost': initial_result['compute_cost']
            })

            results['learned_lightweight'].append({
                'iterations': learned_result['iterations'],
                'final_residual': learned_result['final_residual'],
                'action history': learned_result['precision_history'],
                'compute_cost': learned_result['compute_cost']
            })

        # 统计结果
        print("\n" + "="*80)
        print("📊 评估结果对比")
        print("="*80)

        methods = {
            'full_precision': '全精度基准 (fp64)',
            'model': 'RL模型推理',
            'initial_lightweight': '初始参数轻量级选择器',
            'learned_lightweight': '学习参数轻量级选择器'
        }

        # 显示每个episode的结果
        for idx in range(num_test_episodes):
            print(f"\nEpisode {idx+1}:")
            for method_key, method_name in methods.items():
                res = results[method_key][idx]
                print(f"  {method_name}:")
                print(f"    迭代次数: {res['iterations']}(+{res['iterations']-results['full_precision'][idx]['iterations']})")
                print(f"    最终残差: {res['final_residual']:.6e}")
                print(f"    计算成本: {res['compute_cost']:.6f}")
                print(f"    性能提升: {((results['full_precision'][idx]['compute_cost']-res['compute_cost'])/results['full_precision'][idx]['compute_cost'])*100:.2f}%")
        # 计算平均性能
        print(f"\n" + "="*80)
        print("📈 平均性能统计")
        print("="*80)

        for method_key, method_name in methods.items():
            iterations_list = [r['iterations'] for r in results[method_key]]
            residuals_list = [r['final_residual'] for r in results[method_key]]

            avg_iterations = np.mean(iterations_list)
            std_iterations = np.std(iterations_list)
            avg_residual = np.mean(residuals_list)
            std_residual = np.std(residuals_list)

            print(f"\n{method_name}:")
            print(f"  平均迭代次数: {avg_iterations:.2f} ± {std_iterations:.2f}")
            print(f"  平均最终残差: {avg_residual:.6e} ± {std_residual:.6e}")

        # 比较精度分布
        def analyze_precision_dist(actions_list):
            dist = defaultdict(int)
            total = 0
            for precision_history in actions_list:  # precision_history 是 [[actions_iter1], [actions_iter2], ...]
                for actions in precision_history:  # actions 是 [action1, action2, ...]
                    for action in actions:
                        dist[int(action)] += 1
                        total += 1
            return {k: v/total for k, v in dist.items()} if total > 0 else {}

        print(f"\n" + "="*80)
        print("🎯 精度使用分布")
        print("="*80)

        precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']

        for method_key, method_name in methods.items():
            if method_key == 'full_precision':
                # 全精度都是fp64
                dist = {0: 1.0}
            else:
                dist = analyze_precision_dist([r['action history'] for r in results[method_key]])

            print(f"\n{method_name}:")
            for prec in range(6):
                pct = dist.get(prec, 0) * 100
                print(f"  {precision_names[prec]}: {pct:.1f}%")

        return results

# 主函数
if __name__ == "__main__":
    model_path = "/home/bingxing2/home/scx7axu/program/rlcg/log/size512_tilesize64_20251209_112747/best_shared/model.pt"
    config_path = "/home/bingxing2/home/scx7axu/program/rlcg/config/default.yaml"
    matrix_name = "Muu"
    matrix_size = 512
    epochs = 200
    max_traing_size = 100

    print("=" * 80)
    print("🧠 训练可学习轻量级精度选择器")
    print("=" * 80)

    # 1. 收集训练数据
    print("\n📊 步骤1: 从真实模型收集训练数据")
    training_data = collect_training_data_from_model(
        model_path=model_path,
        config_path=config_path,
        matrix_name=matrix_name,
        matrix_size=matrix_size,
        num_episodes=1,
        random_seed=42
    )

    # 2. 创建可学习选择器
    print("\n🤖 步骤2: 创建可学习选择器")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    selector = LearnableLightweightPrecisionSelector(
        early_iter_threshold=70,
        use_iteration_factor=False,
        learning_rate=0.1,
        device=device
    )

    # 打印初始参数
    print("\n初始参数:")
    initial_params = selector.get_parameters_summary()
    for key, value in initial_params.items():
        print(f"{key}: {value:8.4f}")

    # 3. 训练选择器
    print("\n🎓 步骤3: 训练选择器")
    # 只随机采样 max_traing_size个training_data 来训练
    if len(training_data) > max_traing_size:
        idxs = np.random.choice(len(training_data), max_traing_size, replace=False)
        sampled_training_data = [training_data[i] for i in idxs]
    else:
        sampled_training_data = training_data
    training_data = sampled_training_data
    print(f"训练数据量: {len(training_data)}")

    for epoch in range(epochs):
        loss = selector.update_parameters(training_data)
        print(f"Epoch {epoch:3d}, Loss: {loss:.6f}")

    # 4. 打印训练后的参数
    print("\n训练后的参数:")
    trained_params = selector.get_parameters_summary()
    for key, value in trained_params.items():
        print(f"{key}: {value:8.4f}")

    # 5. 评估性能
    print("\n📈 步骤4: 评估性能")
    eval_results = evaluate_learned_selector(
        selector=selector,
        model_path=model_path,
        config_path=config_path,
        matrix_name=matrix_name,
        matrix_size=matrix_size,
        num_test_episodes=1,  # 只测试1个episode
        random_seed=42
    )

    # 6. 保存模型
    save_path = f"learned_precision_selector_{matrix_name}_{matrix_size}.pth"
    selector.save_parameters(save_path)
    print(f"\n💾 参数已保存到 {save_path}")

    print("\n✅ 训练完成！")

