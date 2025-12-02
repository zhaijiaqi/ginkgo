

**PPO-Guided Mixed-Precision Control for Conjugate Gradient (CG)
with Block-wise SpMV Precision Selection
(Full Simulation Framework for Python + pfrl)**

---

# 1. Project Goal

本项目旨在构建一个完整模拟器，用于验证 **PPO 控制 CG 算法中的 SpMV 分块精度** 的可行性。

目标是实现一个可训练的 RL 环境，其中：

* **外层循环：CG迭代**
* **内层核心算子：SpMV（A·p）使用 tile-based precision actions**
* PPO 代理在 SpMV 时对每个 tile 根据 `sub_p_j` 状态选择计算精度
* 整个 CG 的计算过程（包括 r, p, x, alpha, beta）都被模拟
* 每个 tile 的不同精度将影响：

  * SpMV 的延迟/成本（cost）
  * SpMV 结果的数值误差（error）
  * 最终 CG 收敛速度与终止精度

最终 PPO 学到的策略用于最小化：

* **CG 的总时间 cost**
* **在给定误差容忍下尽量保持收敛质量**

---

# 2. System Architecture

系统包含三大部分：

1. **CG Simulator** — 模拟完整 CG 迭代，包括 r、p、A·p、x、alpha、beta
2. **SpMV Block Simulator** — 将 A·p 分成 tiles，每个 tile 由 PPO 决定精度
3. **PPO Agent** — 学习在每个 tile 选择精度

目录设计如下：

```
project_root/
│
├── env/
│   └── cg_env.py
│
├── simulator/
│   ├── spmv_block_simulator.py
│   └── cg_math_simulator.py
│
├── agent/
│   └── ppo_agent_factory.py
│
├── train/
│   └── train_cg_with_ppo.py
│
├── eval/
│   └── evaluate_agent.py
│
├── config/
│   └── default.yaml
│
└── log/
```

---

# 3. Full CG Algorithm (Simulator Implementation Plan)

CG 求解 Ax=b 的基本流程：

1. 初始化

   ```
   x0 = 0  
   r0 = b − A·x0  
   p0 = r0
   ```

2. 对于每次迭代 k：
   a. 计算 **A·p_k**（这里是 RL 控制的模块）
   b. 计算：

   ```
   alpha_k = (r_k^T r_k) / (p_k^T Ap_k)
   x_{k+1} = x_k + alpha_k * p_k
   r_{k+1} = r_k - alpha_k * Ap_k
   ```

   c. 判断收敛
   d. 更新方向：

   ```
   beta_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
   p_{k+1} = r_{k+1} + beta_k p_k
   ```

在模拟器中，我们不做真实 SpMV，而是使用下面定义的 **block-based precision simulation** 来生成 `Ap_k`。

---

# 4. SpMV Block Simulator

## 4.1 Tile Structure

* 假设 A 按列分块，每块 tilesize 列
* 对应 p 中 continuous tilesize 元素称为 `sub_p_j`
* 对每个 tile，都需要 PPO 决定精度

## 4.2 Precision Choices (action space)

```
0 → fp64
1 → fp32
2 → tf32
3 → fp16
4 → bf16
5 → fp8
```

## 4.3 Simulation Outputs

模拟器给出：

```
(spmv_partial_result, simulated_cost, simulated_error)
```

其中：

* cost：越低精度 → 速度越快（可配置）
* error：越低精度 → quantization noise 越大（可配置）

## 4.4 Aggregation

完成所有 tiles 后：

```
Ap = sum(spmv_partial_result_j + noise_j)
```

noise 来自不同精度的 error accumulation。

---

# 5. RL Environment Design

环境必须模拟 **整个 CG 的迭代过程**。

## 5.1 Episode Definition

* 一个 episode = 一个完整的 CG 求解过程
* PPO 要在每一次 SpMV 的每个 tile 上执行一次 action
* reward 在 CG 每次迭代或 episode 末统一发放（通过配置选择）

## 5.2 State (per tile)

每次 SpMV 的 tile 状态定义为：

```
state_j = {
  sub_p_j,  
  ||sub_p_j||1, 
  ||sub_p_j||2,
  max(abs(sub_p_j)),
  iteration index k,
}
```

可选开启标准化。

## 5.3 Step Transition

一次 tile 精度选择属于单步 step。

整个 CG 的一次 Ap_k 计算（多个 tiles）是一段 sub-episode。

下面给出**仅修改后的 Reward 描述部分**，可直接替换到你前面开发文档的相应段落中：

## 5.4 Reward 设计

为了使 PPO 策略在模拟器中学习到“既保证 CG 收敛、又降低计算成本”的精度分配策略，我们将 Reward 设计为多个指标的加权组合：

[
\text{Reward} = w_1 \cdot R_{\text{err}} + w_2 \cdot R_{\text{cost}} + w_3 \cdot I(|r_k|<\epsilon)
]

其中：

---

### 1. 数值误差奖励（ (R_{\text{err}}) ）**

衡量在本 tilesize 分块的 SpMV 运算中，由精度选择导致的局部数值误差：

[
R_{\text{err}} = - | \hat{A}_j p_j - A_j p_j |
]

数值误差越小，奖励越高。

---

### 2. 计算成本奖励（ (R_{\text{cost}}) ）**

不同精度带来的 GPU 计算代价不同（例如 fp8 最便宜，fp64 最贵）。定义：

[
R_{\text{cost}} = - \text{cost}(\text{precision})
]

cost 可依据 GPU 模型（如 H100）上的 TFLOPs 或 latency 表，使用归一化代价。

---

### 3. 收敛性奖励（新增关键部分）**

考虑到 CG 的目标是收敛至误差阈值 (\epsilon)，如果整个求解过程最终**成功收敛**，给予一个一次性正奖励：

[
R_{\text{conv}} = w_3 \cdot I(|r_k| < \epsilon)
]

* (I(\cdot)) 为指标函数，CG 最终残差 norm 满足阈值条件时为 1，否则为 0
* 该奖励通常在一个 episode（一次完整 CG 求解）结束时给出
* 可显著促进 PPO 策略学习“全球有效的精度分配策略”而非局部贪心

---

### **奖励设计总结**

| 组件                  | 含义      | 作用           |
| ------------------- | ------- | ------------ |
| (R_{\text{err}})    | 本分块数值误差 | 鼓励使用更高精度（可控） |
| (R_{\text{cost}})   | 本分块计算代价 | 鼓励使用低精度      |
| (I(|r_k|<\epsilon)) | 全局收敛性奖励 | 强制策略在整体求解上可行 |

$w_1,w_2,w_3$可以分别设定为：1,0.1,10


---

# 6. Simulator Modules

## 6.1 `cg_math_simulator.py`

提供线性代数：

* vector dot
* saxpy
* norm
* residual tracking

不依赖 numpy 的精度，只负责逻辑。

## 6.2 `spmv_block_simulator.py`

提供以下接口：

```
simulate_spmv_block(
  sub_vector,
  precision_action
) -> partial_y, cost, error
```

以及：

```
simulate_full_spmv(
  p_vector,
  actions_for_all_tiles
) -> Ap_vector, total_cost, total_error
```

---

# 7. PPO Agent

## 7.1 Agent Factory

```
AgentFactory(config) → PPO agent
```

## 7.2 Policy Input

policy 网络输入维度：

```
dim = tilesize_features
```

输出：

```
softmax over 6 precision actions
```

---

# 8. Training Pipeline

## 8.1 Rollout

训练过程：

1. Reset env → produce (r0, p0, A, b)
2. For each CG iteration:

   * Loop through tiles:

     * PPO observes state_j
     * PPO outputs precision action_j
     * env collects (partial_y, cost, error)
   * Aggregates all partial_y to Ap
   * Update CG variables
3. Track rewards
4. PPO updates every N rollouts
5. Log everything

## 8.2 Logging

* cost 曲线
* error 曲线
* residual 曲线
* p-norm 变化
* 动作分布随训练变化
* Ap 精度逐 tile 分析

---

# 9. Evaluation Metrics

## 9.1 CG Convergence Quality

* residual vs iteration
* 与全 fp64 基线对比

## 9.2 Cost Reduction

* total simulated runtime
* average cost per tile

## 9.3 Precision Policy Visualization

* action histogram
* per-tile action heatmap

## 9.4 Ablation Studies

* remove cost penalty
* remove error penalty
* shuffle tile order
* unseen A 分布上的泛化测试

---

# 10. Config System

示例参数：

```
cg:
  max_iter: 50
  stop_tol: 1e-6

spmv:
  tilesize: 32
  precision_cost_table:
    fp64: 1.0
    fp32: 0.7
    tf32: 0.55
    fp16: 0.35
    bf16: 0.33
    fp8: 0.15

reward:
  type: weighted
  alpha: 1.0
  beta: 0.1
```

---

# 11. Development Stages (Cursor Iteration Plan)

### **Stage 1 — CG math simulator**

* 前后向向量计算
* residual tracking

### **Stage 2 — SpMV tile simulator**

* 多精度 cost/error 模型
* tile aggregation

### **Stage 3 — CG Environment**

* CG iteration loop
* tile-level step interface
* reward 汇总接口

### **Stage 4 — PPO Agent**

* pfrl PPO
* policy/value 网络
* rollout + update

### **Stage 5 — Logging + Eval**

* residual 曲线
* cost/error 曲线
* 动作可视化

### **Stage 6 — 扩展 (可选)**

* curriculum learning
* 多样化矩阵
* 多精度误差模型

---

# 12. Summary

本开发文档：

* 覆盖 **CG 全流程模拟**
* 定义 **tile-level SpMV precision control**
* 提供状态、动作、奖励、收敛指标
* 完整训练、评估、日志框架
* 可直接给 Cursor 自动分模块生成代码

如需，我还能生成：

* **状态特征（state encoder）更复杂版本**
* **多种 reward 的数学推导**
* **CG 误差传播模型推导（quantization error → residual error）**

告诉我即可。
