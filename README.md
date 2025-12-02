# PPO-Guided Mixed-Precision Control for Conjugate Gradient with Block-wise SpMV Precision Selection

🚀 使用方法
1. 训练代理：
```
python train/train_cg_with_ppo.py
```
2. 评估代理：
```
python eval/evaluate_agent.py --model_path log/cg_ppo_*/final_model.pt
```