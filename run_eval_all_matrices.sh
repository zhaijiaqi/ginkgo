#!/bin/bash

# 从 cg_results.csv 读取所有 matrix_name 并按顺序执行评估

MODEL_PATH="/home/bingxing2/home/scx7axu/program/rlcg/log/backup/251223/mesh2e1_tilesize64_20251223_172005/best_shared/model.pt"
CSV_FILE="cg_results.csv"
EVALUATOR_SCRIPT="eval/evaluator.py"

echo "开始批量评估所有矩阵..."
echo "模型路径: $MODEL_PATH"
echo "评估脚本: $EVALUATOR_SCRIPT"
echo "----------------------------------------"

# 读取 CSV 文件，跳过标题行，提取第3列（Name列）
MATRIX_NAMES=$(tail -n +2 "$CSV_FILE" | cut -d',' -f3)

TOTAL_MATRICES=$(echo "$MATRIX_NAMES" | wc -l)
CURRENT_COUNT=0

echo "总共需要评估 $TOTAL_MATRICES 个矩阵"
echo ""

for MATRIX_NAME in $MATRIX_NAMES; do
    CURRENT_COUNT=$((CURRENT_COUNT + 1))
    echo "[$CURRENT_COUNT/$TOTAL_MATRICES] 正在评估矩阵: $MATRIX_NAME"

    # 执行评估命令
    python "$EVALUATOR_SCRIPT" --model_path "$MODEL_PATH" --matrix_name "$MATRIX_NAME"

    # 检查命令是否成功执行
    if [ $? -eq 0 ]; then
        echo "✓ 矩阵 $MATRIX_NAME 评估完成"
    else
        echo "✗ 矩阵 $MATRIX_NAME 评估失败"
    fi

    echo "----------------------------------------"
done

echo "所有矩阵评估完成！"