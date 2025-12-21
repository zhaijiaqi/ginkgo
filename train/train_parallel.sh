#!/bin/bash

# 并行训练脚本：分批运行CG PPO训练任务，每次最多12个并发（4张GPU，每张GPU平均3个任务）
# 用法: ./train_parallel.sh [矩阵名称1] [矩阵名称2] ...
# 如果不提供参数，则从cg_results.csv读取所有矩阵

set -e  # 遇到错误立即退出

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/train/train_cg_with_ppo.py"
MATRIX_CSV="${PROJECT_ROOT}/cg_results.csv"
CONFIG_FILE="${PROJECT_ROOT}/config/default.yaml"

# 默认并发数：4张GPU，每张GPU训练3个数据集
MAX_JOBS=12
NUM_GPUS=4  # 可用的GPU数量

# GPU分配策略：确保每个GPU分配的任务数量更均衡
get_gpu_for_task() {
    local task_index="$1"
    # 简单的轮询分配：0->0, 1->1, 2->2, 3->3, 4->0, 5->1, ...
    echo $((task_index % NUM_GPUS))
}

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

# 检查依赖
check_dependencies() {
    if ! command -v python3 &> /dev/null; then
        log_error "未找到 python3 命令"
        exit 1
    fi

    if [ ! -f "$TRAIN_SCRIPT" ]; then
        log_error "训练脚本不存在: $TRAIN_SCRIPT"
        exit 1
    fi

    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi

    if [ ! -f "$MATRIX_CSV" ]; then
        log_error "矩阵CSV文件不存在: $MATRIX_CSV"
        exit 1
    fi
}

# 从CSV文件读取矩阵名称列表
read_matrix_names() {
    local csv_file="$1"
    # 跳过第一行标题，提取第3列（Name列）
    tail -n +2 "$csv_file" | cut -d',' -f3 | sed 's/"//g' | tr -d '\r'
}

# 训练单个矩阵的函数
train_single_matrix() {
    local matrix_name="$1"
    local job_id="$2"
    local gpu_id="$3"

    log_info "[$job_id] 开始训练矩阵: $matrix_name (GPU: $gpu_id)"

    # 创建临时配置文件
    local temp_config="/tmp/train_config_${matrix_name}_$$.yaml"
    sed "s/matrix_name: .*/matrix_name: $matrix_name/" "$CONFIG_FILE" > "$temp_config"

    # 设置CUDA_VISIBLE_DEVICES来强制使用特定GPU，然后运行训练
    if CUDA_VISIBLE_DEVICES="$gpu_id" python3 "$TRAIN_SCRIPT" --config "$temp_config" --single-matrix --gpu 0; then
        log_info "[$job_id] 矩阵 $matrix_name 训练成功完成"
        rm -f "$temp_config"
        return 0
    else
        local exit_code=$?
        log_error "[$job_id] 矩阵 $matrix_name 训练失败 (退出码: $exit_code)"
        rm -f "$temp_config"
        return $exit_code
    fi
}

# 等待队列中的任务
wait_for_slot() {
    local max_jobs="$1"

    while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
        sleep 1
    done
}

# 主函数
main() {
    check_dependencies

    local matrix_names=()

    # 如果提供了命令行参数，使用参数作为矩阵名称
    if [ $# -gt 0 ]; then
        matrix_names=("$@")
    else
        # 否则从CSV文件读取所有矩阵名称
        log_info "从 $MATRIX_CSV 读取矩阵列表..."
        mapfile -t matrix_names < <(read_matrix_names "$MATRIX_CSV")
    fi

    local total_matrices=${#matrix_names[@]}
    log_info "发现 $total_matrices 个矩阵需要训练: ${matrix_names[*]}"
    log_info "最大并发数: $MAX_JOBS"
    log_info "可用GPU数量: $NUM_GPUS"

    # 分批启动训练任务，每次最多启动MAX_JOBS个任务
    log_info "分批启动 $total_matrices 个训练进程，每次最多 $MAX_JOBS 个并发..."
    log_info "每个GPU将平均分配约 $((MAX_JOBS / NUM_GPUS)) 个任务"
    log_info "GPU分配策略: 循环分配给 $NUM_GPUS 张GPU"

    local pids=()
    local result_files=()

    # 创建临时目录用于存储任务结果
    local temp_dir="/tmp/rlcg_training_$$"
    mkdir -p "$temp_dir"

    # 分批启动任务，每次最多启动MAX_JOBS个任务
    for i in $(seq 0 $((total_matrices - 1))); do
        local matrix_name="${matrix_names[$i]}"
        local job_id=$((i + 1))
        local gpu_id=$(get_gpu_for_task "$i")  # 使用函数分配GPU
        local result_file="$temp_dir/job_${job_id}.result"

        # 等待有空闲slot
        wait_for_slot "$MAX_JOBS"

        # 启动后台训练任务（GPU分配在train_single_matrix函数中处理）
        train_single_matrix "$matrix_name" "$job_id" "$gpu_id" &
        local pid=$!
        pids[$i]=$pid

        log_info "[$job_id] 已启动训练任务: $matrix_name (GPU: $gpu_id, PID: $pid)"
    done

    # 显示当前运行状态
    log_info "所有任务已启动，开始监控执行状态..."
    log_info "运行中的任务数: $(jobs -r | wc -l)"

    # 等待所有任务完成
    log_info "等待所有训练任务完成..."
    local start_time=$(date +%s)
    local last_report_time=$start_time

    while true; do
        local running_count=$(jobs -r | wc -l)
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        # 每30秒报告一次进度
        if [ $((current_time - last_report_time)) -ge 30 ]; then
            log_info "训练进行中... 已运行 ${elapsed}秒，剩余运行任务: $running_count"
            last_report_time=$current_time
        fi

        # 检查是否所有任务都完成了
        if [ $running_count -eq 0 ]; then
            break
        fi

        sleep 5
    done

    # 确保所有后台进程都已终止
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
        fi
    done

    # 统计结果
    local completed_jobs=0
    local failed_jobs=0
    local total_time=$(($(date +%s) - start_time))

    log_info "收集训练结果..."
    for result_file in "$temp_dir"/*.result; do
        if [ -f "$result_file" ]; then
            local result=$(cat "$result_file")
            if [[ $result == success:* ]]; then
                ((completed_jobs++))
            elif [[ $result == failed:* ]]; then
                ((failed_jobs++))
            fi
        fi
    done

    # 清理临时目录
    rm -rf "$temp_dir"

    log_info "🎯 训练完成统计:"
    log_info "  总矩阵数: $total_matrices"
    log_info "  成功完成: $completed_jobs"
    log_info "  失败数量: $failed_jobs"
    log_info "  总耗时: ${total_time}秒"
    log_info "  平均每个矩阵耗时: $((total_time / total_matrices))秒"
    log_info "  GPU数量: $NUM_GPUS"
    log_info "  设计并发数: $MAX_JOBS"

    if [ $failed_jobs -eq 0 ]; then
        log_info "🎉 所有训练任务成功完成!"
        exit 0
    else
        log_error "⚠️  有 $failed_jobs 个训练任务失败"
        exit 1
    fi
}

# 显示用法
show_usage() {
    cat << EOF
并行训练脚本：同时至多运行12个CG PPO训练任务（4张GPU，每张GPU训练3个数据集）

用法:
  $0                    # 从cg_results.csv读取所有矩阵并训练
  $0 matrix1 matrix2    # 只训练指定的矩阵
  $0 --help             # 显示此帮助信息

选项:
  --help          显示此帮助信息
  --max-jobs N    设置最大并发数 (默认: 12)

示例:
  $0                          # 训练所有矩阵，最多12个并发
  $0 bcsstk09 Muu            # 只训练bcsstk09和Muu两个矩阵
  $0 --max-jobs 4 bcsstk09   # 最多4个并发，训练bcsstk09

EOF
}

# 处理命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_usage
            exit 0
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            if ! [[ "$MAX_JOBS" =~ ^[0-9]+$ ]] || [ "$MAX_JOBS" -le 0 ]; then
                log_error "无效的并发数: $MAX_JOBS"
                exit 1
            fi
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# 运行主函数
main "$@"
