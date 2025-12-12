#!/bin/bash

# 并行训练脚本：同时至多运行10个CG PPO训练任务
# 用法: ./train_parallel.sh [矩阵名称1] [矩阵名称2] ...
# 如果不提供参数，则从valid_matrix_set.csv读取所有矩阵

set -e  # 遇到错误立即退出

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/train/train_cg_with_ppo.py"
MATRIX_CSV="${PROJECT_ROOT}/valid_matrix_set.csv"
CONFIG_FILE="${PROJECT_ROOT}/config/default.yaml"

# 默认并发数
MAX_JOBS=10

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

    log_info "[$job_id] 开始训练矩阵: $matrix_name"

    # 创建临时配置文件
    local temp_config="/tmp/train_config_${matrix_name}_$$.yaml"
    sed "s/matrix_name: .*/matrix_name: $matrix_name/" "$CONFIG_FILE" > "$temp_config"

    # 运行训练
    if python3 "$TRAIN_SCRIPT" --config "$temp_config" --single-matrix; then
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

    local completed_jobs=0
    local failed_jobs=0

    # 首先启动尽可能多的job（最多MAX_JOBS个）
    local jobs_to_start=$((total_matrices < MAX_JOBS ? total_matrices : MAX_JOBS))
    log_info "同时启动 $jobs_to_start 个训练进程..."

    for i in $(seq 0 $((jobs_to_start - 1))); do
        local matrix_name="${matrix_names[$i]}"
        local job_id=$((i + 1))

        # 启动后台训练任务
        {
            if train_single_matrix "$matrix_name" "$job_id"; then
                ((completed_jobs++))
            else
                ((failed_jobs++))
            fi
        } &

        log_info "[$job_id] 已启动训练任务: $matrix_name"
    done

    local next_matrix_index=$jobs_to_start

    # 当还有矩阵需要训练时，持续监控并启动新job
    while [ $next_matrix_index -lt $total_matrices ]; do
        # 等待有空闲slot
        while [ "$(jobs -r | wc -l)" -ge $MAX_JOBS ]; do
            sleep 2  # 每2秒检查一次
        done

        # 启动下一个job
        local matrix_name="${matrix_names[$next_matrix_index]}"
        local job_id=$((next_matrix_index + 1))

        # 启动后台训练任务
        {
            if train_single_matrix "$matrix_name" "$job_id"; then
                ((completed_jobs++))
            else
                ((failed_jobs++))
            fi
        } &

        log_info "[$job_id] 已启动训练任务: $matrix_name"
        ((next_matrix_index++))
    done

    # 等待最后一批任务完成
    log_info "等待所有训练任务完成..."
    wait

    # 统计结果
    local final_running=$(jobs -r | wc -l)
    log_info "训练完成统计:"
    log_info "  总矩阵数: $total_matrices"
    log_info "  成功完成: $completed_jobs"
    log_info "  失败数量: $failed_jobs"
    log_info "  运行中: $final_running"

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
并行训练脚本：同时至多运行10个CG PPO训练任务

用法:
  $0                    # 从valid_matrix_set.csv读取所有矩阵并训练
  $0 matrix1 matrix2    # 只训练指定的矩阵
  $0 --help             # 显示此帮助信息

选项:
  --help          显示此帮助信息
  --max-jobs N    设置最大并发数 (默认: 10)

示例:
  $0                          # 训练所有矩阵，最多10个并发
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
