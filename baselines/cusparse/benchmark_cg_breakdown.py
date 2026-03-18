#!/usr/bin/env python3
"""
基于 cuSPARSE baseline 的 CG 性能分解分析脚本
遍历 valid_matrix_set.csv 中的所有矩阵，运行 CG 求解器，并输出每个算子的性能分解
"""

import csv
import subprocess
import os
import sys
import json
from pathlib import Path

# 矩阵文件基础路径
MATRIX_BASE_PATH = '/mnt/data/matrix'
VALID_MATRIX_SET_CSV = '/root/program/kernels/valid_matrix_set.csv'
OUTPUT_CSV = 'cg_cusparse_breakdown.csv'

def find_matrix_file(matrix_name):
    """查找矩阵文件"""
    # 尝试多个可能的路径和扩展名
    base_dirs = [
        MATRIX_BASE_PATH,
        '/root/program/kernels',
        '/root/program',
        '../',
        '../../matrices',
        os.path.expanduser('~/data/matrix'),
    ]
    
    extensions = ['.mtx', '.MTX', '.mtx.gz', '.MTX.gz', '.mm', '.MM']
    
    for base_dir in base_dirs:
        for ext in extensions:
            path = os.path.join(base_dir, f"{matrix_name}{ext}")
            if os.path.exists(path):
                return path
    
    # 也尝试相对路径
    for ext in extensions:
        for rel_path in [f"../{matrix_name}{ext}", f"{matrix_name}{ext}"]:
            if os.path.exists(rel_path):
                return os.path.abspath(rel_path)
    
    return None

def read_matrix_names(csv_file):
    """从 CSV 文件读取矩阵名称"""
    matrix_data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            matrix_data.append({
                'id': row.get('id', ''),
                'name': row.get('Name', row.get('name', '')),
                'rows': int(row.get('rows', 0)),
                'cols': int(row.get('cols', 0)),
                'entries': int(row.get('entries', 0)),
            })
    return matrix_data

def run_cg_benchmark(matrix_path, max_iter=10000, tol=1e-5, verbose=False):
    """运行 CG benchmark 并解析输出"""
    # 检查可执行文件是否存在
    test_executable = './test_cusparse_cg'
    if not os.path.exists(test_executable):
        print(f"Error: {test_executable} not found. Please compile it first.")
        return None
    
    # 参数顺序：matrix_file max_iter tol --benchmark warmup iters --json
    # 将 --json 放在最后，避免与 --benchmark 的参数混淆
    cmd = [test_executable, matrix_path, str(max_iter), str(tol), '--benchmark', '3', '10', '--json']
    if verbose:
        cmd.insert(-1, '--verbose')  # 在 --json 之前插入 --verbose
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per matrix
        )
        
        if result.returncode != 0:
            print(f"Error running benchmark: {result.stderr}")
            return None
        
        # 解析输出
        output = result.stdout
        return parse_benchmark_output(output)
        
    except subprocess.TimeoutExpired:
        print(f"Timeout running benchmark for {matrix_path}")
        return None
    except Exception as e:
        print(f"Exception running benchmark: {e}")
        return None

def parse_benchmark_output(output):
    """解析 benchmark 输出（JSON 格式）"""
    result = {}
    
    # 查找 JSON 部分（在 JSON_RESULT_START 和 JSON_RESULT_END 之间）
    json_start_marker = 'JSON_RESULT_START'
    json_end_marker = 'JSON_RESULT_END'
    
    start_idx = output.find(json_start_marker)
    end_idx = output.find(json_end_marker)
    
    if start_idx >= 0 and end_idx > start_idx:
        # 提取 JSON 字符串（跳过标记行）
        json_lines = []
        in_json = False
        for line in output[start_idx:end_idx].split('\n'):
            if json_start_marker in line:
                in_json = True
                continue
            if in_json:
                json_lines.append(line)
        
        json_str = '\n'.join(json_lines)
        
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"JSON string: {json_str[:500]}...")
            return None
    
    # 如果没有找到 JSON，返回 None
    return None

def write_results_csv(results, output_file):
    """将结果写入 CSV 文件"""
    if not results:
        print("No results to write")
        return
    
    # 获取所有字段名（只保留时间占比）
    fieldnames = ['matrix_id', 'matrix_name', 'rows', 'cols', 'entries', 
                  'converged', 'iterations', 'final_residual',
                  'spmv_time_ratio',
                  'dot_time_ratio',
                  'axpy_time_ratio']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow(result)

def main():
    print("CG cuSPARSE Baseline Performance Breakdown Analysis")
    print("=" * 60)
    
    # 读取矩阵列表
    if not os.path.exists(VALID_MATRIX_SET_CSV):
        print(f"Error: {VALID_MATRIX_SET_CSV} not found")
        sys.exit(1)
    
    matrices = read_matrix_names(VALID_MATRIX_SET_CSV)
    print(f"Found {len(matrices)} matrices in {VALID_MATRIX_SET_CSV}")
    
    # 检查可执行文件
    test_executable = './test_cusparse_cg'
    if not os.path.exists(test_executable):
        print(f"Error: {test_executable} not found.")
        print("Please compile it first using: make")
        sys.exit(1)
    
    results = []
    
    for idx, matrix_info in enumerate(matrices, 1):
        matrix_name = matrix_info['name']
        print(f"\n[{idx}/{len(matrices)}] Processing matrix: {matrix_name}")
        
        # 查找矩阵文件
        matrix_path = find_matrix_file(matrix_name)
        if not matrix_path:
            print(f"  Warning: Matrix file not found for {matrix_name}, skipping...")
            continue
        
        print(f"  Matrix file: {matrix_path}")
        print(f"  Size: {matrix_info['rows']}x{matrix_info['cols']}, nnz: {matrix_info['entries']}")
        
        # 运行 benchmark
        print("  Running CG benchmark...")
        benchmark_result = run_cg_benchmark(matrix_path, max_iter=10000, tol=1e-5, verbose=False)
        
        if benchmark_result:
            # 添加矩阵信息
            result_row = {
                'matrix_id': matrix_info['id'],
                'matrix_name': matrix_name,
                'rows': matrix_info['rows'],
                'cols': matrix_info['cols'],
                'entries': matrix_info['entries'],
                'converged': benchmark_result.get('converged', False),
                'iterations': benchmark_result.get('iterations', 0),
                'final_residual': benchmark_result.get('final_residual', 0.0),
                'spmv_time_ratio': benchmark_result.get('spmv_time_ratio', 0.0),
                'dot_time_ratio': benchmark_result.get('dot_time_ratio', 0.0),
                'axpy_time_ratio': benchmark_result.get('axpy_time_ratio', 0.0),
            }
            
            results.append(result_row)
            print(f"  ✓ Completed (iterations: {result_row['iterations']}, converged: {result_row['converged']})")
        else:
            print(f"  ✗ Failed")
    
    # 写入结果
    if results:
        print(f"\nWriting results to {OUTPUT_CSV}...")
        write_results_csv(results, OUTPUT_CSV)
        print(f"✓ Results written to {OUTPUT_CSV}")
        print(f"  Total matrices processed: {len(results)}")
    else:
        print("\nNo results to write")

if __name__ == '__main__':
    main()
