import petsc4py
petsc4py.init()
from petsc4py import PETSc
import scipy.io
import csv
import os
import sys
import argparse
import numpy as np

def load_matrix_list(csv_file):
    """从 CSV 文件加载矩阵字典，以 matrix_name 为键"""
    matrices = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            matrix_name = row['Name']
            matrices[matrix_name] = {
                'id': row['id'],
                'group': row['Group'],
                'name': matrix_name,
                'rows': int(row['rows']),
                'cols': int(row['cols']),
                'entries': int(row['entries'])
            }
    return matrices

def solve_matrix(matrix_info, matrix_dir):
    """对单个矩阵执行 CG 求解"""
    matrix_name = matrix_info['name']
    mtx_file = os.path.join(matrix_dir, f"{matrix_name}.mtx")

    if not os.path.exists(mtx_file):
        print(f"Warning: Matrix file '{mtx_file}' not found, skipping {matrix_name}")
        return None

    print(f"\n=== Solving matrix: {matrix_name} ===")
    print(f"Size: {matrix_info['rows']}x{matrix_info['cols']}")
    print(f"Entries: {matrix_info['entries']}")

    try:
        # === 1. 读取稀疏矩阵 (.mtx) ===
        scipy_mat = scipy.io.mmread(mtx_file).tocsr()  # CSR 格式

        n = scipy_mat.shape[0]

        # === 2. 创建 PETSc 矩阵 ===
        A = PETSc.Mat().createAIJ(size=scipy_mat.shape,
                                  csr=(scipy_mat.indptr, scipy_mat.indices, scipy_mat.data))
        A.assemble()

        # === 3. 创建向量 b 和 x ===
        b = PETSc.Vec().createSeq(n)
        x = PETSc.Vec().createSeq(n)

        # b 初始化为 A 的每列元素之和（通过 A^T * ones 得到）
        ones_vec = PETSc.Vec().createSeq(n)
        ones_vec.set(1.0)
        A.multTranspose(ones_vec, b)  # b = A^T * ones_vec (每列元素之和)
        
        # 输出b的L2范数
        b_norm = b.norm()
        print(f"L2 norm of b: {b_norm}")
        # b.scale(1.0 / b_norm)
        # print(f"L2 norm of b after scaling: {b.norm()}")

        # x 初始化为全0向量
        x.set(0.0)

        # === 4. 创建 KSP 求解器 ===
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setType('cg')       # CG 方法    
        ksp.getPC().setType('none') # 不使用预处理器

        # 设置收敛容差
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        ksp.setTolerances(rtol=1e-10, max_it=1000) # 如果想要用 ||r||_2 来判断，将 rtol 设置为 0, atol 设置为 1e-10

        # 可选：打印迭代信息
        ksp.setFromOptions()
        ksp.setMonitor(lambda ksp, its, rnorm: print(f"Iteration {its}, residual norm = {rnorm}"))

        # === 5. 求解 Ax = b ===
        print("Solving Ax = b using CG method...")
        start_time = os.times()[4]  # CPU time
        ksp.solve(b, x)
        end_time = os.times()[4]

        # 获取求解信息
        iterations = ksp.getIterationNumber()
        converged_reason = ksp.getConvergedReason()

        solve_time = end_time - start_time

        print(f"Converged in {iterations} iterations")
        print(f"residual norm: {ksp.getResidualNorm()}")
        print(f"Convergence reason: {converged_reason}")
        print(f"Solve time: {solve_time:.4f} seconds")


        # 只返回收敛的矩阵信息
        if converged_reason > 0:
            return {
                'id': matrix_info['id'],
                'group': matrix_info['group'],
                'name': matrix_name,
                'rows': matrix_info['rows'],
                'cols': matrix_info['cols'],
                'entries': matrix_info['entries'],
                'converged_iterations': iterations,
                'solve_time': solve_time
            }
        else:
            print(f"Matrix {matrix_name} did not converge")
            return None

    except Exception as e:
        print(f"Error solving matrix {matrix_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='PETSc CG solver - only saves converged matrices from valid_matrix_set.csv')
    parser.add_argument('--list', action='store_true', help='List all available matrices')
    parser.add_argument('--csv', type=str, default='valid_matrix_set.csv',
                       help='Path to matrix set CSV file (default: valid_matrix_set.csv)')
    parser.add_argument('--max-matrices', type=int, default=None,
                       help='Maximum number of matrices to solve (default: all)')
    parser.add_argument('--output', type=str, default='cg_results.csv',
                       help='Output CSV file for results (default: cg_results.csv)')

    args = parser.parse_args()

    # 加载矩阵列表
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' not found!")
        sys.exit(1)

    matrices = load_matrix_list(args.csv)
    print(f"Loaded {len(matrices)} matrices from {args.csv}")

    if args.list:
        print("Available matrices:")
        for i, (name, mat) in enumerate(matrices.items(), 1):
            print(f"{i}. {name}")
        return

    # 确定要处理的矩阵数量
    matrix_names = list(matrices.keys())
    num_to_solve = len(matrix_names) if args.max_matrices is None else min(args.max_matrices, len(matrix_names))
    print(f"Solving {num_to_solve} matrices...")

    matrix_dir = os.path.expanduser("~/data/matrix")
    results = []

    for i, matrix_name in enumerate(matrix_names[:num_to_solve], 1):
        matrix_info = matrices[matrix_name]
        print(f"\nProgress: {i}/{num_to_solve}")
        result = solve_matrix(matrix_info, matrix_dir)
        if result:
            results.append(result)

    # 保存结果到 CSV 文件
    if results:
        print(f"\nSaving {len(results)} converged matrices to {args.output}")
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'group', 'name', 'rows', 'cols', 'entries', 'converged_iterations', 'solve_time'])
            writer.writeheader()
            writer.writerows(results)

        print("Summary:")
        total_iterations = sum(r['converged_iterations'] for r in results)
        total_time = sum(r['solve_time'] for r in results)
        print(f"  Converged matrices: {len(results)}")
        print(f"  Average iterations to converge: {total_iterations/len(results):.1f}")
        print(f"  Average solve time: {total_time/len(results):.4f} seconds")

        # 记录收敛的矩阵
        converged_file = 'converged_matrices.txt'
        print(f"\nRecording converged matrices to {converged_file}")
        with open(converged_file, 'w') as f:
            f.write("# Converged matrices from CG solver\n")
            f.write(f"# Generated on {os.popen('date').read().strip()}\n")
            f.write(f"# Converged matrices: {len(results)}\n\n")
            for result in results:
                f.write(f"{result['name']}\n")
        print(f"Converged matrices saved to {converged_file}")
    else:
        print("No matrices converged!")

if __name__ == "__main__":
    main()
