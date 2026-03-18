#include "cusparse_cg_solver.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>

void print_benchmark_results(const CGSolverResult& result, const std::string& matrix_file, int m, int n, int nnz) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CG Solver Benchmark Results (cuSPARSE Baseline)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Matrix: " << matrix_file << " (" << m << "x" << n << ", nnz=" << nnz << ")" << std::endl;
    
    // 时间占比统计（只输出 spmv, dot product, axpy 的时间占比）
    std::cout << "\nOperator Time Ratio Breakdown:" << std::endl;
    std::cout << "  SpMV time ratio:        " << std::fixed << std::setprecision(2) << result.spmv_time_ratio << " %" << std::endl;
    std::cout << "  Dot product ratio:      " << std::fixed << std::setprecision(2) << result.dot_time_ratio << " %" << std::endl;
    std::cout << "  Axpy ratio:            " << std::fixed << std::setprecision(2) << result.axpy_time_ratio << " %" << std::endl;
    double other_ratio = 100.0 - result.spmv_time_ratio - result.dot_time_ratio - result.axpy_time_ratio;
    if (other_ratio > 0.01) {
        std::cout << "  Other (sync/overhead): " << std::fixed << std::setprecision(2) << other_ratio << " %" << std::endl;
    }
    
    std::cout << "\nIterations:" << std::endl;
    std::cout << "  Total: " << result.iterations << std::endl;
    std::cout << "  Converged: " << (result.converged ? "Yes" : "No") << std::endl;
    std::cout << "  Final residual: " << std::scientific << std::setprecision(2) << result.final_residual << std::endl;
    std::cout << "  Initial residual: " << std::scientific << std::setprecision(2) << result.residual_history[0] << std::endl;
    if (result.residual_history.size() > 1) {
        double reduction = result.residual_history[0] / result.final_residual;
        std::cout << "  Reduction factor: " << std::scientific << std::setprecision(2) << reduction << std::endl;
    }
    
    std::cout << std::string(60, '=') << std::endl;
}

// 输出 JSON 格式的结果（用于脚本解析）
void print_json_results(const CGSolverResult& result, const std::string& matrix_file, int m, int n, int nnz) {
    std::cout << "JSON_RESULT_START" << std::endl;
    std::cout << "{" << std::endl;
    std::cout << "  \"matrix_file\": \"" << matrix_file << "\"," << std::endl;
    std::cout << "  \"rows\": " << m << "," << std::endl;
    std::cout << "  \"cols\": " << n << "," << std::endl;
    std::cout << "  \"nnz\": " << nnz << "," << std::endl;
    std::cout << "  \"converged\": " << (result.converged ? "true" : "false") << "," << std::endl;
    std::cout << "  \"iterations\": " << result.iterations << "," << std::endl;
    std::cout << "  \"final_residual\": " << std::scientific << std::setprecision(15) << result.final_residual << "," << std::endl;
    std::cout << "  \"spmv_time_ratio\": " << std::fixed << std::setprecision(2) << result.spmv_time_ratio << "," << std::endl;
    std::cout << "  \"dot_time_ratio\": " << std::fixed << std::setprecision(2) << result.dot_time_ratio << "," << std::endl;
    std::cout << "  \"axpy_time_ratio\": " << std::fixed << std::setprecision(2) << result.axpy_time_ratio << std::endl;
    std::cout << "}" << std::endl;
    std::cout << "JSON_RESULT_END" << std::endl;
}

static double l2_error_to_ones(const std::vector<double>& x) {
    double acc = 0.0;
    for (double v : x) {
        const double d = v - 1.0;
        acc += d * d;
    }
    return std::sqrt(acc);
}

// 读取矩阵信息（nnz）
int read_matrix_nnz(const std::string& matrix_file) {
    int nnz = 0;
    std::ifstream file(matrix_file);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            if (line[0] != '%' && !line.empty()) {
                std::istringstream iss(line);
                int temp_m, temp_n, temp_nnz;
                if (iss >> temp_m >> temp_n >> temp_nnz) {
                    nnz = temp_nnz;
                    break;
                }
            }
        }
        file.close();
    }
    return nnz;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file.mtx> [max_iter] [tol] [--benchmark] [warmup] [iters] [--json] [--csv] [--csv_out <path>]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../bodyy4.mtx 1000 1e-10" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../bodyy4.mtx 1000 1e-10 --benchmark 3 10" << std::endl;
        std::cerr << "Example: " << argv[0] << " /data/matrix/bodyy4.mtx 100 1e-6 --csv --csv_out cusparse_cg_a100.csv" << std::endl;
        return 1;
    }
    
    std::string matrix_file = argv[1];
    CGSolverConfig config;
    
    // 解析位置参数（跳过选项）
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        // 如果遇到选项，停止解析位置参数
        if (arg.length() >= 2 && arg[0] == '-' && arg[1] == '-') {
            break;
        }
        // 解析 max_iter
        if (i == 2) {
            try {
                config.max_iter = std::stoi(arg);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid max_iter value, using default: " << config.max_iter << std::endl;
            }
        }
        // 解析 tol
        if (i == 3) {
            try {
                config.tol = std::stod(arg);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid tol value, using default: " << config.tol << std::endl;
            }
        }
    }
    
    bool do_benchmark = false;
    bool output_json = false;
    bool output_csv = false;
    std::string csv_out = "cusparse_cg_a100.csv";
    
    // 辅助函数：检查字符串是否是选项（以 -- 开头）
    auto is_option = [](const std::string& s) -> bool {
        return s.length() >= 2 && s[0] == '-' && s[1] == '-';
    };
    
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--benchmark") {
            do_benchmark = true;
            // 检查下一个参数是否是数字（不是选项）
            if (i + 1 < argc && !is_option(argv[i + 1])) {
                try {
                    config.warmup = std::stoi(argv[i + 1]);
                } catch (const std::exception& e) {
                    // 如果转换失败，使用默认值
                    std::cerr << "Warning: Invalid warmup value, using default: " << config.warmup << std::endl;
                }
            }
            // 检查下下个参数是否是数字（不是选项）
            if (i + 2 < argc && !is_option(argv[i + 2])) {
                try {
                    config.benchmark_iters = std::stoi(argv[i + 2]);
                } catch (const std::exception& e) {
                    // 如果转换失败，使用默认值
                    std::cerr << "Warning: Invalid benchmark_iters value, using default: " << config.benchmark_iters << std::endl;
                }
            }
        }
        if (std::string(argv[i]) == "--json") {
            output_json = true;
        }
        if (std::string(argv[i]) == "--csv") {
            output_csv = true;
        }
        if (std::string(argv[i]) == "--csv_out") {
            if (i + 1 < argc && !is_option(argv[i + 1])) {
                csv_out = argv[i + 1];
            } else {
                std::cerr << "Warning: --csv_out requires a path, using default: " << csv_out << std::endl;
            }
        }
    }
    
    // 默认保持详细输出；如果要批处理 CSV，默认关闭 verbose，避免污染 stdout
    config.verbose = !output_csv;
    
    // 初始化 b = A * ones 和 x0 = 0
    int m, n;
    std::vector<double> b = compute_b_from_matrix(matrix_file, m, n);
    std::vector<double> x0(n, 0.0);  // x0 是 n 维（矩阵的列数）
    
    // 读取矩阵信息
    int nnz = read_matrix_nnz(matrix_file);
    
    if (!output_csv) {
        std::cout << std::string(60, '=') << std::endl;
        if (do_benchmark) {
            std::cout << "CG Solver Benchmark (cuSPARSE Baseline)" << std::endl;
        } else {
            std::cout << "CG Solver Test (cuSPARSE Baseline)" << std::endl;
        }
        std::cout << std::string(60, '=') << std::endl;
    }
    
    if (do_benchmark) {
        // Warmup
        std::cout << "\nWarmup runs (" << config.warmup << ")..." << std::endl;
        for (int i = 0; i < config.warmup; i++) {
            std::cout << "  Warmup " << (i + 1) << "/" << config.warmup << "\r" << std::flush;
            CGSolverConfig warmup_config = config;
            warmup_config.verbose = false;
            solve_cg_cusparse(matrix_file, warmup_config, &b, &x0);
        }
        std::cout << "  Warmup completed" << std::string(20, ' ') << std::endl;
        
        // Benchmark runs
        std::cout << "\nBenchmark runs (" << config.benchmark_iters << ")..." << std::endl;
        std::vector<CGSolverResult> results;
        
        for (int i = 0; i < config.benchmark_iters; i++) {
            std::cout << "  Run " << (i + 1) << "/" << config.benchmark_iters << "\r" << std::flush;
            CGSolverConfig bench_config = config;
            bench_config.verbose = false;
            results.push_back(solve_cg_cusparse(matrix_file, bench_config, &b, &x0));
        }
        std::cout << "  Benchmark completed" << std::string(20, ' ') << std::endl;
        
        // 计算统计信息
        if (!results.empty()) {
            // 使用第一个结果打印详细信息
            if (output_json) {
                print_json_results(results[0], matrix_file, m, n, nnz);
            } else {
                print_benchmark_results(results[0], matrix_file, m, n, nnz);
            }
            
            // 计算所有运行的平均值
            double avg_spmv_ratio = 0.0;
            double avg_dot_ratio = 0.0;
            double avg_axpy_ratio = 0.0;
            int total_iterations = 0;
            bool all_converged = true;
            
            for (const auto& r : results) {
                avg_spmv_ratio += r.spmv_time_ratio;
                avg_dot_ratio += r.dot_time_ratio;
                avg_axpy_ratio += r.axpy_time_ratio;
                total_iterations += r.iterations;
                if (!r.converged) all_converged = false;
            }
            
            avg_spmv_ratio /= results.size();
            avg_dot_ratio /= results.size();
            avg_axpy_ratio /= results.size();
            
            std::cout << "\nAggregated Results (over " << results.size() << " runs):" << std::endl;
            std::cout << "  Average SpMV time ratio: " << std::fixed << std::setprecision(2) << avg_spmv_ratio << " %" << std::endl;
            std::cout << "  Average Dot product time ratio: " << std::fixed << std::setprecision(2) << avg_dot_ratio << " %" << std::endl;
            std::cout << "  Average Axpy time ratio: " << std::fixed << std::setprecision(2) << avg_axpy_ratio << " %" << std::endl;
            std::cout << "  Average iterations: " << (double)total_iterations / results.size() << std::endl;
            std::cout << "  All converged: " << (all_converged ? "Yes" : "No") << std::endl;
        }
        if (!output_csv) {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "Benchmark completed successfully!" << std::endl;
            std::cout << std::string(60, '=') << std::endl;
        }
    } else {
        // 单次测试或 CSV 模式：CSV 时 warmup 3 + solve 10，输出中间值
        CGSolverResult result;
        std::vector<double> solve_times;

        if (output_csv) {
            // CSV 模式：warmup 3 次，求解 10 次，输出每次耗时
            config.warmup = 3;
            config.benchmark_iters = 10;
            config.verbose = false;

            for (int i = 0; i < config.warmup; i++) {
                solve_cg_cusparse(matrix_file, config, &b, &x0);
            }
            for (int i = 0; i < config.benchmark_iters; i++) {
                result = solve_cg_cusparse(matrix_file, config, &b, &x0);
                solve_times.push_back(result.solve_time_ms);
                std::cout << "  solve[" << i << "]=" << std::fixed << std::setprecision(3) << result.solve_time_ms << " ms" << std::endl;
            }
            double avg_time = std::accumulate(solve_times.begin(), solve_times.end(), 0.0) / solve_times.size();
            std::cout << "avg solve time=" << std::fixed << std::setprecision(3) << avg_time << " ms (over " << config.benchmark_iters << " runs)" << std::endl;

            const double norm = l2_error_to_ones(result.solution);
            std::ostringstream oss;
            oss << "n=" << n
                << ",nnzR=" << nnz
                << ",petsc_norm=" << std::scientific << std::setprecision(6) << norm
                << ", iterations=" << std::dec << result.iterations
                << ", total_time=" << std::fixed << std::setprecision(6) << avg_time
                << ",\n";

            std::ofstream out(csv_out, std::ios::app);
            if (!out.is_open()) {
                std::cerr << "Error: cannot open csv_out: " << csv_out << std::endl;
                return 2;
            }
            out << matrix_file << "," << oss.str();
            out.close();
        } else {
            result = solve_cg_cusparse(matrix_file, config, &b, &x0);
            if (output_json) {
                print_json_results(result, matrix_file, m, n, nnz);
            } else {
                print_benchmark_results(result, matrix_file, m, n, nnz);
            }
        }
        
        if (!output_csv) {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "Test completed successfully!" << std::endl;
            std::cout << std::string(60, '=') << std::endl;
        }
    }
    
    return 0;
}

