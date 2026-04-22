// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// Ginkgo CG benchmark: reads an SPD matrix in MatrixMarket format (.mtx),
// solves Ax=b on the CUDA executor, and appends timing results to a CSV.
//
// Usage:
//   simple-solver <matrix.mtx> [max_iter] [tol] [--csv] [--csv_out file.csv]
//
// CSV output format (no header):
//   <path>,<rows>,<nnz>,<time_ms>,<iters>

#include <ginkgo/ginkgo.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>


int main(int argc, char* argv[])
{
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;

    // Parse arguments
    if (argc < 2 || std::string(argv[1]) == "--help") {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix.mtx> [max_iter] [tol] [--csv] [--csv_out file.csv]\n";
        return 1;
    }

    const std::string matrix_path = argv[1];
    int max_iter = 10000;
    double tol = 1e-10;
    bool do_csv = false;
    std::string csv_out = "ginkgo_cg_a100.csv";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv") {
            do_csv = true;
        } else if (arg == "--csv_out" && i + 1 < argc) {
            csv_out = argv[++i];
            do_csv = true;
        } else if (i == 2) {
            max_iter = std::atoi(argv[i]);
        } else if (i == 3) {
            tol = std::atof(argv[i]);
        }
    }

    // Create CUDA executor
    auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());

    // Read matrix
    std::ifstream mtx_file(matrix_path);
    if (!mtx_file.is_open()) {
        std::cerr << "ERROR: cannot open " << matrix_path << "\n";
        return 1;
    }
    auto A = gko::share(gko::read<mtx>(mtx_file, exec));
    const auto rows = A->get_size()[0];
    const auto nnz = A->get_num_stored_elements();

    // Create x = all-zeros on host then transfer
    auto cpu_exec = gko::OmpExecutor::create();
    auto ones_cpu = gko::matrix::Dense<ValueType>::create(
        cpu_exec, gko::dim<2>{rows, 1});
    auto x_cpu = gko::matrix::Dense<ValueType>::create(
        cpu_exec, gko::dim<2>{rows, 1});
    for (gko::size_type i = 0; i < rows; ++i) {
        ones_cpu->at(i, 0) = 1.0;
        x_cpu->at(i, 0) = 0.0;
    }
    auto ones = gko::clone(exec, ones_cpu);
    auto b = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{rows, 1});
    A->apply(ones, b);

    // Build CG solver
    const RealValueType reduction_factor{static_cast<RealValueType>(tol)};
    auto solver_gen =
        cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(
                    static_cast<unsigned>(max_iter)),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(reduction_factor))
            .on(exec);
    auto solver = solver_gen->generate(A);

    // Warm-up
    auto x_warm = gko::clone(exec, x_cpu);
    solver->apply(b, x_warm);
    exec->synchronize();

    // Timed solve — attach a fresh logger to capture iteration count
    auto logger = gko::share(gko::log::Convergence<ValueType>::create());
    solver->add_logger(logger);
    auto x_sol = gko::clone(exec, x_cpu);
    auto t0 = std::chrono::high_resolution_clock::now();
    solver->apply(b, x_sol);
    exec->synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    gko::size_type iters = logger->get_num_iterations();

    std::cout << matrix_path << ", rows=" << rows << ", nnz=" << nnz
              << ", time_ms=" << time_ms << ", iters=" << iters << "\n";

    if (do_csv) {
        std::ofstream csv(csv_out, std::ios::app);
        csv << matrix_path << "," << rows << "," << nnz << "," << time_ms
            << "," << iters << "\n";
    }

    return 0;
}
