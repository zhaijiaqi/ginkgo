// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the fstream header to read from data from files.
#include <fstream>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <string>
#include <cstring>
#include <sys/time.h>
int main(int argc, char* argv[])
{
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row. The advantage of this concept is that using
    // multiple vectors is a now a natural extension of adding columns/rows are
    // necessary.
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using hp_vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    // The gko::matrix::Csr class is used here, but any other matrix class such
    // as gko::matrix::Coo, gko::matrix::Hybrid, gko::matrix::Ell or
    // gko::matrix::Sellp could also be used.
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    // The gko::solver::Cg is used here, but any other solver class can also be
    // used.
    using cg = gko::solver::Cg<ValueType>;
    using bicg = gko::solver::Bicgstab<ValueType>;
    using gmres = gko::solver::Gmres<ValueType>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    // Print help on how to execute this example.
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }
    char *file_name=argv[2];
    printf("%s\n",file_name);
    int flag = atoi(argv[3]); //flag 1 cg flag 2 bicg
    // @sect3{Where do you want to run your solver ?}
    // The gko::Executor class is one of the cornerstones of Ginkgo. Currently,
    // we have support for
    // an gko::OmpExecutor, which uses OpenMP multi-threading in most of its
    // kernels, a gko::ReferenceExecutor, a single threaded specialization of
    // the OpenMP executor and a gko::CudaExecutor which runs the code on a
    // NVIDIA GPU if available.
    // @note With the help of C++, you see that you only ever need to change the
    // executor and all the other functions/ routines within Ginkgo should
    // automatically work and run on the executor with any other changes.
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid
    const auto exec1 = exec_map.at("reference")(); 
    // @sect3{Reading your data and transfer to the proper device.}
    // Read the matrix, right hand side and the initial solution using the @ref
    // read function.
    // @note Ginkgo uses C++ smart pointers to automatically manage memory. To
    // this end, we use our own object ownership transfer functions that under
    // the hood call the required smart pointer functions to manage object
    // ownership. gko::share and gko::give are the functions that you would need
    // to use.
    auto A = gko::share(gko::read<mtx>(std::ifstream(file_name), exec));
    auto A1 = share(gko::clone(exec1, A));
    auto A_dim = A1->get_size();
    auto row_ptr = A1->get_row_ptrs();
    auto col_idx = A1->get_col_idxs();
    auto val = A1->get_values();
    auto b_dim = gko::dim<2>{A_dim[0], 1};
    auto host_b = hp_vec::create(exec->get_master(), b_dim);
    // for (int i = 0; i < host_b->get_size()[0]; i++) {
    //     host_b->at(i, 0) = 1;
    // }
    auto x_dim = gko::dim<2>{A_dim[0], 1};
    auto host_x = hp_vec::create(exec->get_master(), x_dim);
    for (int i = 0; i < host_x->get_size()[0]; i++) {
        host_x->at(i, 0) = 1;
    }
    for(int i=0;i<A_dim[0];i++)
    {
        double sum=0;
        for(int j=row_ptr[i];j<row_ptr[i+1];j++)
        {
            int col=col_idx[j];
            sum+=1*val[j];
            //printf("%d %lf\n",col_idx[j],val[j]);
        }
        //printf("%lf\n",sum);
        host_b->at(i, 0)=sum;
    }
    for (int i = 0; i < host_x->get_size()[0]; i++) {
        host_x->at(i, 0) = 0;
    }
    // auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    // auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);
    auto b = share(gko::clone(exec, host_b));
    auto x = share(gko::clone(exec, host_x));
    //printf("%d %d\n",A_dim[0],A_dim[1]);
    // @sect3{Creating the solver}
    // Generate the gko::solver factory. Ginkgo uses the concept of Factories to
    // build solvers with certain
    // properties. Observe the Fluent interface used here. Here a cg solver is
    // generated with a stopping criteria of maximum iterations of 20 and a
    // residual norm reduction of 1e-7. You also observe that the stopping
    // criteria(gko::stop) are also generated from factories using their build
    // methods. You need to specify the executors which each of the object needs
    // to be built on.
    const RealValueType reduction_factor{1e-10};
    if(flag==2)
    {
        constexpr int WARMUP_CNT = 3;
        constexpr int SOLVE_CNT = 10;
        auto solver_gen =
        bicg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(10000u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
            .on(exec);
        auto solver = solver_gen->generate(A);

        /* Warmup */
        for (int k = 0; k < WARMUP_CNT; k++) {
            x->copy_from(host_x);
            solver->apply(b, x);
        }

        /* Timed solve runs */
        struct timeval t_start, t_stop;
        double solve_times[SOLVE_CNT];
        for (int k = 0; k < SOLVE_CNT; k++) {
            x->copy_from(host_x);
            gettimeofday(&t_start, NULL);
            solver->apply(b, x);
            gettimeofday(&t_stop, NULL);
            solve_times[k] = (t_stop.tv_sec - t_start.tv_sec) * 1000.0 + (t_stop.tv_usec - t_start.tv_usec) / 1000.0;
            printf("  solve[%d]=%.3f ms\n", k, solve_times[k]);
        }

        double total_time = 0.0;
        for (int k = 0; k < SOLVE_CNT; k++) total_time += solve_times[k];
        total_time /= SOLVE_CNT;
        printf("avg solve time=%.3f ms (over %d runs)\n", total_time, SOLVE_CNT);
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);
    char *s = (char *)malloc(sizeof(char) * 100);
    sprintf(s, "%zu, %d, %lf\n",static_cast<size_t>(A_dim[0]),row_ptr[A_dim[0]],total_time);
    FILE *file1 = fopen("ginkgo_bicg_a100.csv", "a");
    if (file1 == NULL)
    {
        printf("open error!\n");
        return 0;
    }
    fwrite(file_name, strlen(file_name), 1, file1);
    fwrite(",", strlen(","), 1, file1);
    fwrite(s, strlen(s), 1, file1);
    free(s);
    fclose(file1);
    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, res);
    }
    else if(flag==1)
    {
        constexpr int WARMUP_CNT = 3;
        constexpr int SOLVE_CNT = 10;
        auto solver_gen =
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(10000u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
            .on(exec);
        auto solver = solver_gen->generate(A);

        /* Warmup */
        for (int k = 0; k < WARMUP_CNT; k++) {
            x->copy_from(host_x);
            solver->apply(b, x);
        }

        /* Timed solve runs */
        struct timeval t_start, t_stop;
        double solve_times[SOLVE_CNT];
        for (int k = 0; k < SOLVE_CNT; k++) {
            x->copy_from(host_x);
            gettimeofday(&t_start, NULL);
            solver->apply(b, x);
            gettimeofday(&t_stop, NULL);
            solve_times[k] = (t_stop.tv_sec - t_start.tv_sec) * 1000.0 + (t_stop.tv_usec - t_start.tv_usec) / 1000.0;
            printf("  solve[%d]=%.3f ms\n", k, solve_times[k]);
        }

        double total_time = 0.0;
        for (int k = 0; k < SOLVE_CNT; k++) total_time += solve_times[k];
        total_time /= SOLVE_CNT;
        printf("avg solve time=%.3f ms (over %d runs)\n", total_time, SOLVE_CNT);

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);
    char *s = (char *)malloc(sizeof(char) * 100);
    sprintf(s, "%zu, %d, %lf\n",static_cast<size_t>(A_dim[0]),row_ptr[A_dim[0]],total_time);
    FILE *file1 = fopen("ginkgo_cg_a100.csv", "a");
    if (file1 == NULL)
    {
        printf("open error!\n");
        return 0;
    }
    fwrite(file_name, strlen(file_name), 1, file1);
    fwrite(",", strlen(","), 1, file1);
    fwrite(s, strlen(s), 1, file1);
    free(s);
    fclose(file1);
    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, res);
    }
}
