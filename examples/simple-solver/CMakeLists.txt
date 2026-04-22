cmake_minimum_required(VERSION 3.16)
project(simple-solver)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.12.0 REQUIRED)
endif()

add_executable(simple-solver simple-solver.cpp)
target_link_libraries(simple-solver Ginkgo::ginkgo)
# Allow unresolved batch-solver symbols in shared libs (batch kernels are
# intentionally excluded from the CUDA build to work around nvcc 11.5 + GCC 11
# std::function template-pack issue; they are not used by simple-solver).
target_link_options(simple-solver PRIVATE -Wl,--allow-shlib-undefined)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)
