# Makefile for cuSPARSE Baseline CG Solver

NVCC = nvcc
CXX = g++

# 编译选项
# 支持多个 GPU 架构：sm_70 (V100), sm_75 (RTX 20xx), sm_80 (A100), sm_86 (RTX 30xx)
NVCC_FLAGS = -O3 -std=c++14 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -rdc=true \
    --use_fast_math \
    --allow-unsupported-compiler
CXX_FLAGS = -O3 -std=c++17

# 包含目录
INCLUDES = -I. -I.. -I../utils

# 源文件
CUSPARSE_CG_SOLVER_CU = cusparse_cg_solver.cu
CUSPARSE_CG_SOLVER_CPP = cusparse_cg_solver.cpp
TEST_CPP = test_cusparse_cg.cpp

# 目标文件
CUSPARSE_CG_SOLVER_O = cusparse_cg_solver.o
CUSPARSE_CG_SOLVER_CPP_O = cusparse_cg_solver_cpp.o
TEST_O = test_cusparse_cg.o

# 可执行文件
TARGET = test_cusparse_cg

# 默认目标
all: $(TARGET)

# 链接所有目标文件（需要链接 cuSPARSE 和 cuBLAS）
$(TARGET): $(CUSPARSE_CG_SOLVER_O) $(CUSPARSE_CG_SOLVER_CPP_O) $(TEST_O)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $^ -lcudart -lcusparse -lcublas

# 编译 CUDA 源文件
$(CUSPARSE_CG_SOLVER_O): $(CUSPARSE_CG_SOLVER_CU) cusparse_cg_solver.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

# 编译 C++ 源文件
$(CUSPARSE_CG_SOLVER_CPP_O): $(CUSPARSE_CG_SOLVER_CPP) cusparse_cg_solver.h
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c -o $@ $<

$(TEST_O): $(TEST_CPP) cusparse_cg_solver.h
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c -o $@ $<

# 清理
clean:
	rm -f $(TARGET) *.o

# 运行测试
test: $(TARGET)
	./$(TARGET) ../bodyy4.mtx 1000 1e-10

# 运行基准测试（默认 warmup 3, iter 10）
benchmark: $(TARGET)
	./$(TARGET) ../bodyy4.mtx 1000 1e-10 --benchmark 3 10

.PHONY: all clean test benchmark

