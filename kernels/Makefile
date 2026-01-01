# Makefile for cuSPARSE SpMV benchmark

CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 -Wall
NVCCFLAGS = -std=c++17 -O3 -arch=sm_70 -arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_89 -arch=sm_90

# CUDA paths (adjust if needed)
CUDA_PATH ?= /usr/local/cuda
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64

# Libraries
LIBS = -lcusparse -lcudart -lcublas

# Targets
TARGET = test_cusparse_spmv
SRC = test_cusparse_spmv.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(CUDA_INC) -o $(TARGET) $(SRC) $(CUDA_LIB) $(LIBS)

clean:
	rm -f $(TARGET)

# Example usage:
# make
# ./test_cusparse_spmv ~/data/matrix/matrix_name.mtx 10 100

