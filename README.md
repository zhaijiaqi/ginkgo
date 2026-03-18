# Ginkgo 线性代数库

[Ginkgo](https://github.com/ginkgo-project/ginkgo) 是一个高性能数值线性代数库，专注于稀疏线性系统求解。

## 安装信息

- **版本**: v1.11.0
- **安装路径**: `baselines/ginkgo/install/`
- **已启用后端**: Reference、OpenMP、MPI、**CUDA 12.4**

## 使用方法

### CMake 集成

在项目的 `CMakeLists.txt` 中：

```cmake
set(Ginkgo_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/baselines/ginkgo/install")
find_package(Ginkgo REQUIRED)
target_link_libraries(your_target Ginkgo::ginkgo)
```

### 编译选项

```bash
# 编译时指定 include 和 lib 路径
g++ -I baselines/ginkgo/install/include -L baselines/ginkgo/install/lib -lginkgo your_code.cpp -o your_app
```

### 运行时

确保库路径在 `LD_LIBRARY_PATH` 中：

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/baselines/ginkgo/install/lib
```

## 目录结构

```
baselines/ginkgo/
├── install/          # 安装目录（头文件、库、CMake 配置）
│   ├── include/     # 头文件
│   ├── lib/         # 共享库
│   └── lib/cmake/   # CMake 配置文件
├── build/           # 构建目录
└── README.md        # 本文件
```
