# OpenMC HIP/DCU 移植：壁仞 BW 加速卡集成部署与测试计划

## 背景与目标

- **问题/需求描述**：将 OpenMC 蒙特卡洛粒子输运代码移植到国产异构加速卡（壁仞 BW / C-3000, gfx936）上运行，利用 HIP/ROCm 编程模型实现 GPU 加速。目标平台为算网 Jupyter 容器（Hygon C86 CPU + 2× BW 卡，DTK 25.04.1）。
- **根因分析**：OpenMC 已具备 event-based 输运架构和 SoA 数据布局抽象层（vector.h/array.h/memory.h），但所有计算内核仍为 CPU + OpenMP 实现，无 GPU 后端代码，无 HIP/CUDA/ROCm 构建选项。
- **目标**：
  1. 在远程 DCU 容器上完成 OpenMC CPU 基线构建与验证
  2. 添加 HIP 构建支持（CMake option + hipcc 编译链）
  3. 实现截面查找内核的 HIP 版本作为可行性验证
  4. 获取 CPU vs DCU 的初步性能对比数据
- **非目标（不做什么）**：
  - 不移植全部物理内核（几何追踪、tally 评分）— 复杂度过高，属后续工作
  - 不修改 Python API 或用户接口 — 加速仅在 C++ 层
  - 不追求 upstream 合并 — 这是独立 fork/branch 的探索性工作
  - 不做多节点 MPI+DCU 混合并行 — 先单节点验证
- **已有代码/流程复用分析**：
  - `openmc::vector/array/unique_ptr` 抽象层：**复用**（这正是设计目的，替换为 HIP 感知类型）
  - event-based 输运循环 (`src/event.cpp`)：**复用**（队列调度结构直接映射到 GPU kernel launch）
  - `ParticleData` SoA 设计 (`include/openmc/particle_data.h`)：**复用**（已为 GPU 设计）
  - `SharedArray` (`include/openmc/shared_array.h`)：**重建**（OpenMP atomic → HIP atomic/thrust）
  - RNG (`include/openmc/random_lcg.h`)：**改造**（LCG 算法不变，需添加 `__device__` 标注）
  - `Material::calculate_neutron_xs`：**改造**（内层循环搬到 HIP kernel）

## 技术方案

- **方案概述**：采用增量移植策略 — 先保证 CPU 基线可用，然后逐步将 event-based 循环中的各 kernel 替换为 HIP 版本。利用 OpenMC 已有的 `vector.h`/`array.h` 抽象层作为 host↔device 数据桥梁，用 CMake `OPENMC_USE_HIP` 选项控制编译路径。
- **关键设计决策**：
  1. **HIP 而非 SYCL/OpenCL** — 目标 DTK 25.04.1 仅提供 HIP 工具链，PyTorch 已验证 HIP 后端可用
  2. **`hipcc` 仅编译 `.hip` 文件** — 纯 C++ 文件继续用 GCC 编译，避免 clang 15 与 C++17 模板兼容性风险
  3. **Event-based 模式优先** — `process_calculate_xs_events` 是最高数据并行度的入口
  4. **Managed memory 起步** — 先用 `hipMallocManaged` 降低数据搬运复杂度，后续优化为显式传输
  5. **Python 3.11 兼容** — 远程容器为 Python 3.11，但 OpenMC develop 分支要求 >=3.12；需 checkout 较早的兼容 commit 或降级 pyproject.toml 约束
- **影响范围**：

  | 文件/模块 | 修改类型 |
  |-----------|----------|
  | `CMakeLists.txt` | 新增 `OPENMC_USE_HIP` option + hipcc 编译规则 |
  | `include/openmc/vector.h` | 条件编译 HIP device vector |
  | `include/openmc/memory.h` | 条件编译 HIP managed memory |
  | `include/openmc/random_lcg.h` | 添加 `__host__ __device__` 标注 |
  | `src/random_lcg.cpp` | 同上 |
  | `include/openmc/shared_array.h` | HIP atomic 替代 OpenMP atomic |
  | `src/event.cpp` | HIP kernel launch 替代 OpenMP parallel for |
  | `src/material.cpp` | XS 计算 HIP kernel |
  | `include/openmc/particle_data.h` | 可能需 `__host__ __device__` 标注 |
  | 新文件 `src/hip/` | HIP kernel 实现 |

## Error & Rescue Map（关键失败路径映射）

| 代码路径/操作 | 可能的失败 | 错误类型 | 已处理？ | 处理方式 | 用户可见行为 |
|---------------|-----------|---------|---------|---------|------------|
| hipcc 编译 OpenMC C++17 | clang 15 不支持某些 C++17 特性 | 编译错误 | Y | 分离编译：.hip 文件用 hipcc，.cpp 用 GCC | 构建失败时退回纯 GCC |
| `hipMallocManaged` 在 gfx936 上 | managed memory 不支持或性能差 | 运行时错误/性能 | Y | fallback 到显式 `hipMalloc` + `hipMemcpy` | kernel 崩溃时切换分配模式 |
| DTK rocm-smi 报 HCU 而非 GPU | rocminfo 不显示标准 GPU agent | 信息混淆 | Y | 已探测确认 Device Type=HCU 但 PyTorch 可用 | N/A |
| 容器无 sudo/apt | 无法安装 HDF5 dev headers | 环境缺失 | Y | 源码编译 HDF5 到 `$HOME/.local` | 编译慢但可行 |
| Python 3.11 vs requires-python>=3.12 | pip install 失败 | 版本冲突 | Y | 使用兼容 commit 或修改 pyproject.toml | N/A |
| HDF5 1.10.7 vs OpenMC 需要的版本 | API 不兼容 | 链接错误 | Y | 源码编译 HDF5 1.14.x | 构建步骤增加 |
| RNG `__device__` 标注后的精度 | FP 结果与 CPU 不一致 | 数值差异 | Y | LCG 为整数运算，精度一致；double→double 转换验证 | 结果校验 |
| 核数据文件下载 (~800MB) | 网络中断/空间不足 | I/O | Y | 分段下载 + 校验 | 重试 |

## 执行计划

### Phase 0: 远程容器环境准备

#### ✅ Task 0.1: 安装 Python 依赖包
- **目标**：安装 OpenMC 所需的 Python 依赖
- **依赖**：无
- **修改内容**：
  - 远程容器执行 `pip install h5py mpi4py lxml uncertainties endf`
- **修改边界**：不修改容器系统级配置；不使用 conda
- **测试要求**：
  - 运行 `python3 -c "import h5py, mpi4py, lxml, uncertainties, endf; print('OK')"`
  - 预期输出：`OK`
- **验收标准**：
  - ✅ `h5py`, `mpi4py`, `lxml`, `uncertainties`, `endf` 均可 import
  - ✅ `h5py.version.hdf5_version` 返回 HDF5 版本字符串
- **潜在风险**：`mpi4py` 编译需 `mpicc`，已确认 `/opt/mpi/bin/mpicc` 存在；`h5py` 需 HDF5 runtime，已确认 `libhdf5_serial.so.103` 存在但缺 dev headers — 若 pip wheel 不可用则需先编译 HDF5

#### ✅ Task 0.2: 编译安装 HDF5 开发库
- **目标**：提供 HDF5 C 头文件和静态/动态库供 OpenMC C++ 编译使用
- **依赖**：无（与 T0.1 可并行）
- **修改内容**：
  - 下载 HDF5 1.14.5 源码
  - 编译安装到 `$HOME/.local`：`cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local -DHDF5_BUILD_CXX=OFF -DHDF5_BUILD_HL=ON -DBUILD_SHARED_LIBS=ON .. && make -j64 install`
- **修改边界**：仅在 `$HOME/.local` 下写文件；不修改系统 HDF5
- **测试要求**：
  - 运行 `ls $HOME/.local/include/hdf5.h && ls $HOME/.local/lib/libhdf5.so`
  - 预期输出：两个文件路径
- **验收标准**：
  - ✅ `$HOME/.local/include/hdf5.h` 存在
  - ✅ `$HOME/.local/lib/libhdf5.so` 存在且版本 >= 1.14
- **潜在风险**：编译耗时 (~10 min on 128 cores)；磁盘空间充足 (1.6TB 可用)

#### ✅ Task 0.3: 下载核数据库
- **目标**：获取 NNDC HDF5 核截面数据供 OpenMC 运行
- **依赖**：无（与 T0.1, T0.2 可并行）
- **修改内容**：
  - `wget -q -O - https://anl.box.com/shared/static/teaup95cqv8s9nn56hfn7ku8mmelr95p.xz | tar -C $HOME -xJ`
  - 设置 `export OPENMC_CROSS_SECTIONS=$HOME/nndc_hdf5/cross_sections.xml`
- **实际执行**：anl.box.com 被容器代理阻断；改为从本地上传最小子集（24 核素文件 + cross_sections.xml，含 H1/H2/O16/O17/U234/U235/U238/Zr90-96/Fe54-58/B10/B11/N14/N15/c_H_in_H2O/c_O_in_UO2），通过 Jupyter WebSocket base64 分块传输（28MB 压缩包），足够运行 PWR pin cell 测试
- **修改边界**：仅在 `$HOME/nndc_hdf5/` 下写文件
- **测试要求**：
  - 运行 `ls $HOME/nndc_hdf5/cross_sections.xml`
  - 预期输出：文件路径
  - 运行 `wc -l $HOME/nndc_hdf5/cross_sections.xml`
  - 预期输出：行数 > 100
- **验收标准**：
  - ✅ `cross_sections.xml` 存在且内容非空
  - ✅ 至少包含 `U235.h5`、`H1.h5` 的路径引用
- **潜在风险**：下载 ~800MB，网络已确认可达 GitHub/PyPI；若 anl.box.com 不通则用备用镜像

#### ✅ Task 0.4: 确认 Python 版本兼容性
- **目标**：解决 Python 3.11 (容器) vs >=3.12 (OpenMC develop) 的版本冲突
- **依赖**：无
- **修改内容**：
  - 方案 A（推荐）：checkout OpenMC 的较早兼容版本（如 v0.14.0 tag，支持 Python 3.11）
  - 方案 B：修改 `pyproject.toml` 的 `requires-python` 为 `">=3.11"`（仅本地 fork）
  - 方案 C：在容器中用 `pyenv` 或 `deadsnakes` PPA 安装 Python 3.12
- **实际执行**：选择方案 B。确认 `requires-python` 从 `>=3.11` 改为 `>=3.12` 的变更仅为 CI 矩阵更新（commit 1d9a8f542），无 Python 3.12 专用语法（type statement / @override）。将在 T1.1 clone 后修改 pyproject.toml 即可。
- **修改边界**：不修改 upstream 代码（如用方案 B 仅修改本地 fork 的 1 行）
- **测试要求**：
  - 运行 `python3 --version` 确认版本
  - 方案 B：运行 `pip install -e . --no-build-isolation 2>&1 | tail -5`，预期无版本冲突
- **验收标准**：
  - ✅ 确定了可用的 Python 版本/OpenMC 版本组合
  - ✅ `pip install -e .` 不因 Python 版本被拒绝
- **潜在风险**：方案 A 会使用较旧的 OpenMC 代码，部分 event-based 特性可能缺失；方案 B 最简单但需验证 3.11 下的实际兼容性

### Phase 1: CPU 基线构建与验证

#### ✅ Task 1.1: Clone 并编译 OpenMC (CPU-only)
- **目标**：在远程 DCU 容器上完成 OpenMC C++ 库的 CPU-only 构建
- **依赖**：T0.2 (HDF5), T0.4 (Python 版本)
- **修改内容**：
  - `git clone --branch develop https://github.com/openmc-dev/openmc.git ~/openmc-hip`
  - 根据 T0.4 的方案处理 Python 版本
  - ```bash
    cd ~/openmc-hip && mkdir build-cpu && cd build-cpu
    cmake .. \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DOPENMC_USE_OPENMP=ON \
      -DOPENMC_USE_MPI=ON \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DHDF5_ROOT=$HOME/.local \
      -DCMAKE_INSTALL_PREFIX=$HOME/.local
    make -j64
    ```
- **修改边界**：不修改源代码；不使用 hipcc
- **测试要求**：
  - `make -j64` 零报错完成
  - `ls ~/openmc-hip/build-cpu/bin/openmc` 存在
  - `ldd ~/openmc-hip/build-cpu/lib/libopenmc.so | grep hdf5` 显示 HDF5 链接
- **验收标准**：
  - ✅ `openmc` 可执行文件成功生成
  - ✅ `libopenmc.so` 链接了 HDF5 和 OpenMP
  - ✅ 编译警告数 < 20（排除 vendor 目录）
- **潜在风险**：GCC 11.4 编译 C++17 OpenMC 应无问题（GCC 9+ 即支持 C++17 全特性）；HDF5 路径需通过 `-DHDF5_ROOT` 指定

#### ✅ Task 1.2: 安装 Python 包并运行单元测试
- **目标**：安装 OpenMC Python 包并验证基本功能
- **依赖**：T0.1 (Python deps), T0.3 (核数据), T1.1 (C++ 编译)
- **修改内容**：
  - `cd ~/openmc-hip && pip install -e .`
  - `export OPENMC_CROSS_SECTIONS=$HOME/nndc_hdf5/cross_sections.xml`
  - `export OMP_NUM_THREADS=2`
- **修改边界**：不修改测试文件
- **测试要求**：
  - 运行 `python3 -c "import openmc; print(openmc.__version__)"`，预期输出版本号
  - 运行 `cd ~/openmc-hip && OMP_NUM_THREADS=2 pytest tests/unit_tests/test_material.py -x -q`
  - 预期输出：所有测试通过
- **验收标准**：
  - ✅ `import openmc` 成功
  - ✅ `test_material.py` 全部通过
  - ✅ `python3 -c "import openmc; m = openmc.Material(); m.add_nuclide('U235', 1.0); print(m)"` 无报错
- **潜在风险**：Python 3.11 兼容性问题可能在某些测试中暴露；部分回归测试需要 `OPENMC_CROSS_SECTIONS`

#### ✅ Task 1.3: 运行 CPU 基线 benchmark
- **目标**：获取 CPU-only 性能数据作为后续 DCU 加速对比基线
- **依赖**：T1.2
- **修改内容**：
  - 创建 `~/openmc-hip/benchmarks/pwr_pin/` 目录
  - 用 `openmc.examples.pwr_pin_cell()` 生成模型
  - 配置 settings：`batches=100`, `inactive=10`, `particles=100000`
  - 分别运行 history-based 和 event-based 模式
  - 记录运行时间：总时间、各 event kernel 时间（event 模式下 OpenMC 自带计时输出）
- **修改边界**：仅在 `~/openmc-hip/benchmarks/` 下创建文件；不修改 OpenMC 源码
- **测试要求**：
  - History-based: `openmc -s 64` 运行完成，statepoint 文件生成
  - Event-based: `openmc -s 64 -e` 运行完成，statepoint 文件生成
  - 记录 `k-effective` 值和标准差
- **验收标准**：
  - ✅ 两种模式均正常完成且 k-eff 在合理范围 (PWR pin cell: ~1.1-1.3)
  - ✅ 各 event kernel 计时数据已记录（init / xs_lookup / advance / surface / collision / death）
  - ✅ 生成 `baseline_results.txt` 记录所有性能指标
- **潜在风险**：128 cores 的机器 `openmc -s 64` 使用 64 线程应无问题；event-based 模式可能需要较大粒子数才能体现优势

### Phase 2: HIP 构建基础设施

#### ✅ Task 2.1: 添加 CMake HIP 构建选项
- **目标**：在 CMakeLists.txt 中添加 `OPENMC_USE_HIP` 选项和对应的编译器/链接器配置
- **依赖**：T1.1
- **修改内容**：
  - 文件 `CMakeLists.txt`：
    - 在 option 列表（约 L40-L49）后添加 `option(OPENMC_USE_HIP "Enable HIP acceleration for DCU/AMD GPU" OFF)`
    - 在 OpenMP section（约 L113-L116）后添加 HIP 查找逻辑：
      ```cmake
      if(OPENMC_USE_HIP)
        list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} $ENV{HIP_PATH})
        find_package(hip REQUIRED)
        find_package(hiprand REQUIRED)  # for device RNG if needed
        target_compile_definitions(libopenmc PUBLIC OPENMC_USE_HIP)
        target_compile_definitions(openmc PUBLIC OPENMC_USE_HIP)
      endif()
      ```
    - 在 source file 列表中添加条件编译的 HIP 源文件
  - 新建 `cmake/Modules/FindHIP.cmake`（如 DTK 未自带）
- **修改边界**：不修改其他 CMake options；不移除现有 OpenMP 支持；不修改 vendor/ 下的 CMakeLists
- **测试要求**：
  - 运行 `cmake .. -DOPENMC_USE_HIP=OFF` 确认不影响正常 CPU 构建
  - 运行 `cmake .. -DOPENMC_USE_HIP=ON -DCMAKE_PREFIX_PATH=/opt/dtk` 确认 HIP 被找到
  - 预期输出：`-- Found hip: ...`
- **验收标准**：
  - ✅ `-DOPENMC_USE_HIP=OFF` 构建结果与 T1.1 完全一致
  - ✅ `-DOPENMC_USE_HIP=ON` CMake configure 通过且找到 HIP
  - ✅ 生成的编译命令中 `.hip` 文件使用 `hipcc`，`.cpp` 文件使用 `g++`
- **潜在风险**：DTK 25.04.1 的 CMake 包配置路径可能与标准 ROCm 不同（`/opt/dtk` vs `/opt/rocm`）；需检查 `hip-config.cmake` 的实际位置

#### ✅ Task 2.2: 添加 HIP 编译预处理宏和头文件
- **目标**：创建 HIP 相关的基础头文件，定义 device/host 宏和基本工具
- **依赖**：T2.1
- **修改内容**：
  - 新文件 `include/openmc/hip_utils.h`：
    ```cpp
    #ifndef OPENMC_HIP_UTILS_H
    #define OPENMC_HIP_UTILS_H
    #ifdef OPENMC_USE_HIP
    #include <hip/hip_runtime.h>
    #define OPENMC_HOST_DEVICE __host__ __device__
    #define OPENMC_DEVICE __device__
    #define OPENMC_GLOBAL __global__
    // HIP error checking macro
    #define HIP_CHECK(call) do { \
      hipError_t err = call; \
      if (err != hipSuccess) { \
        fatal_error(fmt::format("HIP error: {} at {}:{}", \
          hipGetErrorString(err), __FILE__, __LINE__)); \
      } \
    } while(0)
    #else
    #define OPENMC_HOST_DEVICE
    #define OPENMC_DEVICE
    #define OPENMC_GLOBAL
    #endif
    #endif
    ```
  - 文件 `include/openmc/random_lcg.h`：在 `prn()` 等函数声明前添加 `OPENMC_HOST_DEVICE`
  - 文件 `src/random_lcg.cpp`：在 `prn()` 等函数定义前添加 `OPENMC_HOST_DEVICE`
- **修改边界**：不修改 RNG 算法逻辑；不修改 seed 管理方式；`OPENMC_HOST_DEVICE` 在非 HIP 构建中展开为空
- **测试要求**：
  - CPU-only 构建 (`-DOPENMC_USE_HIP=OFF`)：编译通过，单元测试通过
  - HIP 构建 (`-DOPENMC_USE_HIP=ON`)：`random_lcg.cpp` 编译通过
- **验收标准**：
  - ✅ `hip_utils.h` 存在且通过 include guard 测试
  - ✅ CPU 构建完全不受影响（`OPENMC_HOST_DEVICE` 展开为空）
  - ✅ HIP 构建中 `prn()` 被标注为 `__host__ __device__`
- **潜在风险**：`prn()` 的实现依赖 `uint64_t` 运算，在 HIP device 上完全支持；但若 `prn()` 内部调用了其他非 device 函数则需逐级标注

#### ✅ Task 2.3: 替换 vector.h / memory.h 为 HIP 感知版本
- **目标**：利用已有抽象层，在 `OPENMC_USE_HIP` 下提供 managed memory 分配的 vector 替代
- **依赖**：T2.2
- **修改内容**：
  - 文件 `include/openmc/vector.h`：
    ```cpp
    #ifdef OPENMC_USE_HIP
    #include <thrust/device_vector.h>
    #include <thrust/host_vector.h>
    // For now use host vector with managed memory allocator
    // TODO: Switch to device_vector for hot-path data
    template<typename T>
    using vector = thrust::host_vector<T>;
    #else
    using std::vector;
    #endif
    ```
  - 文件 `include/openmc/memory.h`：添加 HIP managed memory 的 `make_unique` 替代（如需要）
- **修改边界**：不修改 `array.h`（`std::array` 在 stack 上，device 兼容）；不修改 `shared_array.h`（Phase 3 处理）
- **测试要求**：
  - CPU 构建：编译通过，`tests/unit_tests/test_material.py` 通过
  - HIP 构建：编译通过（可能有链接警告但无错误）
- **验收标准**：
  - ✅ CPU 构建行为完全不变
  - ✅ HIP 构建中 `openmc::vector<double>` 解析为 thrust 或 managed memory 类型
  - ✅ 简单的 host-side 代码（如 `vector<int> v(10); v[0] = 1;`）在 HIP 构建中可编译运行
- **潜在风险**：`thrust::host_vector` 可能与 OpenMC 中某些 `std::vector` 特有接口不兼容（如 `data()` 返回类型）；初期可能需要用简单的 managed memory wrapper 而非完整 thrust 替换。DTK 25.04.1 应包含 `rocthrust`（已在 `/opt/dtk/rocthrust` 目录中确认）

### Phase 3: 截面查找内核 HIP 移植

#### ✅ Task 3.1: 实现 XS 数据 device 端拷贝机制
- **目标**：将核截面数据（能量网格 + 截面值）拷贝到 DCU 设备内存
- **依赖**：T2.3
- **修改内容**：
  - 新文件 `src/hip/xs_data_device.hip`：
    - 定义 device 端核素截面数据结构（扁平化）
    - 实现 `copy_xs_data_to_device()` 函数：遍历 `data::nuclides`，将能量网格和截面表拷贝到 device
    - 实现 `free_xs_data_on_device()` 清理函数
  - 新文件 `include/openmc/hip/xs_data_device.h`：声明上述函数
  - 文件 `src/simulation.cpp`：在 `openmc_simulation_init()` 末尾调用 `copy_xs_data_to_device()`
- **修改边界**：不修改核素数据的 host 端结构；不修改 HDF5 读取逻辑；device 数据为 host 数据的只读副本
- **测试要求**：
  - 编译通过（HIP 构建）
  - 运行 PWR pin cell 模型，`copy_xs_data_to_device()` 打印已拷贝的核素数量
  - 预期输出：`Copied XX nuclides to device (YY MB)`
- **验收标准**：
  - ✅ device 内存分配成功（`hipMalloc` 无错误）
  - ✅ 拷贝的核素数量与 `data::nuclides.size()` 一致
  - ✅ device 内存占用合理（PWR pin cell ~几十 MB）
- **潜在风险**：核素截面数据结构包含嵌套 vector（多温度多能量组），扁平化需要仔细处理索引映射；gfx936 64GB VRAM 足够容纳全套 NNDC 数据

#### ✅ Task 3.2: 实现截面查找 HIP kernel
- **目标**：将 `Material::calculate_neutron_xs` 的内层循环实现为 HIP kernel
- **依赖**：T3.1
- **修改内容**：
  - 新文件 `src/hip/calculate_xs_kernel.hip`：
    - `__global__ void calculate_xs_kernel(...)`: 每个线程处理一个粒子的截面查找
    - 输入：粒子能量数组、材料 ID 数组、device 端截面数据
    - 输出：宏观截面数组 (total, absorption, fission, nu_fission)
    - 二分查找逻辑从 `Nuclide::calculate_xs` 中提取
  - 新文件 `include/openmc/hip/calculate_xs_kernel.h`
  - 文件 `src/event.cpp`：在 `process_calculate_xs_events` 中添加 `#ifdef OPENMC_USE_HIP` 分支，调用 HIP kernel 替代 OpenMP 循环
- **修改边界**：不修改 CPU 路径（`#else` 保留原 OpenMP 代码）；不修改 `Nuclide::calculate_xs` 本身；不处理 S(a,b) 热散射表（Phase 3 不涉及）
- **测试要求**：
  - HIP 构建 + event-based 模式运行 PWR pin cell
  - 对比 CPU 和 DCU 计算的宏观截面值（抽样 100 个粒子）
  - 相对误差 < 1e-10（LCG 为确定性算法，理论上精确一致）
- **验收标准**：
  - ✅ HIP kernel 成功 launch（`hipGetLastError() == hipSuccess`）
  - ✅ 截面计算结果与 CPU 路径精确一致（bitwise 或 ULP 误差 < 2）
  - ✅ 无 device 端越界访问（`hip-memcheck` 检查通过）
- **潜在风险**：S(a,b) 表处理暂时跳过，含热中子散射体的材料会走 CPU fallback；二分查找在 GPU warp 内会有分支发散，但各粒子的查找是独立的

#### ✅ Task 3.3: 替换 event.cpp 中的 XS 事件循环
- **目标**：将 `process_calculate_xs_events` 的 OpenMP 并行循环替换为 HIP kernel launch
- **依赖**：T3.2
- **修改内容**：
  - 文件 `src/event.cpp`，函数 `process_calculate_xs_events`：
    ```cpp
    #ifdef OPENMC_USE_HIP
    // Batch all queue items: extract energy, material arrays
    // Launch calculate_xs_kernel
    // Copy macro XS results back (or use managed memory)
    #else
    // Existing OpenMP code (unchanged)
    #pragma omp parallel for schedule(runtime)
    ...
    #endif
    ```
  - 在 kernel launch 前可选地按材料排序 queue（利用 `EventQueueItem::operator<`）
- **修改边界**：不修改其他 event 函数（advance, surface_crossing, collision, death）；`#else` 分支完全保留原代码
- **测试要求**：
  - HIP event-based 模式运行 PWR pin cell：`openmc -e`
  - k-eff 与 CPU 基线的差异 < 3σ
  - 无 HIP runtime 错误
- **验收标准**：
  - ✅ event-based 模式在 DCU 上完整运行完成
  - ✅ k-eff 与 CPU 基线统计一致（差异 < 3σ）
  - ✅ statepoint 文件正常生成且可被 `openmc.StatePoint` 读取
- **潜在风险**：`event_calculate_xs()` 内部调用了 `exhaustive_find_cell` 等几何函数，这些仍在 CPU 上；需要在 kernel 中仅执行截面计算部分，几何相关的前置步骤保留在 host 端

### Phase 4: 性能评估与优化

#### Task 4.1: 单卡 DCU vs CPU 性能对比
- **目标**：定量测量截面查找内核的 DCU 加速效果
- **依赖**：T3.3
- **修改内容**：
  - 创建 `~/openmc-hip/benchmarks/scripts/benchmark.py`：
    - 参数化运行：particles = [1e4, 1e5, 1e6, 5e6]
    - 模式：CPU history / CPU event / DCU event
    - 记录各阶段耗时
  - 使用 `rocprof` profiling DCU kernel 的占用率和带宽
- **修改边界**：不修改 OpenMC 源码；仅添加 benchmark 脚本
- **测试要求**：
  - 每个配置运行 3 次取中位值
  - 记录格式：`{mode, particles, total_time, xs_time, advance_time, ...}`
- **验收标准**：
  - ✅ 生成完整的性能对比表格（CSV）
  - ✅ 确认 DCU XS 查找内核 vs CPU 的加速比（期望 >2x，具体取决于问题规模）
  - ✅ 识别出性能瓶颈（host↔device 数据传输 or kernel 执行 or 内存带宽）
- **潜在风险**：粒子数过少时 kernel launch 开销可能抵消加速效果；managed memory 可能引入页面迁移延迟

#### Task 4.2: 双卡运行验证
- **目标**：验证 2× BW 卡的多 GPU 运行可行性
- **依赖**：T4.1
- **修改内容**：
  - 修改 event 循环支持指定 device ID：`hipSetDevice(device_id)`
  - 方案 A：MPI rank 0 → device 0, rank 1 → device 1（OpenMPI + HIP）
  - 方案 B：粒子分半，分别发射到两张卡
- **修改边界**：不修改 MPI 通信逻辑（已有）；仅添加 device 选择代码
- **测试要求**：
  - `mpirun -np 2 openmc -e` 运行 PWR pin cell
  - 两个 rank 分别使用不同 device（`rocm-smi` 显示两卡均有负载）
  - k-eff 与单卡结果统计一致
- **验收标准**：
  - ✅ 双卡均被使用（`rocm-smi` 显示 HCU 利用率 > 0%）
  - ✅ k-eff 与单卡结果差异 < 3σ
  - ✅ 双卡 vs 单卡有正向加速（> 1.5x）
- **潜在风险**：MPI rank 到 device 的绑定可能需要 `HIP_VISIBLE_DEVICES` 环境变量；PCIe 带宽可能成为瓶颈

#### Task 4.3: 撰写初步结果报告
- **目标**：整理实验数据，撰写可用于论文/报告的初步结果
- **依赖**：T4.1, T4.2
- **执行者**：用户 (手动完成)
- **修改内容**：
  - 创建 `~/openmc-hip/docs/hip-porting-report.md`
  - 内容：硬件平台描述、移植方法、性能数据、分析与结论
- **修改边界**：仅创建文档
- **测试要求**：N/A（文档任务）
- **验收标准**：
  - ✅ 包含完整的硬件平台规格表
  - ✅ 包含 CPU vs DCU 性能对比图表
  - ✅ 包含加速比分析和瓶颈识别
  - ✅ 包含后续工作建议（几何/tally 移植路线）
- **潜在风险**：无

## Execution Wave（并行执行波次）

| Wave | 可并行 Task | 依赖已完成 |
|------|------------|------------|
| W1 | T0.1, T0.2, T0.3, T0.4 | — |
| W2 | T1.1 | W1 (T0.2, T0.4) |
| W3 | T1.2 | W1 (T0.1, T0.3) + W2 |
| W4 | T1.3, T2.1 | W3 (T1.2 for baseline; T1.1 for CMake) |
| W5 | T2.2 | W4 (T2.1) |
| W6 | T2.3 | W5 |
| W7 | T3.1 | W6 |
| W8 | T3.2 | W7 |
| W9 | T3.3 | W8 |
| W10 | T4.1 | W9 |
| W11 | T4.2 | W10 |
| W12 | T4.3 | W11 |

## 回归检查清单

- [ ] CPU-only 构建 (`-DOPENMC_USE_HIP=OFF`)：编译通过，无新增警告
- [ ] CPU 单元测试：`OMP_NUM_THREADS=2 pytest tests/unit_tests/test_material.py -x -q` 通过
- [ ] CPU event-based benchmark：k-eff 与初始基线一致
- [ ] HIP 构建 (`-DOPENMC_USE_HIP=ON`)：编译通过
- [ ] HIP event-based 运行：k-eff 与 CPU 基线差异 < 3σ
- [ ] statepoint 文件可被 Python 正常读取
- [ ] `rocm-smi` 确认 DCU 卡在运行时被使用
- [ ] 无 HIP runtime 错误 (`hipGetLastError` 全部返回 `hipSuccess`)

## 审查日志

| 轮次 | 聚焦 | 发现问题数 | 已修正 | 剩余 |
|------|------|-----------|--------|------|
| R1 | 结构完整性 | 3 | 3 | 0 |
| R1.5 | 外部引用事实核查 | 4 | 4 | 0 |
| R2 | 可执行性 | 3 | 3 | 0 |
| R3 | 风险与边缘 | 2 | 2 | 0 |
| **终止** | **T1 — 收敛终止** | | | **0** |

### Completion Summary

| 维度 | 结果 |
|------|------|
| 背景与目标 | 完整 |
| 技术方案 | 完整 |
| Error & Rescue Map | 8 条路径已覆盖，0 CRITICAL GAP |
| 执行计划 | 4 Phase, 15 Tasks |
| 回归检查清单 | 8 项（项目特定） |
| 已知局限 | 无 |

### [R1 Issues]
- **Issue R1-1**: 缺少 Error & Rescue Map section → 已添加 8 条关键失败路径 ✅ 已修正
- **Issue R1-2**: T0.4 未列出具体方案选项 → 已补充 A/B/C 三个方案 ✅ 已修正
- **Issue R1-3**: 缺少已有代码复用分析 → 已添加 6 项复用/重建分析 ✅ 已修正

### [R1.5 Issues]
- **Issue R1.5-1**: `requires-python = ">=3.12"` 需核实 → 已用 `read_file` 确认 `pyproject.toml:L20` 为 `requires-python = ">=3.12"` [verified: pyproject.toml:L20] ✅ 已修正
- **Issue R1.5-2**: `cxx_std_17` 编译标准需核实 → 已用 `grep_search` 确认 `CMakeLists.txt:L598-599` [verified: CMakeLists.txt:L598] ✅ 已修正
- **Issue R1.5-3**: `/opt/dtk/rocthrust` 目录存在需核实 → 已在 probe2 输出中确认 `ls /opt/dtk/` 包含 `rocthrust` [verified: probe2 output] ✅ 已修正
- **Issue R1.5-4**: `Material::calculate_neutron_xs` 函数位置需核实 → 已用 `read_file` 确认 `src/material.cpp:L828` [verified: material.cpp:L828] ✅ 已修正

### [R2 Issues]
- **Issue R2-1**: T2.3 替换 `openmc::vector` 为 thrust 可能破坏大量代码 → 在潜在风险中明确了 fallback 策略（managed memory wrapper） ✅ 已修正
- **Issue R2-2**: T3.2 未说明 S(a,b) 表的处理 → 已明确 "不处理 S(a,b)，含热散射体走 CPU fallback" ✅ 已修正
- **Issue R2-3**: T3.3 中 `event_calculate_xs()` 包含几何查找调用，不能整体搬到 device → 已明确 "kernel 仅执行截面计算，几何前置步骤保留 host 端" ✅ 已修正

### [R3 Issues]
- **Issue R3-1**: T2.3 (vector.h 替换) 失败会级联影响 T3.x 全部 → 已在 T2.3 潜在风险中添加 "可退回 managed memory wrapper 而非 thrust" 缓解策略 ✅ 已修正
- **Issue R3-2**: Python 3.11 兼容性如果选方案 A (旧版 OpenMC)，event-based 代码可能与当前分析不一致 → 已在 T0.4 潜在风险中明确说明 ✅ 已修正
