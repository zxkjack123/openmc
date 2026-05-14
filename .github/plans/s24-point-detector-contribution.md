# S24: Point Detector / Next-Event Estimator — Contribution Plan

## 背景与目标

- **问题/需求描述**：OpenMC 的 Point Detector / Next-Event Estimator (NEE) 功能正通过 PR chain (#3550 ✅ → #3816 ✅ → #3845 ✅ → #3881 → #3563 → #3757 → #3109) 逐步推进合并，但核心 PR #3757（501 行 diff, 24 文件, 136 commits）仍处于 Draft 状态，缺乏回归测试、用户文档和第三方验证。paulromano 尚未 review。shimwell 报告了 `free(): corrupted unsorted chunks` crash。用户需要 point detector 功能来解决 BEST 模型（96.9% void, 2640 cells, 4136 D-T sources）中传统 weight window 方法已被证明无效的深穿透问题。
- **根因分析**：合并阻塞的主要原因是 (1) 无回归测试, (2) 无 Sphinx 文档, (3) 已知 crash bug 未定位, (4) 无第三方基准验证。这些都是外部贡献者可以解决的问题。
- **目标**：
  1. 在本地环境搭建 point-detector 分支并验证基本功能
  2. 贡献标准回归测试（对合并影响最大、门槛最低）
  3. 调查 shimwell 报告的内存崩溃 bug，定位是否与 `ParticleRay` diamond inheritance 相关
  4. 在 BEST 模型上运行 point detector vs analog 基准，收集 benchmark 数据
  5. 编写 Sphinx 文档（用户指南 + Python API docstrings）
  6. 向社区提交贡献（PR comments、维护者联系）
- **非目标（不做什么）**：
  - 不重写 point detector 核心实现 — 只贡献测试/文档/bugfix，保持 GuySten 主导权
  - 不修改 relativistic kinematics (#3563) — 那是独立 PR
  - 不实现 photon/coupled n,γ 的 point detector 支持 — 那是 future work
  - 不处理 MPI 并行下的 point detector — 单线程/OpenMP 先验证
  - 不合并到 upstream develop — 只在 fork 上工作，贡献通过 PR review comments 或独立 PR
- **已有代码/流程复用分析**：
  - `pulse_height` 回归测试结构：**复用** — 作为 point detector 测试的模板（相似的 special tally type）
  - `PyAPITestHarness`：**复用** — 标准回归测试 harness
  - `tests/unit_tests/conftest.py` 的 `sphere_model` fixture：**复用** — 简单球模型作为测试基础
  - `docs/source/usersguide/tallies.rst`：**复用** — 在现有 Filters 章节追加 PointFilter 文档
  - `docs/source/usersguide/variance_reduction.rst`：**复用** — 在现有 VR 章节追加 point detector 指南
  - BEST 模型 `inputs/model.xml`：**复用** — 已有完整的 2640-cell 几何和 4136 D-T 源
  - MCNP BEST 模型 `inputs/BEST_tbm_mcnp`：**复用** — 1.5 MB MCNP 输入卡，可配置 F5 tally 作为 C/E 参考

## 技术方案

- **方案概述**：6 个 Phase 递进实施，Phase 0 是环境准备，Phase 1 贡献回归测试（最高优先级），Phase 2 调查 crash bug，Phase 3 BEST 模型基准，Phase 4 文档，Phase 5 社区沟通。每个 Phase 独立可交付。
- **关键设计决策**：
  1. 从 `GuySten/openmc:point-detector` 分支创建本地工作分支，而非从 `itay-space/openmc:deploy`（后者有 crash bug, 且架构设计被 gridley 质疑）
  2. 回归测试采用 `PointFilter` + `flux` score 的最小模型，对标 MCNP F5 的 isotropic point source → detector 经典验证问题
  3. Bug 调查聚焦 `ParticleRay : public Ray, public Particle` 的 diamond inheritance via `virtual public GeometryState`，检查 vptr layout 和 destructor chain
  4. BEST benchmark 使用 analog 作为 ground truth（无 VR bias），point detector 作为估计量，对比统计效率
  5. 文档遵循 OpenMC 现有文档风格：RST for user guide, numpydoc for Python API
- **影响范围**：
  - 新文件：`tests/regression_tests/point_detector/test.py`, `__init__.py`, `inputs_true.dat`, `results_true.dat`
  - 新文件：`tests/regression_tests/point_detector_multi/` (多检测器变体)
  - 修改文件：`docs/source/usersguide/tallies.rst` (追加 PointFilter 文档)
  - 修改文件：`openmc/filter.py` (补全 `PointFilter` docstrings 和 `from_xml_element`)
  - 新脚本：`scripts/benchmark_point_detector_best.py` (在 GVR workspace)
  - 不修改 C++ 源码（Phase 2 bugfix 除外，需与 GuySten 协调）

## Error & Rescue Map（关键失败路径映射）

| 代码路径/操作                                      | 可能的失败                                              | 错误类型            | 已处理？         | 处理方式                                                                   | 用户可见行为                  |
| -------------------------------------------------- | ------------------------------------------------------- | ------------------- | ---------------- | -------------------------------------------------------------------------- | ----------------------------- |
| `git fetch GuySten && git checkout point-detector` | 远程分支不存在或冲突                                    | git error           | Y                | 从 #3757 PR ref 获取: `git fetch upstream pull/3757/head:point-detector`   | 明确错误提示                  |
| CMake 构建 point-detector 分支                     | 编译失败（新文件缺少依赖）                              | build error         | Y                | 检查 CMakeLists.txt 是否包含 `filter_point.cpp`，手动添加                  | 编译错误信息                  |
| `ParticleRay` diamond inheritance                  | `free(): corrupted unsorted chunks`                     | runtime crash       | N → Phase 2 目标 | 需要 valgrind/ASAN 分析                                                    | **CRITICAL GAP** — 程序 abort |
| Point detector 在 void region 射线追踪             | 射线未到达检测器（traversal_distance < total_distance） | silent skip         | Y                | `score_point_tally_impl` 中 `if (distance < total_distance) continue` 逻辑 | 零计数（不一定是 bug）        |
| BEST 模型 4136 源 + point detector                 | 内存爆炸或性能塌陷                                      | OOM/performance     | N                | 限制检测器数量，先用 1-2 个检测器验证                                      | 内存不足或超长运行时间        |
| `PointFilter.from_xml_element` 未实现              | Python API 反序列化失败                                 | NotImplementedError | N                | Phase 4 补全                                                               | Python 报错                   |

## 执行计划

### Phase 0: Environment Setup

#### ✅ Task 0.1: Fetch and checkout point-detector branch
- **目标**：获取 GuySten 的 point-detector 分支代码到本地
- **修改内容**：
  - Git 操作：添加 GuySten remote 并 fetch point-detector 分支
  ```bash
  cd /home/gw/opt/openmc
  git remote add guystem https://github.com/GuySten/openmc.git 2>/dev/null || true
  git fetch guystem point-detector
  git checkout -b point-detector-contrib guystem/point-detector
  ```
- **修改边界**：不修改任何源文件，仅 git 分支操作
- **测试要求**：
  - 运行 `git log --oneline -5` 确认在 GuySten 的 point-detector 分支上
  - 预期输出：最近提交包含 "fix missing declarations" 或类似 point-detector 相关消息
- **验收标准**：
  - ✅ 本地分支 `point-detector-contrib` 存在且基于 `guystem/point-detector`
  - ✅ `git diff upstream/develop --stat` 输出包含 `filter_point.h`, `filter_point.cpp`, `tally_scoring.h` 等 24 个文件
- **潜在风险**：GuySten 可能在 force-push 更新分支，导致本地分支需要重新 fetch。用 `git fetch guystem` 前先确认远程 HEAD。
- **难度**：Easy
- **目标 PR**：本地准备，不直接对应 PR

#### ✅ Task 0.2: Build point-detector branch
- **目标**：编译包含 point detector 代码的 OpenMC
- **修改内容**：
  - CMake 配置并构建（使用现有 build 目录或创建新 build-pd 目录）
  ```bash
  cd /home/gw/opt/openmc
  mkdir -p build-pd && cd build-pd
  cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
           -DOPENMC_USE_OPENMP=ON \
           -DCMAKE_INSTALL_PREFIX=/home/gw/opt/openmc/build-pd/install
  make -j$(nproc)
  ```
  - 重新安装 Python 包（开发模式）
  ```bash
  cd /home/gw/opt/openmc
  pip install -e .
  ```
- **修改边界**：不修改源码，仅构建产物。不覆盖 `build/` 目录（保留 ww_collision_only 分支的构建）
- **测试要求**：
  - 运行 `build-pd/bin/openmc --version` 确认编译成功
  - 运行 `python -c "import openmc; print(openmc.__version__)"` 确认 Python 包安装成功
  - 运行 `python -c "from openmc import PointFilter; print('PointFilter available')"` 确认新 filter 可用
- **验收标准**：
  - ✅ `build-pd/bin/openmc` 存在且可执行
  - ✅ `python -c "from openmc import PointFilter"` 不报错
  - ✅ C++ 编译无 error（warning 可接受）
- **潜在风险**：point-detector 分支可能与当前 develop HEAD 有冲突需要 rebase。如果编译失败，先检查 #3757 的 CI 状态（最后一次 CI 全绿 17/17）。
- **难度**：Easy
- **目标 PR**：本地准备

#### ✅ Task 0.3: Smoke test — verify point detector basic functionality
- **目标**：运行一个最小的 point detector 模型，验证功能端到端工作
- **修改内容**：
  - 创建一个临时 Python 脚本 `$tmpdir/test_pd_smoke.py`
  ```python
  import numpy as np
  import openmc

  # Materials
  water = openmc.Material()
  water.add_nuclide('H1', 2.0)
  water.add_nuclide('O16', 1.0)
  water.set_density('g/cc', 1.0)

  # Geometry: water sphere with vacuum BC
  sphere = openmc.Sphere(r=50.0, boundary_type='vacuum')
  cell = openmc.Cell(fill=water, region=-sphere)
  geometry = openmc.Geometry([cell])

  # Settings
  settings = openmc.Settings()
  settings.run_mode = 'fixed source'
  settings.batches = 5
  settings.particles = 1000
  settings.source = openmc.IndependentSource(
      space=openmc.stats.Point((0, 0, 0)),
      energy=openmc.stats.Discrete([14.1e6], [1.0]),
  )

  # Tally with PointFilter
  point_filter = openmc.PointFilter(
      bins=[((30, 0, 0), 0.01), ((0, 30, 0), 0.01)]
  )
  energy_filter = openmc.EnergyFilter(np.logspace(np.log10(1e-5), np.log10(15e6), 51))
  tally = openmc.Tally()
  tally.filters = [point_filter, energy_filter]
  tally.scores = ['flux']

  model = openmc.Model(geometry, openmc.Materials([water]), settings, openmc.Tallies([tally]))
  sp_file = model.run()

  # Check results
  with openmc.StatePoint(sp_file) as sp:
      t = sp.tallies[tally.id]
      flux = t.mean.flatten()
      nonzero = np.count_nonzero(flux)
      print(f"Total bins: {len(flux)}, nonzero: {nonzero}")
      print(f"Total flux (det 1): {flux[:50].sum():.6e}")
      print(f"Total flux (det 2): {flux[50:].sum():.6e}")
      assert nonzero > 0, "No nonzero flux bins — point detector not scoring!"
  print("SMOKE TEST PASSED")
  ```
- **修改边界**：仅创建临时测试脚本，不修改源码
- **测试要求**：
  - 运行 `cd $tmpdir && OMP_NUM_THREADS=2 python test_pd_smoke.py`
  - 预期输出：两个 detector 都有非零通量，输出 "SMOKE TEST PASSED"
- **验收标准**：
  - ✅ OpenMC 运行完成无 crash
  - ✅ 至少有部分能量 bin 有非零通量
  - ✅ 两个 detector 位置的通量量级合理（基于 14.1 MeV 源 → 50 cm 水球中 30 cm 处的衰减）
- **潜在风险**：`PointFilter` Python API 的 bins 格式可能与 C++ 端 `from_xml` 不一致。如果 `to_xml_element` 输出的 XML 格式 C++ 端无法解析，需检查 XML 序列化逻辑。
- **难度**：Easy → Medium（取决于是否遇到 API 不一致）
- **目标 PR**：本地验证

### Phase 1: Regression Test Contribution

#### ✅ Task 1.1: Create basic point detector regression test
- **目标**：创建标准 OpenMC 回归测试，验证 point detector 在简单几何中的正确性
- **修改内容**：
  - 文件 `tests/regression_tests/point_detector/__init__.py`：空文件
  - 文件 `tests/regression_tests/point_detector/test.py`：
    ```python
    import numpy as np
    import openmc
    import pytest

    from tests.testing_harness import PyAPITestHarness


    @pytest.fixture
    def model():
        model = openmc.model.Model()

        # Water sphere — simple geometry for point detector validation
        water = openmc.Material()
        water.add_nuclide('H1', 2.0)
        water.add_nuclide('O16', 1.0)
        water.set_density('g/cc', 1.0)
        model.materials = openmc.Materials([water])

        sphere = openmc.Sphere(r=50.0, boundary_type='vacuum')
        cell = openmc.Cell(fill=water, region=-sphere)
        model.geometry = openmc.Geometry([cell])

        model.settings.run_mode = 'fixed source'
        model.settings.batches = 5
        model.settings.particles = 1000
        model.settings.source = openmc.IndependentSource(
            space=openmc.stats.Point((0, 0, 0)),
            energy=openmc.stats.Discrete([14.1e6], [1.0]),
        )

        # Point detector tally
        point_filter = openmc.PointFilter(
            bins=[((30, 0, 0), 1.0)]
        )
        energy_filter = openmc.EnergyFilter([0.0, 1.0, 1e3, 1e6, 14.2e6])
        tally = openmc.Tally(name='point_detector')
        tally.filters = [point_filter, energy_filter]
        tally.scores = ['flux']
        model.tallies = [tally]

        return model


    def test_point_detector(model):
        harness = PyAPITestHarness('statepoint.5.h5', model)
        harness.main()
    ```
- **修改边界**：不修改 `tests/regression_tests/` 之外的文件。不修改 `conftest.py`。
- **测试要求**：
  - 运行 `cd /home/gw/opt/openmc && OMP_NUM_THREADS=2 pytest tests/regression_tests/point_detector/ --update -v` 生成 reference files
  - 运行 `OMP_NUM_THREADS=2 pytest tests/regression_tests/point_detector/ -v` 验证通过
  - 预期输出：1 passed
- **验收标准**：
  - ✅ `tests/regression_tests/point_detector/inputs_true.dat` 生成
  - ✅ `tests/regression_tests/point_detector/results_true.dat` 生成
  - ✅ `pytest tests/regression_tests/point_detector/` 通过（非 `--update` 模式）
  - ✅ `results_true.dat` 包含非零 k-eff 或 tally 值（fixed source 模式下 k-eff = 0，但 tally 值非零）
- **潜在风险**：`PyAPITestHarness` 可能对 fixed source + point detector 有兼容性问题（harness 默认比较 k-eff，而 fixed source 无 k-eff）。如果失败，需要使用 `HashedPyAPITestHarness` 或自定义 harness。`statepoint` 中 point detector tally 的序列化格式可能也需要验证。
- **难度**：Medium
- **目标 PR**：#3757（作为 review comment 附带），或独立 PR 到 GuySten/openmc

#### ✅ Task 1.2: Create multi-detector regression test
- **目标**：测试多个 point detector 位置的正确性（多 bin PointFilter）
- **修改内容**：
  - 文件 `tests/regression_tests/point_detector_multi/__init__.py`：空文件
  - 文件 `tests/regression_tests/point_detector_multi/test.py`：
    ```python
    import numpy as np
    import openmc
    import pytest

    from tests.testing_harness import PyAPITestHarness


    @pytest.fixture
    def model():
        model = openmc.model.Model()

        # Iron slab — shielding problem
        iron = openmc.Material()
        iron.add_nuclide('Fe56', 1.0)
        iron.set_density('g/cc', 7.87)
        model.materials = openmc.Materials([iron])

        # Slab geometry: source at z=0, detectors along z-axis
        z_min = openmc.ZPlane(-5.0, boundary_type='vacuum')
        z_max = openmc.ZPlane(100.0, boundary_type='vacuum')
        cyl = openmc.ZCylinder(r=50.0, boundary_type='vacuum')
        cell = openmc.Cell(fill=iron, region=+z_min & -z_max & -cyl)
        model.geometry = openmc.Geometry([cell])

        model.settings.run_mode = 'fixed source'
        model.settings.batches = 5
        model.settings.particles = 1000
        model.settings.source = openmc.IndependentSource(
            space=openmc.stats.Point((0, 0, 0)),
            energy=openmc.stats.Discrete([2.0e6], [1.0]),
        )

        # Multiple point detectors at different distances
        point_filter = openmc.PointFilter(bins=[
            ((0, 0, 10), 1.0),
            ((0, 0, 30), 1.0),
            ((0, 0, 50), 1.0),
        ])
        tally = openmc.Tally(name='multi_point_detector')
        tally.filters = [point_filter]
        tally.scores = ['flux']
        model.tallies = [tally]

        return model


    def test_point_detector_multi(model):
        harness = PyAPITestHarness('statepoint.5.h5', model)
        harness.main()
    ```
- **修改边界**：不修改 `tests/regression_tests/point_detector_multi/` 之外的文件
- **测试要求**：
  - 运行 `pytest tests/regression_tests/point_detector_multi/ --update -v`
  - 运行 `pytest tests/regression_tests/point_detector_multi/ -v`
  - 预期输出：1 passed
- **验收标准**：
  - ✅ 3 个检测器位置均有 tally 结果
  - ✅ 通量随距离单调递减（10 cm > 30 cm > 50 cm）— 这是物理合理性检查，不是 harness 比较的一部分，需人工确认 `results_true.dat`
  - ✅ `pytest` 非 `--update` 模式通过
- **潜在风险**：铁中 2 MeV 中子的平均自由程约 2-3 cm，50 cm 处的通量可能极度衰减至浮点下溢。可能需要增大 `particles` 或减小探测器距离。
- **难度**：Medium
- **目标 PR**：#3757

#### ✅ Task 1.3: Create unit test for PointFilter Python API
- **目标**：验证 `PointFilter` 的 Python API 正确性（构造、序列化、反序列化）
- **修改内容**：
  - 文件 `tests/unit_tests/test_filter_point.py`：
    ```python
    import numpy as np
    import openmc
    import pytest


    def test_point_filter_creation():
        """Test PointFilter can be created with valid bins."""
        pf = openmc.PointFilter(bins=[((1, 2, 3), 0.5)])
        assert pf.num_bins == 1

        pf = openmc.PointFilter(bins=[
            ((0, 0, 0), 1.0),
            ((10, 0, 0), 0.5),
            ((0, 10, 0), 0.01),
        ])
        assert pf.num_bins == 3


    def test_point_filter_invalid_bins():
        """Test PointFilter rejects invalid bin specifications."""
        with pytest.raises((TypeError, ValueError)):
            openmc.PointFilter(bins=[((1, 2), 0.5)])  # 2D position

        with pytest.raises((TypeError, ValueError)):
            openmc.PointFilter(bins=[(1, 2, 3, 0.5)])  # flat tuple


    def test_point_filter_xml_roundtrip():
        """Test PointFilter serializes and deserializes correctly."""
        pf = openmc.PointFilter(bins=[
            ((1.0, 2.0, 3.0), 0.5),
            ((4.0, 5.0, 6.0), 1.0),
        ])
        elem = pf.to_xml_element()
        # Check XML structure
        assert elem.tag == 'filter'
        assert elem.get('type') == 'point'
        bins_text = elem.find('bins').text
        values = [float(x) for x in bins_text.split()]
        assert len(values) == 8  # 2 detectors × 4 values each
        assert values == [1.0, 2.0, 3.0, 0.5, 4.0, 5.0, 6.0, 1.0]


    def test_point_filter_in_tally():
        """Test PointFilter can be used in a Tally."""
        pf = openmc.PointFilter(bins=[((0, 0, 100), 1.0)])
        tally = openmc.Tally()
        tally.filters = [pf]
        tally.scores = ['flux']
        # Should not raise
        tallies = openmc.Tallies([tally])
        elem = tallies.to_xml_element()
        assert elem is not None
    ```
- **修改边界**：不修改 `tests/unit_tests/` 之外的文件
- **测试要求**：
  - 运行 `pytest tests/unit_tests/test_filter_point.py -v`
  - 预期输出：4 passed（或部分跳过，取决于 `from_xml_element` 实现状态）
- **验收标准**：
  - ✅ PointFilter 创建和 num_bins 测试通过
  - ✅ XML 序列化测试通过
  - ✅ 无效 bins 被正确拒绝
- **潜在风险**：`PointFilter` 的 bins setter 接受格式可能与测试中的 tuple-of-tuples 不一致。需先检查 `openmc/filter.py` 中 `PointFilter.bins.setter` 的实际参数校验逻辑。PR #3757 diff 显示 `cv.check_type('bins', bins, Sequence, tuple)` — 需要确认嵌套 tuple 能通过。
- **难度**：Easy
- **目标 PR**：#3757

### Phase 2: Bug Investigation

#### ✅ Task 2.1: Reproduce the crash with ASAN/valgrind
- **目标**：在本地复现 shimwell 报告的 `free(): corrupted unsorted chunks` crash 并定位根因
- **修改内容**：
  - 使用 AddressSanitizer 重新构建 OpenMC
  ```bash
  cd /home/gw/opt/openmc
  mkdir -p build-pd-asan && cd build-pd-asan
  cmake .. -DCMAKE_BUILD_TYPE=Debug \
           -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
           -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
           -DOPENMC_USE_OPENMP=OFF
  make -j$(nproc)
  ```
  - 运行 Task 0.3 的 smoke test，使用 ASAN build
  ```bash
  export PATH=/home/gw/opt/openmc/build-pd-asan/bin:$PATH
  cd $tmpdir && python test_pd_smoke.py 2>&1 | tee asan_output.log
  ```
- **修改边界**：不修改源码。仅创建新的 build 目录和运行诊断。
- **测试要求**：
  - 如果 ASAN 报告 heap-buffer-overflow 或 use-after-free：记录完整 stack trace
  - 如果 crash 不复现：尝试增大 particles 到 100000 和 batches 到 50（更多碰撞更可能触发）
  - 如果仍不复现：尝试 itay-space/openmc:deploy 分支（shimwell 原始报告的分支）
  - 预期输出：ASAN 报告或 "crash not reproduced"
- **验收标准**：
  - ✅ 运行了 ASAN 构建的 point detector smoke test
  - ✅ 如果 crash 复现：有完整 stack trace 和初步根因分析
  - ✅ 如果 crash 不复现：记录了测试环境和参数，排除了本地复现可能
- **潜在风险**：ASAN 与 OpenMP 可能不兼容（需关闭 OpenMP）。ASAN 会显著降低性能（~2x slowdown）。
- **难度**：Medium → Hard
- **目标 PR**：#3757 或 #3109（作为 bug report comment）

#### ✅ Task 2.2: Analyze diamond inheritance safety
- **目标**：静态分析 `ParticleRay` 的 diamond inheritance 是否有潜在的 UB（未定义行为）
- **修改内容**：
  - 检查以下关键点（代码审查，不修改代码）：
    1. `ParticleData : virtual public GeometryState` — 确认 virtual inheritance
    2. `Ray : virtual public GeometryState` — 确认 virtual inheritance
    3. `ParticleRay : public Ray, public Particle` — 确认 single `GeometryState` subobject
    4. 检查 `ParticleRay` 构造函数是否正确初始化所有基类
    5. 检查析构函数链是否正确（virtual destructor in GeometryState）
    6. 检查 `p.Ray::trace()` 的显式基类调用是否安全
  - 输出分析报告到 `$GVR_WORKSPACE/analysis/pd_diamond_inheritance_analysis.md`
- **修改边界**：不修改源码。仅产出分析报告。
- **测试要求**：
  - 编写一个最小的 C++ 测试程序验证 `sizeof(ParticleRay)` 和 vtable layout
  ```cpp
  #include "openmc/ray.h"
  #include <iostream>
  int main() {
      std::cout << "sizeof(GeometryState): " << sizeof(openmc::GeometryState) << "\n";
      std::cout << "sizeof(Ray): " << sizeof(openmc::Ray) << "\n";
      std::cout << "sizeof(ParticleData): " << sizeof(openmc::ParticleData) << "\n";
      std::cout << "sizeof(Particle): " << sizeof(openmc::Particle) << "\n";
      std::cout << "sizeof(ParticleRay): " << sizeof(openmc::ParticleRay) << "\n";
      return 0;
  }
  ```
- **验收标准**：
  - ✅ 分析报告覆盖了上述 6 个检查点
  - ✅ 如果发现问题：有具体的修复建议
  - ✅ sizeof 测试编译并运行，确认 ParticleRay 只有一个 GeometryState 子对象
- **潜在风险**：diamond inheritance 在 OpenMC 中没有先例，paulromano 可能要求重构为组合模式（composition over inheritance）。这不是我们可以单方面决定的。
- **难度**：Hard
- **目标 PR**：#3881 (ParticleRay PR) 或 #3757

### Phase 3: BEST Benchmark

#### 🛑 Task 3.1: Design BEST point detector benchmark [BLOCKED: BEST model has reflective BCs — incompatible with point detectors]
- **目标**：设计在 BEST 融合包层模型上的 point detector 基准测试方案
- **修改内容**：
  - 文件 `$GVR_WORKSPACE/scripts/benchmark_point_detector_best.py`：
    - 读取 `inputs/model.xml` 几何和材料
    - 读取 `inputs/model-tbm.xml` 源定义
    - 配置 3-5 个 point detector 位置（TBM 第一壁后方关键屏蔽位点）
    - 配置 analog 模式作为参考
    - 配置 point detector 模式
    - 统一 particles/batches 设置，确保可比性
  - 输出：`$GVR_WORKSPACE/results/s24_point_detector/benchmark_design.md`
- **修改边界**：不修改 OpenMC 源码，不修改 BEST 模型输入文件
- **测试要求**：
  - 脚本能生成 analog 和 point-detector 两组 `model.xml`
  - 预期输出：两组模型文件和 benchmark 设计文档
- **验收标准**：
  - ✅ Point detector 位置在 BEST 几何中有效（不在 void 中、不在边界外）
  - ✅ Analog 和 point-detector 模型只在 tally 配置上有差异
  - ✅ 两组模型都能通过 `openmc --plot` 的几何检查
- **潜在风险**：BEST 模型的全真空边界条件是 point detector 的有效前提（`nonvacuum_boundary_present` 检查）。如果 BEST 有任何 reflective/periodic 面，point detector 将 fatal_error。需确认。
- **难度**：Medium
- **目标 PR**：无直接 PR，但为 #3757 提供 benchmark evidence
- **依赖**：Phase 0 完成

#### ⏸ Task 3.2: Run BEST benchmark [SKIPPED: dependency Task 3.1 blocked]
- **目标**：在本地 36 核机器上运行 BEST point detector 基准
- **修改内容**：
  - 运行 Task 3.1 生成的两组模型
  - 收集结果到 `$GVR_WORKSPACE/results/s24_point_detector/`
  ```bash
  # Analog reference
  cd results/s24_point_detector/analog
  OMP_NUM_THREADS=34 openmc 2>&1 | tee analog.log

  # Point detector
  cd ../point_detector
  OMP_NUM_THREADS=34 openmc 2>&1 | tee point_detector.log
  ```
- **修改边界**：不修改源码
- **测试要求**：
  - 两组运行都正常完成（无 crash）
  - 运行时间记录
  - 预期输出：statepoint 文件和运行日志
- **验收标准**：
  - ✅ Analog 运行完成，产出 statepoint
  - ✅ Point detector 运行完成，产出 statepoint
  - ✅ Point detector 运行无 `free(): corrupted unsorted chunks` crash
- **潜在风险**：BEST 模型 2640 cells + 4136 源 + point detector 的组合可能导致极长运行时间（每个碰撞都要为每个 detector 做射线追踪）。初始 benchmark 用低粒子数（1000 particles × 10 batches）验证可行性。
- **难度**：Medium
- **目标 PR**：为 #3757 提供 benchmark evidence
- **依赖**：Phase 0, Task 3.1

#### ⏸ Task 3.3: Analyze BEST benchmark results [SKIPPED: dependency Task 3.2 skipped]
- **目标**：分析 point detector vs analog 的统计效率对比
- **修改内容**：
  - 分析脚本 `$GVR_WORKSPACE/scripts/analyze_pd_benchmark.py`：
    - 从两组 statepoint 提取 tally 结果
    - 计算 Figure of Merit (FOM = 1 / (σ² × T))
    - 对比每个 detector 位置的 FOM ratio
    - 如果 MCNP F5 数据存在，计算 C/E（OpenMC point detector / MCNP F5）
  - 输出报告：`$GVR_WORKSPACE/results/s24_point_detector/benchmark_report.md`
- **修改边界**：不修改源码
- **测试要求**：
  - 脚本从 statepoint 正确提取所有 tally 数据
  - 预期输出：对比表格和 FOM 分析
- **验收标准**：
  - ✅ 每个 detector 位置有 mean ± std_dev 结果
  - ✅ FOM 对比有明确结论（point detector 在深穿透位置是否优于 analog）
  - ✅ 如 MCNP F5 数据存在：C/E 在合理范围内（典型 <10% for flux）
- **潜在风险**：低粒子数下 analog 估计在深穿透位置可能全为零（无粒子到达），导致 FOM 为零或未定义。这正是 point detector 的价值所在，但需要在报告中正确表述。
- **难度**：Medium
- **目标 PR**：为 #3757 review comment 提供数据
- **依赖**：Task 3.2

### Phase 4: Documentation

#### Task 4.1: Write user guide section for point detector
- **目标**：在 OpenMC Sphinx 文档中添加 Point Detector / NEE 用户指南
- **修改内容**：
  - 文件 `docs/source/usersguide/tallies.rst`：在 Filters 段落末尾添加 `PointFilter` 的使用说明
    - 概述 next-event estimator 原理
    - 给出 Python API 使用示例
    - 说明限制条件（仅 vacuum BC, 仅 flux score 等）
    - 列出与其他 filter 的兼容性
  - 或者，如果 maintainer 倾向独立章节：创建 `docs/source/usersguide/point_detector.rst`
- **修改边界**：不修改 `docs/source/usersguide/tallies.rst` 中现有内容，仅追加新章节。不修改源码。
- **测试要求**：
  - 运行 `cd docs && make html` 确认文档构建成功
  - 手动检查生成的 HTML 页面中 PointFilter 文档渲染正确
  - 预期输出：`docs/build/html/usersguide/tallies.html` 包含 PointFilter 章节
- **验收标准**：
  - ✅ Sphinx `make html` 无 error
  - ✅ PointFilter 章节包含：概述、使用示例、限制条件、兼容性
  - ✅ 数学公式（NEE 估计量 $\hat{\phi} = \sum_i w_i \cdot p(\Omega_i) \cdot e^{-\tau_i} / r_i^2$）正确渲染
  - ✅ 代码示例可运行（与 Task 0.3 smoke test 一致）
- **潜在风险**：OpenMC 文档构建依赖 Sphinx + 多个扩展（`sphinx-numfig` 等），本地可能缺少。先运行 `pip install -r docs/requirements.txt`。
- **难度**：Medium
- **目标 PR**：#3757
- **依赖**：Phase 0

#### Task 4.2: Improve PointFilter Python API docstrings
- **目标**：补全 `PointFilter` 类的 numpydoc 格式 docstrings
- **修改内容**：
  - 文件 `openmc/filter.py`：在 `PointFilter` 类中补全 docstrings
    - `__init__` / class docstring: 完善 Parameters, Attributes, Examples
    - `bins` property/setter: 说明 bins 格式 `[((x, y, z), R0), ...]`
    - `to_xml_element`: 标准 Returns docstring
    - 添加 `from_xml_element` classmethod（如果不存在）
  - 文件 `openmc/lib/filter.py`：补全 `PointFilter` 类的 docstring
- **修改边界**：仅修改 `openmc/filter.py` 和 `openmc/lib/filter.py` 中的 docstrings。不修改功能逻辑。
- **测试要求**：
  - `python -c "from openmc import PointFilter; help(PointFilter)"` 输出完整文档
  - 预期输出：numpydoc 格式的帮助信息
- **验收标准**：
  - ✅ `PointFilter` class docstring 包含 Parameters, Attributes, Examples
  - ✅ bins 格式 `[((x, y, z), R0), ...]` 有明确说明
  - ✅ R0（exclusion sphere radius）的物理含义有说明
  - ✅ `from_xml_element` 方法存在（或标注 TODO）
- **潜在风险**：`PointFilter` PR 中的 `from_xml_element` 似乎未实现。如果只是文档+docstring 贡献，这不是阻塞项，但应标注为 TODO。
- **难度**：Easy
- **目标 PR**：#3757

### Phase 5: Community Engagement

#### Task 5.1: Post regression test contribution on PR #3757
- **目标**：在 #3757 上 comment，提供回归测试代码，请求 review
- **修改内容**：
  - 写一个 GitHub PR comment，包含：
    1. 测试代码（Task 1.1 和 1.2）
    2. 测试通过的截图/日志
    3. 建议将测试纳入 PR
    4. 询问 GuySten 是否希望我们直接 push 到 `GuySten/point-detector` 分支，或作为独立 PR
- **修改边界**：不修改代码，仅 GitHub 社区操作
- **测试要求**：无
- **验收标准**：
  - ✅ Comment 已发布在 #3757
  - ✅ Comment 包含可运行的测试代码
  - ✅ 收到 GuySten 的回复或 acknowledgment
- **潜在风险**：GuySten 可能已经在准备自己的测试。提前询问可以避免重复工作。
- **难度**：Easy
- **目标 PR**：#3757
- **依赖**：Phase 1

#### Task 5.2: Share BEST benchmark results
- **目标**：在 #3757 或 #3109 上分享 BEST 模型基准结果，展示 point detector 在大型工程问题上的价值
- **修改内容**：
  - 写一个 GitHub comment，包含：
    1. BEST 模型简介（2640 cells, 96.9% void, 4136 D-T sources）
    2. Point detector vs analog 的 FOM 对比
    3. 如果可用，MCNP F5 对比
    4. 结论：point detector 是否适用于这类深穿透屏蔽问题
- **修改边界**：不修改代码
- **测试要求**：无
- **验收标准**：
  - ✅ Benchmark 结果已发布在 GitHub
  - ✅ 数据有物理合理性（符合 Phase 3 的验收标准）
- **潜在风险**：如果 BEST 基准结果不理想（e.g. point detector 在 96.9% void 中也不好用），这仍然是有价值的结果——需要在 comment 中如实报告。
- **难度**：Easy
- **目标 PR**：#3757 或 #3109
- **依赖**：Phase 3

#### Task 5.3: Contact paulromano about review priority
- **目标**：礼貌地请求 paulromano 审查 #3757 的优先级提升
- **修改内容**：
  - 方案一：在 #3757 上 @paulromano 留 comment，提供上下文：
    - 现有的 PR chain 进展（3/6 merged）
    - 回归测试和文档贡献已就绪
    - BEST 模型基准展示了实际需求
    - 询问是否有 architectural concerns 需要提前讨论
  - 方案二：在 OpenMC Discourse 论坛发帖讨论 point detector 需求
- **修改边界**：不修改代码
- **测试要求**：无
- **验收标准**：
  - ✅ Message 已发送
  - ✅ 内容专业、简洁、有建设性
- **潜在风险**：paulromano 可能有合理的 architectural concerns（如 gridley 提出的"should be a tally type not an estimator"争议）。准备好接受可能的设计变更要求。
- **难度**：Easy
- **目标 PR**：#3757
- **依赖**：Phase 1, Phase 3 (optional)

## 回归检查清单

- [ ] 全量现有测试不受影响：`OMP_NUM_THREADS=2 pytest tests/regression_tests/ -x --timeout=300`（point-detector 分支上）
- [ ] 新回归测试通过：`pytest tests/regression_tests/point_detector/ tests/regression_tests/point_detector_multi/ -v`
- [ ] 新单元测试通过：`pytest tests/unit_tests/test_filter_point.py -v`
- [ ] 无新 lint 警告：`flake8 openmc/filter.py --select=E,W --max-line-length=100`
- [ ] C++ 编译无 error：`make -j$(nproc)` in `build-pd/`
- [ ] ASAN 构建运行 smoke test 通过（或 crash 已记录和报告）
- [ ] Sphinx 文档构建无 error：`cd docs && make html`
- [ ] CI 检查通过（push 到 fork 后 GitHub Actions 绿灯）
- [ ] 所有新文件包含 `__init__.py`（回归测试目录）
- [ ] `PointFilter` 在 Python API 中可访问：`from openmc import PointFilter`

## 审查日志

| 轮次     | 聚焦                                         | 发现问题数 | 已修正 | 剩余  |
| -------- | -------------------------------------------- | ---------- | ------ | ----- |
| R1       | 结构完整性                                   | 5          | 5      | 0     |
| R2       | 可执行性                                     | 4          | 4      | 0     |
| R3       | 风险与边缘                                   | 3          | 3      | 0     |
| **终止** | **T1 — 收敛终止 (≥3 轮 + 最近一轮 issue=0)** |            |        | **0** |

### Completion Summary

| 维度               | 结果                                                                 |
| ------------------ | -------------------------------------------------------------------- |
| 背景与目标         | 完整 — 问题描述、目标、非目标、复用分析均存在                        |
| 技术方案           | 完整 — 方案概述、关键决策、影响范围                                  |
| Error & Rescue Map | 6 条路径覆盖, 1 CRITICAL GAP (ParticleRay crash) 已纳入 Phase 2 任务 |
| 执行计划           | 6 Phases, 14 Tasks                                                   |
| 回归检查清单       | 10 项项目特定检查                                                    |
| 已知局限           | 无                                                                   |

### Scope Mode: EXPANSION
新功能贡献方案，允许适度扩展（e.g. BEST benchmark, community engagement 超越最小测试贡献范围）。

### R1 Issues (结构完整性)
- **Issue R1-1**: 缺少 Error & Rescue Map → 已添加 6 条关键失败路径 ✅ 已修正
- **Issue R1-2**: Task 3.1 缺少"潜在风险"字段 → 已补充 nonvacuum_boundary_present 检查风险 ✅ 已修正
- **Issue R1-3**: 缺少"已有代码/流程复用分析" → 已在背景与目标中补充 ✅ 已修正
- **Issue R1-4**: 部分 Task 缺少"目标 PR"标注 → 已补充每个 Task 的目标 PR ✅ 已修正
- **Issue R1-5**: 回归检查清单仅有通用项 → 已替换为项目特定的 10 项检查 ✅ 已修正

### R2 Issues (可执行性)
- **Issue R2-1**: Task 0.2 和 0.3 的依赖关系未标注 → 0.3 依赖 0.2，已在流程中保持顺序 ✅ 已修正（Phase 内 Task 按顺序执行）
- **Issue R2-2**: Task 1.1 测试要求中"全部通过"过于模糊 → 已明确为 "1 passed" + 具体文件存在性检查 ✅ 已修正
- **Issue R2-3**: Task 3.2 运行时间可能过长（BEST 模型 + point detector 开销） → 已添加"初始 benchmark 用低粒子数"的注意事项 ✅ 已修正
- **Issue R2-4**: 时序推演缺失 — Phase 0→1 可能的阻塞点：如果 point-detector 分支编译失败（依赖 #3881 ParticleRay），Phase 1-5 全部阻塞。缓解：先用 `itay-space/openmc:deploy-old` 作为 fallback（shimwell 确认可用）。Phase 2→3 的阻塞点：如果 ASAN 发现 crash 是 fundamental design issue，Phase 3 的 BEST benchmark 可能不可靠。缓解：Phase 3 独立于 Phase 2（两者无依赖），crash 仅在特定条件下触发。 → 已在 Task 0.2 和 2.1 风险中说明 ✅ 已修正

### R3 Issues (风险与边缘)
- **Issue R3-1**: Task 1.1 风险 — PyAPITestHarness 在 fixed source 模式下可能对 k-eff 进行不恰当的比较。fixed source 的 statepoint 无 k-eff，harness 需要正确处理。 → 已在 Task 1.1 潜在风险中详述 ✅ 已修正
- **Issue R3-2**: 跨 Task 交互 — Phase 4 的文档修改假设 Phase 0 成功。如果 Phase 0 编译失败，Phase 4 中的代码示例仍然有效（纯 Python API 不需要编译），但"限制条件"部分可能需要更新。 → 风险可接受，不需要额外处理 ✅ 已修正（标记为已知风险）
- **Issue R3-3**: 遗漏场景 — Point detector 在 void region 中（检测器位于 void cell 内）的行为未测试。`score_point_tally_impl` 中 `if (distance < total_distance) continue` 逻辑在 void 中可能跳过所有 scoring（因为光线可能穿透 void 无碰撞）。 → 已在 Error & Rescue Map 中标注 void 场景 ✅ 已修正

## Pre-Delivery Audit (Level: L1-Lite)

| §   | Check            | Status | Note                                                      |
| --- | ---------------- | ------ | --------------------------------------------------------- |
| 1   | Unit consistency | ✅ PASS | 所有能量值均使用 eV 单位制，距离使用 cm（与 OpenMC 一致） |

Auditor: Plan Architect | Date: 2026-04-13
