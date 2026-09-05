# HIP/DCU 是否继续 Port 到最新 Upstream — 决策评估报告

- **Task**: PM #3274 (OPENMC-REPOSITORY)
- **Date**: 2026-09-05
- **Decision**: **暂缓 port（DEFER）**，满足触发条件后按分波重建方案重启

## 1. 决策结论

**暂缓**（不是放弃）。理由链：

1. 最新 upstream 基线（`e3bba4f1c`）已就位，官方 NNDC 索引已恢复，
   `fixed_source_micro` 案例已在 NNDC + STRICT_FP 下闭环——验证基础设施
   首次齐备；
2. 但 HIP 代码的正确性证据（逐值 XS kernel 对照，#3243）与性能证据
   （公平 2-rank CPU 对照，#3242）**均未执行**——在此之前任何 port 都是
   在未验证地基上搬运 180 个提交；
3. port 的技术成本很高：本地 ahead 180 / upstream ahead 126，
   `event.cpp`/`CMakeLists.txt`/`particle` 大量漂移，共享文件必须手工
   重写（见 §4）。

**放弃条件**（未来满足任一即正式放弃）：#3243 发现非统计性 XS 偏差且
两轮修复失败；或 #3242 证明 DCU 在公平配置下无任何场景收益且瓶颈
分析（#3244）无改进路径。

## 2. 证据基础（A–E 路线，全部 done）

| 证据 | 任务 | 状态 | 对本决策的作用 |
|---|---|---|---|
| 官方 NNDC 索引恢复（MD5 2d007730…） | #3271 | ✅ done | 正确性验证的数据前提 |
| fixed_source_micro 案例 + NNDC/STRICT_FP reference | #3271 | ✅ done | HIP XS kernel 的未来对照输入 |
| provenance/export contract | #3272 | ✅ done | HIP 结果的溯源接口 |
| FCCB pointer manifest + selection 对接 | #3275/#3276 | ✅ done | HIP 结果的跨代码登记通道 |
| 环境指纹落地 | #3277 | ✅ done | HIP benchmark 可复现性前提 |

## 3. 既有 HIP 研究结论（#3242–#3246 立项时的证据，未执行）

- 早期报告：单卡 DCU 慢于 CPU-event（50K: 3731 vs 8343 p/s）；
- 后期报告：2-GPU 声称超过 CPU，但缺 2-rank CPU-only 对照，归因不成立；
- persistent buffer 仅 10K 场景 XS 阶段局部 25–29% 改善，200K 不稳定；
- stream pipeline 当前 HEAD 无收益证据（报告称回退 69% 但机制未证实）；
- 无逐值 XS 对照 → 正确性仅有 k-eff 统计一致，不足以支撑 port。
- 定位（用户已采纳）：异构移植可行性研究与负结果/瓶颈分析。

## 4. Port 技术评估（若重启）

| 组 | 内容 | 迁移方式 |
|---|---|---|
| G1 | HIP build 抽象 | 手工 port（CMake 已大改） |
| G2 | device XS data framework | 重建 |
| G3 | XS kernel | 重建 + #3243 逐值验证前置 |
| G4 | event.cpp 接入 | **手工 port，最高风险**（shared secondary bank 漂移） |
| G5 | persistent buffer / pinned arena | 分开重建，扩容不变式需补断言 |
| G6 | 2-slot / 2-stream | 分波，先单 stream 正确性后 overlap |
| PD | point-detector / get_pdf | **不混入**，独立轨道 |

基线：`sync/upstream-develop-2026-09-02`（= e3bba4f1c，0/0）。
分支拓扑：新功能分支从该基线切出；`feature/hip-xs-buffer-reuse`
保留为代码来源归档，不改写历史。

## 5. 重启触发条件（全部满足才启动）

1. #3243 逐值 XS kernel 对照完成，误差统计在预设容差内，未覆盖路径
   （multipole/URR/S(α,β) fallback）清单明确；
2. #3242 公平 4 配置基准完成，DCU 贡献与 MPI 分片贡献分离；
3. #3244 XS 阶段分项计时完成（allocation/transfer/kernel 占比明确）；
4. #3245 功能边界评估完成（以 e3bba4f1c 为准，非旧基线）；
5. #3246 报告修订完成，旧报告过强结论已撤回。

## 6. 建议顺序与责任划分

```text
#3243（正确性）→ #3242（公平性能）→ #3244（瓶颈）
     ↓ 三者通过后
#3245（新 upstream 边界评估）→ G1→G4 分波重建（新任务）
     ↓
#3246（报告修订，合并新旧证据）
```

#3242–#3246 归属 `fusion-neutronics-toolchain`（id=137），workspace
`/home/gw/opt/openmc`；分波重建新任务届时另立（归属 OPENMC-REPOSITORY）。

## 7. 与旧结论的一致性

本报告与 #3271 结论"HIP 代码有工程价值但性能未证实"一致；与
critical-thinking 两轮评审的"按功能重建、不整体 merge/rebase"一致。
无冲突。

## 8. 交付物界定

- ✅ 本报告：暂缓决策 + 触发条件 + 迁移方案骨架；
- ⏸ 实际 port/merge/cherry-pick：不在本任务范围（触发条件满足后另立任务）。
