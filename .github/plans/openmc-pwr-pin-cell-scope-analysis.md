# PWR Pin-Cell 案例口径分析（S(α,β) / k-eff 跨代码语义）

- **Task**: PM #3273 (OPENMC-REPOSITORY)
- **Date**: 2026-09-05
- **Status**: 口径分析 v0.1 — 先于任何案例实现/登记

## 1. 结论先行

1. **k-eff 与 S(α,β) 均不可作为跨代码对比真值**。本案例若进入 FCCB，只允许
   `comparison.mode = code-to-code`（互比差异报告），禁止 `reference` /
   `experiment` 语义下的"真值"判定。
2. 案例适合作为 **physics.code-to-code 的 PWR 热谱案例**，而不是
   OpenMC 仓库内的 regression 参考（OpenMC 仓库内 k-eff 回归无跨代码含义，
   且 `pwr_pin_cell` 示例已被 15+ 个回归测试——`mgxs_library_*`、
   `random_ray_*`、`model_xml` 等——复用为输入模型，再新增独立
   pin-cell regression 属重复建设）。
3. 实现评估：OpenMC 侧已有 `openmc.examples.pwr_pin_cell()`（BEAVRS
   beginning-of-cycle 2.4 w/o UO₂），但默认 100 粒子/10 批仅作 smoke；
   跨代码互比需要**独立高统计计算脚本**（≥200 active batches、≥10⁵
   particles/batch、inactive 充分），不应复用示例的默认设置。

## 2. 案例物理口径

| 项 | 口径 | 来源 |
|---|---|---|
| 燃料 | 2.4 w/o 富集 UO₂，10.29769 g/cm³ | BEAVRS BOC |
| 包壳 | 天然 Zr 合金（5 同位素），6.55 g/cm³ | BEAVRS |
| 慢化剂 | 含硼热水 0.740582 g/cm³，**含 S(α,β) c_H_in_H2O** | BEAVRS |
| 几何 | 2D pin-cell，反射边界，pitch 由 openmc 常量给定 | `openmc/examples.py` |
| 问题类型 | k-eigenvalue | — |

## 3. S(α,β) 跨代码口径差异（为什么不能当真值）

1. **数据源不同**：
   - OpenMC 侧 NNDC HDF5 的 `c_H_in_H2O` 来自 ENDF/B-VII.1 热散射子库；
   - MCNP6 侧对应 `lwtr`（或 `h/z` 系列）S(α,β) 表来自 MCNP6_DATA 的 ACE 库，
     加工器与评价版本可能不同（ENDF/B-VII.0/VII.1 差异、TJS 加工差异）。
2. **温度网格不同**：热散射表温度点（如 293.6/300/350/600 K）两库并不一致；
   案例温度 300 K 时两侧可能落在不同插值区间。
3. **后果**：热中子区（<1 eV）能谱与吸收率出现系统性偏移，进而影响
   `k-eff`、硼吸收与四因子分解。这类偏移是**数据源差异**而非"谁对谁错"。

## 4. k-eff 跨代码口径差异（为什么不能当真值）

1. **核素截面库不同**：NNDC（ENDF/B-VII.1）vs MCNP6_DATA（多数为
   ENDF/B-VII.0/VII.1 的 LANL 加工 ACE）。U-235/U-238 共振区与硼截面存在
   已知数百 pcm 级差异。
2. **未分辨共振区（URR）处理不同**：OpenMC 与 MCNP 概率表实现不同。
3. **统计波动**：k-eff 比较必须报告 ±σ 与置信区间；无统计带宽的
   "差异 100 pcm"不构成结论。
4. **文献参考带**：BEAVRS 这类 pin-cell 的文献值（如 1.17–1.19 区间）
   可作为**合理性参考带**，不是逐位真值；偏离文献带外才触发调查。

## 5. 若登记 FCCB：必须遵守的语义约束

- `comparison.mode: code-to-code`；
- 结果标注差异 + 双方统计不确定度，不产生"真值"；
- S(α,β) 与 URR 处理差异必须在 manifest metadata 或 validation 记录中声明；
- 参考带用途（文献量级）单独字段，与互比差异报告分离；
- 不写入 OpenMC 仓库 regression reference（避免把跨代码差异误变成
  OpenMC 自回归门槛）。

## 6. 实现评估

| 方案 | 评估 |
|---|---|
| A. 复用 `openmc.examples.pwr_pin_cell()` 直接互比 | ❌ 默认 100 粒子统计不足，且内置 plot 需剥离 |
| B. 新写 OpenMC 高统计脚本 + MCNP6 等价输入，在 FCCB 登记 | ✅ 推荐；OpenMC 侧仅需一个独立脚本（如 `benchmarks/` 或 FCCB case 目录），不改 OpenMC 源码 |
| C. 在 OpenMC 仓库新增 regression | ⚠️ 无跨代码含义，与既有 `pincell` 回归重叠，不推荐 |

推荐 B：口径分析先行已满足；具体脚本与 MCNP6 输入属于 FCCB
`physics.code-to-code` 登记工作，另立任务。

## 7. 未决问题（留给实现任务）

1. MCNP6 侧 S(α,β) 表选择（lwtr 温度点）与 OpenMC c_H_in_H2O 的精确对齐策略；
2. 统计规格（batch 数、每批粒子数、inactive 数、随机种子约定）；
3. 参考带取值来源（BEAVRS 文献 vs SINBAD 类实验值）与偏离触发阈值。

## 8. 本任务交付物界定

- ✅ 口径分析（本文档）：S(α,β)/k-eff 不可作真值，语义约束与方案评估；
- ⏸ 案例脚本/MCNP 输入/登记：不在本任务范围（另立 FCCB 任务）。
