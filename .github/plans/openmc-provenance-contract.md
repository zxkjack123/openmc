# OpenMC fixed-source micro case — Provenance / Export Contract

- **Task**: PM #3272 (OPENMC-REPOSITORY)
- **Status**: v0.1 draft — input for FCCB #3275 (wrapper/manifest alignment)
- **Date**: 2026-09-03

## 1. Purpose and scope

This contract defines what the OpenMC side **guarantees to export** for the
`fixed_source_micro` regression case, and how downstream consumers (notably the
`fusion-cross-code-benchmarks` project, FCCB) must reference it. It is a
description of an interface, **not** a second implementation of OpenMC's test
harness.

Out of scope for this case (by design):

- physical parity claims or cross-code "truth" production;
- HIP/DCU execution;
- point-detector / get_pdf logic;
- large model benchmarks.

The case is a **smoke + regression drift** test. `results_true.dat` is an
OpenMC self-regression reference only. It MUST NOT be used as a cross-code
oracle.

## 2. Case identity

| Field | Value |
|---|---|
| case id | `openmc-fixed-source-micro` |
| tier | `micro` |
| purpose | `smoke` / `drift` |
| code | `openmc` |
| runtime budget | < 30 s per run |
| particles / batches | 50 / 5, fixed source |
| temperature | 300 K default (single value) |
| physics content | O16 + U238 sphere, flux tally, leakage |

## 3. OpenMC-side deliverables

The following are the only artifacts the OpenMC repo promises to maintain for
this case:

1. **Test body** (in-repo, under version control):

   ```
   tests/regression_tests/fixed_source_micro/__init__.py
   tests/regression_tests/fixed_source_micro/test.py
   tests/regression_tests/fixed_source_micro/inputs_true.dat
   tests/regression_tests/fixed_source_micro/results_true.dat
   ```

2. **Entry command** (from repo root):

   ```bash
   OMP_NUM_THREADS=2 python -m pytest \
     tests/regression_tests/fixed_source_micro/test.py -v
   ```

   Reference files are generated / regenerated **only** with `--update` and
   only under the standard environment (see §4).

3. **Comparison semantics**: `comparison_mode = exact` — byte-for-byte
   comparison of `inputs_true.dat` and `results_true.dat`, performed by
   OpenMC's own `PyAPITestHarness`. FCCB must not reimplement this
   comparison; it consumes the pytest exit code.

4. **Results summary fields** (what the case exposes, format fixed by the
   harness):

   ```
   tally 1:
   <sum flux, format 12.6E>
   <sum_sq flux, format 12.6E>
   ```

## 4. Standard environment contract

The reference values are only meaningful under this environment:

| Item | Value |
|---|---|
| OpenMC commit | `6b36aeaa3` (feature/openmc-fixed-source-micro; parent 2ea24730e) |
| build | `-DOPENMC_ENABLE_STRICT_FP=on`, RelWithDebInfo, MPI off, DAGMC off |
| compiler | GNU 13.3.0 |
| cross sections | official NNDC HDF5 index, MD5 `2d00773012eda670bc9f95d96a31c989` |
| env script | `source /home/gw/NucData/env-openmc-nndc.sh` |
| threads | `OMP_NUM_THREADS=2` |

Alternative library note: FENDL-3.2 (`env-openmc-fendl32.sh`) is the machine
default but is **not** the reference environment for this case. Reference
regeneration requires the NNDC environment above.

## 5. Contract for downstream consumers (FCCB #3275)

A FCCB wrapper manifest for this case must:

- reference the OpenMC repo **path + commit** (git content-addressing), not a
  copy of `results_true.dat`;
- store `entry_command` as the pytest command in §3;
- store `comparison_mode: exact` and delegate the comparison to the OpenMC
  harness;
- reference `env_fingerprint_ref` instead of inlining environment fields
  (avoid double-bookkeeping with FCCB #3250 `environment_fingerprint.py`);
- declare `reference.authority = openmc_git` (the reference file lives in the
  OpenMC repo, not on NAS);
- include `tier/purpose/runtime_budget_s/particles` fields for FCCB case
  selection (#3250 `select_cases.py`);
- record the nuclear-data logical id + MD5 as provenance, never as a physics
  truth.

## 6. Open questions for FCCB alignment

1. Whether `manifest.schema.yaml` (FCCB #3248) accepts a case-level extension
   block (e.g. `extensions.openmc.*`) or needs a `$ref`.
2. Whether `entry_command` is stored as argv list or string.
3. Which `env_fingerprint_ref` format #3250 emits (id? hash?).
