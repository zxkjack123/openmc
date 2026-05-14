# PR #3757 Comment Draft

> This file is a draft for the GitHub PR comment. Copy the content below the line into the PR comment box.

---

## Contributed: Regression Tests, Unit Tests, Docs & Bug Investigation for Point Detectors

Hi @GuySten, thanks for the great work on point detectors! I've been testing the NEE implementation on my branch and have prepared several contributions that may help move this PR toward merge-readiness. Here's a summary:

### 1. Regression Tests (2 new test suites)

**`tests/regression_tests/point_detector/`** — Basic point detector test:
- Isotropic 1 MeV source at center of 5 cm water sphere + 30 cm void shell
- 2 detectors at (0,0,20) and (20,0,0), R0 = 1 cm
- PointFilter × EnergyFilter, flux score, 10 batches × 1000 particles
- Also includes negative tests for MG mode and non-vacuum BC rejection

**`tests/regression_tests/point_detector_multi/`** — Multi-detector test:
- Cylindrical iron geometry (r=10 cm, h=60 cm)
- 3 detectors along z-axis at z = 10, 30, 50 cm
- Verifies monotonically decreasing flux with depth through iron
- Results: 2.45e-3 → 5.44e-4 → 4.31e-5 (physically reasonable iron attenuation)

### 2. Unit Test for PointFilter Python API

**`tests/unit_tests/test_filter_point.py`** — 7 tests covering:
- Construction, invalid bins rejection, XML roundtrip, HDF5 roundtrip
- Tally integration, equality operator, pandas DataFrame output
- Also added `PointFilter.from_xml_element()` classmethod in `openmc/filter.py`

### 3. Bug Investigation (shimwell's crash)

Re: @shimwell's `free(): corrupted unsorted chunks` report — I ran extensive ASAN testing (10k particles on both simple and complex geometries) on your branch with `-fsanitize=address` and could **not reproduce** the crash. The crash was reported on `itay-space/openmc:deploy` which has a different code base.

I also analyzed the `ParticleRay : public Ray, public Particle` diamond inheritance via `virtual public GeometryState` — confirmed a single `GeometryState` subobject exists (sizeof test shows 928 bytes < 1176 naive sum), so the diamond inheritance is safe. Minor note: `GeometryState` lacks a virtual destructor, but since these objects are stack-allocated, this is low risk.

### 4. User Guide Documentation

Added a "Point Detectors (Next-Event Estimator)" section to `docs/source/usersguide/tallies.rst` including:
- NEE formula (LaTeX math)
- Python code examples (basic + energy-filtered)
- Comprehensive limitations list (vacuum BC only, flux only, no MG, no NCrystal, etc.)

Also added comprehensive numpydoc docstrings to `PointFilter` class and all its methods.

### How to integrate

These changes are on my `point-detector-contrib` branch. I can:
1. Open a separate PR targeting your `point-detector` branch, or
2. Provide patch files you can apply directly, or
3. Whatever workflow you prefer

Let me know how you'd like to proceed! Happy to adjust anything based on your feedback.

### Test output

All tests pass:
```
tests/unit_tests/test_filter_point.py — 7 passed
tests/regression_tests/point_detector/ — passed
tests/regression_tests/point_detector_multi/ — passed
```

---

@paulromano — If you have a moment, I'd appreciate your perspective on the remaining architectural considerations for this PR (e.g., the estimator-vs-tally-type discussion from #3109). The PR chain is progressing well (3/6 merged: #3550, #3816, #3845), and with regression tests, unit tests, and user docs now available, I believe this is approaching merge-readiness. Happy to address any concerns.
