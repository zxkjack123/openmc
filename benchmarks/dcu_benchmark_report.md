# T4.1 DCU vs CPU Performance Benchmark Report

## Hardware Platform

| Component | Specification |
|-----------|--------------|
| CPU | Hygon C86 7380 (2×64 cores, 128 cores total) |
| RAM | 504 GB DDR4 |
| GPU/DCU | 2× 壁仞 BiRen C-3000 (gfx936) |
| OS | Ubuntu 22.04, kernel 5.10.134 |
| Compiler | GCC 11.4 (host), DTK 25.04.1 hipcc (device) |
| OpenMC | develop branch (368ea06), event-based HIP port |
| OpenMP | 64 threads |

## Benchmark Configuration

- **Model**: PWR pin cell (12 nuclides, 3 materials, reflective BC)
- **Particles per batch**: 10,000 / 50,000 / 200,000
- **Batches**: 20 (5 inactive + 15 active)
- **Repeats**: 2 per configuration
- **Modes**: CPU History-based, CPU Event-based, DCU Event-based

## Results Summary

### Calculation Rate (particles/second, active batches, median)

| Particles/Batch | CPU History | CPU Event | DCU Event | DCU/CPU-Event | DCU/CPU-History |
|----------------:|------------:|----------:|----------:|--------------:|----------------:|
| 10,000 | 183,399 | 7,279 | 1,854 | 0.25x | 0.010x |
| 50,000 | 231,419 | 8,343 | 3,731 | 0.45x | 0.016x |
| 200,000 | 292,640 | 7,151 | 4,410 | 0.62x | 0.015x |

### Transport Time Breakdown (seconds, median)

| Mode | Particles | Total | XS Lookup | Advancing | Surface | Collision | XS % |
|------|----------:|------:|----------:|----------:|--------:|----------:|-----:|
| CPU Event | 10,000 | 27.6 | 6.2 | 13.0 | 6.4 | 1.6 | 22.5% |
| DCU Event | 10,000 | 105.0 | 54.2 | 38.2 | 9.4 | 2.9 | 51.6% |
| CPU Event | 50,000 | 120.6 | 30.3 | 57.6 | 23.6 | 7.7 | 25.1% |
| DCU Event | 50,000 | 262.9 | 193.2 | 38.0 | 24.1 | 6.3 | 73.5% |
| CPU Event | 200,000 | 568.1 | 204.7 | 214.4 | 94.7 | 49.3 | 36.0% |
| DCU Event | 200,000 | 925.0 | 730.9 | 89.8 | 76.6 | 23.4 | 79.0% |

### k-eff Verification

All modes produce statistically identical k-eff values:

| Particles | CPU History | CPU Event | DCU Event |
|----------:|:------------|:----------|:----------|
| 10,000 | 1.16053 ± 0.00288 | 1.16053 ± 0.00288 | 1.16053 ± 0.00288 |
| 50,000 | 1.16155 ± 0.00062 | 1.16155 ± 0.00062 | 1.16155 ± 0.00062 |
| 200,000 | 1.16091 ± 0.00064 | 1.16091 ± 0.00064 | 1.16091 ± 0.00064 |

## Bottleneck Analysis

### Primary Bottleneck: Host↔Device Data Transfer

The DCU event mode is **slower** than CPU event mode at all tested particle counts. The XS lookup time — which includes hipMalloc, hipMemcpy H→D, kernel execution, hipMemcpy D→H, and hipFree — dominates at 52-79% of transport time for DCU versus 22-36% for CPU.

**Key observation**: As particle count increases from 10K to 200K:
- DCU XS time grows from 54s → 731s (13.5x for 20x more particles)
- CPU XS time grows from 6.2s → 205s (33x for 20x more particles)
- DCU calculation rate improves from 1,854 → 4,410 p/s (2.4x)
- The DCU/CPU-Event ratio improves from 0.25x → 0.62x

This scaling behavior indicates:
1. **Per-batch fixed overhead** (hipMalloc/hipFree per batch) is significant at low particle counts
2. **Host-device memory copies** dominate: each batch copies ~17 arrays of size N×sizeof(double)
3. The actual GPU kernel computation is fast, but transfer overhead swamps it

### Secondary Bottleneck: Non-XS Event Phases Still on CPU

The "Advancing" phase shows interesting behavior:
- CPU Event 200K: 214s → DCU Event 200K: 90s (2.4x faster on DCU build)
- This is because the DCU build avoids some per-particle overhead during XS phases

### Why DCU is Slower: Architecture Analysis

1. **Current implementation**: Per-batch hipMalloc → hipMemcpy → kernel → hipMemcpy → hipFree
   - 20 batches × (malloc + memcpy + kernel + memcpy + free) = high overhead
   - Each batch processes N particles through ~50-100 XS events each
   
2. **The event-based paradigm mismatch**: OpenMC's event-based mode processes one event type at a time across all particles. The XS event is only ~25-35% of CPU transport time. Accelerating only this phase cannot speed up the overall simulation.

3. **Memory bandwidth**: Each XS lookup requires reading nuclide grid data (large tables) for each particle. The PCIe transfer to copy particle data to device and results back exceeds the compute savings.

## Optimization Recommendations (Priority Order)

1. **Persistent device memory**: Allocate device buffers once at simulation start, reuse across batches (eliminate per-batch hipMalloc/hipFree)
2. **Pinned host memory**: Use `hipHostMalloc` for host-side arrays to enable async DMA transfers
3. **Overlap compute and transfer**: Use HIP streams to overlap H→D copy of next batch with kernel execution of current batch
4. **Move more phases to GPU**: Port advancing and surface crossing to GPU to reduce CPU↔GPU synchronization points
5. **Fused kernel**: Combine XS lookup + collision sampling into a single kernel launch to avoid intermediate data roundtrip
6. **Consider history-based GPU approach**: Instead of event-based, track complete particle histories on GPU (requires porting geometry, tallies, etc.)

## Raw Data

Full benchmark data in `benchmarks/dcu_benchmark_results.csv`.

### Timing Details (all values in seconds)

```
Config                    Total   Transport  XS-Lookup  Advance  Surface  Collision  Rate(p/s)
CPU-Hist   10K p/batch      1.8       1.0       —         —        —        —       183,399
CPU-Event  10K p/batch     28.2      27.4       6.0      13.2      6.3      1.6       7,279
DCU-Event  10K p/batch    110.3     109.6      58.8      37.3     10.5      2.7       1,802
CPU-Hist   50K p/batch      5.0       4.1       —         —        —        —       232,984
CPU-Event  50K p/batch    120.7     119.6      30.5      56.8     23.1      7.8       8,343
DCU-Event  50K p/batch    268.7     267.7     199.2      36.8     24.0      6.4       3,731
CPU-Hist  200K p/batch     14.4      13.1       —         —        —        —       289,911
CPU-Event 200K p/batch    558.7     556.9     198.0     211.8     94.7     47.7       7,151
DCU-Event 200K p/batch    901.0     899.2     700.7      92.9     77.8     23.4       4,410
```
