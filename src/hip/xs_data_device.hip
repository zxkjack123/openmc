#include "openmc/hip/xs_data_device.h"
#include "openmc/hip/calculate_xs_kernel.h"

#include <hip/hip_runtime.h>

#include "openmc/error.h"
#include "openmc/material.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/settings.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace openmc {
namespace hip {

//==============================================================================
// Device-side flat cross section data structure
//==============================================================================

// The host Nuclide stores data per temperature as nested vectors:
//   grid_[t].energy    — vector<double> of energy grid points
//   grid_[t].grid_index — vector<int>   log-grid acceleration indices
//   xs_[t]             — Tensor<double> shape (n_energy, 5) row-major
//
// We flatten all nuclide-temperature data into contiguous arrays with
// per-entry offset tables for GPU access.

struct DeviceXSData {
  int n_nuclides = 0;
  int n_entries = 0; // total nuclide-temperature pairs

  // Per-nuclide metadata (size: n_nuclides)
  int* d_n_temps = nullptr;
  int* d_entry_start = nullptr; // first entry index for this nuclide
  double* d_awr = nullptr;
  int* d_fissionable = nullptr;

  // Per-nuclide temperature values (concatenated, size: sum of n_temps)
  double* d_kTs = nullptr;
  int* d_kTs_offset = nullptr; // offset into d_kTs per nuclide

  // Per-entry (nuclide-temp pair) offset tables
  int64_t* d_energy_offset = nullptr;
  int* d_energy_size = nullptr;
  int64_t* d_grid_index_offset = nullptr;
  int* d_grid_index_size = nullptr;
  int64_t* d_xs_offset = nullptr;

  // Concatenated data arrays
  double* d_energy = nullptr;   // all energy grids
  int* d_grid_index = nullptr;  // all grid_index arrays
  double* d_xs = nullptr;       // all XS data (5 channels per energy point)

  // Material data (flattened)
  int n_materials = 0;
  int* d_mat_n_nuclides = nullptr;     // number of nuclides per material
  int* d_mat_nuclide_offset = nullptr; // offset into flat nuclide/density arrays
  int* d_mat_nuclide_ids = nullptr;    // flattened nuclide indices
  double* d_mat_atom_density = nullptr; // flattened atom densities

  // Total bytes allocated on device
  size_t total_bytes = 0;

  // Maximum number of nuclides in any material
  int max_nucs_per_mat = 0;
};

// Global device data instance
static DeviceXSData device_xs;

//==============================================================================
// Persistent per-event I/O buffer pool (refactor: eliminates per-call
// hipMalloc/hipFree on the calculate_xs_*_on_device hot path)
//==============================================================================

struct DeviceIOBuffers {
  // Inputs (sized N)
  double* d_energy = nullptr;
  double* d_sqrtkT = nullptr;
  double* d_density_mult = nullptr;
  int* d_material = nullptr;
  int* d_i_log_union = nullptr;

  // Macro outputs (sized N)
  double* d_macro_total = nullptr;
  double* d_macro_abs = nullptr;
  double* d_macro_fis = nullptr;
  double* d_macro_nufis = nullptr;

  // Micro outputs (sized N * max_nucs)
  double* d_mt = nullptr;
  double* d_ma = nullptr;
  double* d_mf = nullptr;
  double* d_mnf = nullptr;
  double* d_mpp = nullptr;
  double* d_mi = nullptr;
  int* d_mig = nullptr;
  int* d_mit = nullptr;

  // Slot B — second-half of 2-slot double-buffering scheme (see T2.1/T2.2).
  // Each slot is independently grown; per-slot capacity for the 2-slot path
  // is ceil(N/2) so total VRAM stays neutral vs the single-slot layout.
  double* d_energy_b = nullptr;
  double* d_sqrtkT_b = nullptr;
  double* d_density_mult_b = nullptr;
  int* d_material_b = nullptr;
  int* d_i_log_union_b = nullptr;
  double* d_macro_total_b = nullptr;
  double* d_macro_abs_b = nullptr;
  double* d_macro_fis_b = nullptr;
  double* d_macro_nufis_b = nullptr;
  double* d_mt_b = nullptr;
  double* d_ma_b = nullptr;
  double* d_mf_b = nullptr;
  double* d_mnf_b = nullptr;
  double* d_mpp_b = nullptr;
  double* d_mi_b = nullptr;
  int* d_mig_b = nullptr;
  int* d_mit_b = nullptr;

  int capacity_n = 0;            // slot A capacity (N for single-slot path)
  int capacity_max_nucs = 0;
  int capacity_n_b = 0;          // slot B capacity (independent grow)

  // Two HIP streams for chunked async pipelining (T2.2 wrapper rewrite).
  hipStream_t streams[2] = {nullptr, nullptr};
  bool streams_initialized = false;

  void ensure_capacity(int n, int max_nucs);
  void ensure_capacity_2slot(int n_total, int max_nucs);
  void init_streams();
  void free_io_buffers();
};

static DeviceIOBuffers io_buf;

static constexpr double kIOBufGrowFactor = 1.5;

template<typename T>
static void grow_buffer(T*& p, size_t new_count)
{
  if (p) {
    hipFree(p);
    p = nullptr;
  }
  if (new_count == 0)
    return;
  hipError_t e = hipMalloc(&p, new_count * sizeof(T));
  if (e != hipSuccess) {
    openmc::fatal_error(fmt::format("hipMalloc {} elems ({} bytes) failed: {}",
      new_count, new_count * sizeof(T), hipGetErrorString(e)));
  }
}

void DeviceIOBuffers::ensure_capacity(int n, int max_nucs)
{
  bool need_n = (n > capacity_n);
  bool need_m = (max_nucs > capacity_max_nucs);
  if (!need_n && !need_m)
    return;

  int new_n = need_n
    ? std::max<int>(static_cast<int>(std::ceil(n * kIOBufGrowFactor)), n)
    : capacity_n;
  int new_m = need_m
    ? std::max<int>(
        static_cast<int>(std::ceil(max_nucs * kIOBufGrowFactor)), max_nucs)
    : capacity_max_nucs;

  if (need_n) {
    grow_buffer(d_energy, new_n);
    grow_buffer(d_sqrtkT, new_n);
    grow_buffer(d_density_mult, new_n);
    grow_buffer(d_material, new_n);
    grow_buffer(d_i_log_union, new_n);
    grow_buffer(d_macro_total, new_n);
    grow_buffer(d_macro_abs, new_n);
    grow_buffer(d_macro_fis, new_n);
    grow_buffer(d_macro_nufis, new_n);
  }
  if (need_n || need_m) {
    size_t new_micro = static_cast<size_t>(new_n) * new_m;
    grow_buffer(d_mt, new_micro);
    grow_buffer(d_ma, new_micro);
    grow_buffer(d_mf, new_micro);
    grow_buffer(d_mnf, new_micro);
    grow_buffer(d_mpp, new_micro);
    grow_buffer(d_mi, new_micro);
    grow_buffer(d_mig, new_micro);
    grow_buffer(d_mit, new_micro);
  }

  capacity_n = new_n;
  capacity_max_nucs = new_m;
}

void DeviceIOBuffers::init_streams()
{
  if (streams_initialized)
    return;
  hipError_t e0 = hipStreamCreate(&streams[0]);
  hipError_t e1 = hipStreamCreate(&streams[1]);
  if (e0 != hipSuccess || e1 != hipSuccess) {
    openmc::fatal_error(fmt::format(
      "hipStreamCreate failed: stream0={} stream1={}",
      hipGetErrorString(e0), hipGetErrorString(e1)));
  }
  streams_initialized = true;
}

void DeviceIOBuffers::ensure_capacity_2slot(int n_total, int max_nucs)
{
  // Per-slot capacity is ceil(n_total / 2) so the two-chunk pipeline keeps
  // total VRAM neutral compared to the single-slot ensure_capacity(n_total).
  int n_per_slot = (n_total + 1) / 2;

  // Slot A reuses the existing fields and ensure_capacity grow policy.
  ensure_capacity(n_per_slot, max_nucs);

  // Slot B grows independently with the same 1.5x policy.
  bool need_n = (n_per_slot > capacity_n_b);
  bool need_m = (max_nucs > capacity_max_nucs);
  if (need_n || need_m) {
    int new_n_b = need_n
      ? std::max<int>(
          static_cast<int>(std::ceil(n_per_slot * kIOBufGrowFactor)), n_per_slot)
      : capacity_n_b;
    int new_m = capacity_max_nucs; // already grown above to cover max_nucs

    if (need_n) {
      grow_buffer(d_energy_b, new_n_b);
      grow_buffer(d_sqrtkT_b, new_n_b);
      grow_buffer(d_density_mult_b, new_n_b);
      grow_buffer(d_material_b, new_n_b);
      grow_buffer(d_i_log_union_b, new_n_b);
      grow_buffer(d_macro_total_b, new_n_b);
      grow_buffer(d_macro_abs_b, new_n_b);
      grow_buffer(d_macro_fis_b, new_n_b);
      grow_buffer(d_macro_nufis_b, new_n_b);
    }
    size_t new_micro_b = static_cast<size_t>(new_n_b) * new_m;
    grow_buffer(d_mt_b, new_micro_b);
    grow_buffer(d_ma_b, new_micro_b);
    grow_buffer(d_mf_b, new_micro_b);
    grow_buffer(d_mnf_b, new_micro_b);
    grow_buffer(d_mpp_b, new_micro_b);
    grow_buffer(d_mi_b, new_micro_b);
    grow_buffer(d_mig_b, new_micro_b);
    grow_buffer(d_mit_b, new_micro_b);

    capacity_n_b = new_n_b;
  }

  init_streams();
}

void DeviceIOBuffers::free_io_buffers()
{
  auto F = [](auto*& p) {
    if (p) {
      hipFree(p);
      p = nullptr;
    }
  };
  F(d_energy);
  F(d_sqrtkT);
  F(d_density_mult);
  F(d_material);
  F(d_i_log_union);
  F(d_macro_total);
  F(d_macro_abs);
  F(d_macro_fis);
  F(d_macro_nufis);
  F(d_mt);
  F(d_ma);
  F(d_mf);
  F(d_mnf);
  F(d_mpp);
  F(d_mi);
  F(d_mig);
  F(d_mit);
  // Slot B
  F(d_energy_b);
  F(d_sqrtkT_b);
  F(d_density_mult_b);
  F(d_material_b);
  F(d_i_log_union_b);
  F(d_macro_total_b);
  F(d_macro_abs_b);
  F(d_macro_fis_b);
  F(d_macro_nufis_b);
  F(d_mt_b);
  F(d_ma_b);
  F(d_mf_b);
  F(d_mnf_b);
  F(d_mpp_b);
  F(d_mi_b);
  F(d_mig_b);
  F(d_mit_b);
  capacity_n = 0;
  capacity_max_nucs = 0;
  capacity_n_b = 0;

  // Destroy HIP streams if they were created.
  if (streams_initialized) {
    if (streams[0])
      hipStreamDestroy(streams[0]);
    if (streams[1])
      hipStreamDestroy(streams[1]);
    streams[0] = streams[1] = nullptr;
    streams_initialized = false;
  }
}

//==============================================================================
// Host staging arena (pinned host memory pool) — feeds hipMemcpyAsync paths.
// Each named slot grows with the same 1.5x policy as DeviceIOBuffers. If
// hipHostMalloc fails (e.g. hipErrorOutOfMemory) the slot transparently falls
// back to a pageable std::malloc'd block and logs a warning; the caller never
// sees the difference and execution continues (transfers just lose DMA speed).
//==============================================================================

struct HostStagingArena {
  struct Slot {
    void* ptr = nullptr;
    size_t bytes = 0;
    bool pinned = false;
  };
  std::unordered_map<std::string, Slot> slots_;

  void* alloc_bytes(const char* slot_name, size_t bytes);
  void free_all();
};

static HostStagingArena host_arena;

void* HostStagingArena::alloc_bytes(const char* slot_name, size_t bytes)
{
  if (bytes == 0)
    return nullptr;

  std::string key(slot_name);
  Slot& s = slots_[key];
  if (s.ptr && s.bytes >= bytes)
    return s.ptr;

  size_t new_bytes = std::max<size_t>(
    static_cast<size_t>(std::ceil(bytes * kIOBufGrowFactor)), bytes);

  if (s.ptr) {
    if (s.pinned)
      hipHostFree(s.ptr);
    else
      std::free(s.ptr);
    s.ptr = nullptr;
    s.bytes = 0;
  }

  void* p = nullptr;
  hipError_t e = hipHostMalloc(&p, new_bytes, hipHostMallocDefault);
  if (e == hipSuccess) {
    s.ptr = p;
    s.bytes = new_bytes;
    s.pinned = true;
  } else {
    // Pinned allocation failed — fall back to pageable. Common cause is
    // hipErrorOutOfMemory on many-rank nodes where pinned budget is tight.
    fmt::print(stderr,
      " WARN: host_arena pinned alloc failed for slot '{}' ({} bytes): {} "
      "— falling back to pageable\n",
      key, new_bytes, hipGetErrorString(e));
    p = std::malloc(new_bytes);
    if (!p) {
      openmc::fatal_error(fmt::format(
        "host_arena pageable fallback malloc failed for slot '{}' ({} bytes)",
        key, new_bytes));
    }
    s.ptr = p;
    s.bytes = new_bytes;
    s.pinned = false;
  }
  return s.ptr;
}

void HostStagingArena::free_all()
{
  for (auto& kv : slots_) {
    Slot& s = kv.second;
    if (!s.ptr)
      continue;
    if (s.pinned)
      hipHostFree(s.ptr);
    else
      std::free(s.ptr);
    s.ptr = nullptr;
    s.bytes = 0;
  }
  slots_.clear();
}

double* host_alloc_double(const char* slot_name, std::size_t n)
{
  return static_cast<double*>(
    host_arena.alloc_bytes(slot_name, n * sizeof(double)));
}

int* host_alloc_int(const char* slot_name, std::size_t n)
{
  return static_cast<int*>(
    host_arena.alloc_bytes(slot_name, n * sizeof(int)));
}

void host_arena_free()
{
  host_arena.free_all();
}

//==============================================================================
// Helper: allocate and copy
//==============================================================================

template<typename T>
static T* hip_alloc_copy(const T* host_data, size_t count, size_t& bytes)
{
  if (count == 0)
    return nullptr;
  T* d_ptr = nullptr;
  size_t sz = count * sizeof(T);
  hipError_t err = hipMalloc(&d_ptr, sz);
  if (err != hipSuccess) {
    openmc::fatal_error(fmt::format(
      "hipMalloc failed ({} bytes): {}", sz, hipGetErrorString(err)));
  }
  err = hipMemcpy(d_ptr, host_data, sz, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    openmc::fatal_error(fmt::format(
      "hipMemcpy failed ({} bytes): {}", sz, hipGetErrorString(err)));
  }
  bytes += sz;
  return d_ptr;
}

template<typename T>
static T* hip_alloc_copy(const std::vector<T>& v, size_t& bytes)
{
  return hip_alloc_copy(v.data(), v.size(), bytes);
}

//==============================================================================
// copy_xs_data_to_device
//==============================================================================

void copy_xs_data_to_device()
{
  // Select GPU device based on MPI rank (for multi-GPU runs)
  int device_count = 0;
  hipGetDeviceCount(&device_count);
  if (device_count > 0) {
    int device_id = mpi::rank % device_count;
    hipSetDevice(device_id);
    fmt::print(" MPI rank {} using HIP device {}/{}\n", mpi::rank, device_id,
      device_count);
  }

  const auto& nuclides = data::nuclides;
  int n_nuc = static_cast<int>(nuclides.size());
  if (n_nuc == 0)
    return;

  // Count total entries (nuclide-temperature pairs)
  int total_entries = 0;
  for (int i = 0; i < n_nuc; ++i) {
    total_entries += static_cast<int>(nuclides[i]->kTs_.size());
  }

  // Build host-side offset tables
  std::vector<int> h_n_temps(n_nuc);
  std::vector<int> h_entry_start(n_nuc);
  std::vector<double> h_awr(n_nuc);
  std::vector<int> h_fissionable(n_nuc);

  // kT arrays
  std::vector<double> h_kTs_all;
  std::vector<int> h_kTs_offset(n_nuc);

  // Per-entry offsets and sizes
  std::vector<int64_t> h_energy_offset(total_entries);
  std::vector<int> h_energy_size(total_entries);
  std::vector<int64_t> h_grid_index_offset(total_entries);
  std::vector<int> h_grid_index_size(total_entries);
  std::vector<int64_t> h_xs_offset(total_entries);

  // Concatenated data
  std::vector<double> h_energy_all;
  std::vector<int> h_grid_index_all;
  std::vector<double> h_xs_all;

  int entry_idx = 0;
  for (int i = 0; i < n_nuc; ++i) {
    const auto& nuc = *nuclides[i];
    int nt = static_cast<int>(nuc.kTs_.size());
    h_n_temps[i] = nt;
    h_entry_start[i] = entry_idx;
    h_awr[i] = nuc.awr_;
    h_fissionable[i] = nuc.fissionable_ ? 1 : 0;

    h_kTs_offset[i] = static_cast<int>(h_kTs_all.size());
    for (int t = 0; t < nt; ++t) {
      h_kTs_all.push_back(nuc.kTs_[t]);
    }

    for (int t = 0; t < nt; ++t) {
      const auto& grid = nuc.grid_[t];
      const auto& xs = nuc.xs_[t];

      // Energy grid
      h_energy_offset[entry_idx] = static_cast<int64_t>(h_energy_all.size());
      h_energy_size[entry_idx] = static_cast<int>(grid.energy.size());
      h_energy_all.insert(
        h_energy_all.end(), grid.energy.begin(), grid.energy.end());

      // Grid index (log-grid acceleration)
      h_grid_index_offset[entry_idx] =
        static_cast<int64_t>(h_grid_index_all.size());
      h_grid_index_size[entry_idx] =
        static_cast<int>(grid.grid_index.size());
      h_grid_index_all.insert(
        h_grid_index_all.end(), grid.grid_index.begin(),
        grid.grid_index.end());

      // XS data (stored row-major: n_energy × 5)
      h_xs_offset[entry_idx] = static_cast<int64_t>(h_xs_all.size());
      size_t xs_count = xs.size(); // n_energy * 5
      const double* xs_data = xs.data();
      h_xs_all.insert(h_xs_all.end(), xs_data, xs_data + xs_count);

      ++entry_idx;
    }
  }

  // Allocate and copy to device
  size_t bytes = 0;

  device_xs.n_nuclides = n_nuc;
  device_xs.n_entries = total_entries;

  device_xs.d_n_temps = hip_alloc_copy(h_n_temps, bytes);
  device_xs.d_entry_start = hip_alloc_copy(h_entry_start, bytes);
  device_xs.d_awr = hip_alloc_copy(h_awr, bytes);
  device_xs.d_fissionable = hip_alloc_copy(h_fissionable, bytes);

  device_xs.d_kTs = hip_alloc_copy(h_kTs_all, bytes);
  device_xs.d_kTs_offset = hip_alloc_copy(h_kTs_offset, bytes);

  device_xs.d_energy_offset = hip_alloc_copy(h_energy_offset, bytes);
  device_xs.d_energy_size = hip_alloc_copy(h_energy_size, bytes);
  device_xs.d_grid_index_offset = hip_alloc_copy(h_grid_index_offset, bytes);
  device_xs.d_grid_index_size = hip_alloc_copy(h_grid_index_size, bytes);
  device_xs.d_xs_offset = hip_alloc_copy(h_xs_offset, bytes);

  device_xs.d_energy = hip_alloc_copy(h_energy_all, bytes);
  device_xs.d_grid_index = hip_alloc_copy(h_grid_index_all, bytes);
  device_xs.d_xs = hip_alloc_copy(h_xs_all, bytes);

  // === Material data ===
  const auto& materials = model::materials;
  int n_mat = static_cast<int>(materials.size());

  std::vector<int> h_mat_n_nuclides(n_mat);
  std::vector<int> h_mat_nuclide_offset(n_mat);
  std::vector<int> h_mat_nuclide_ids;
  std::vector<double> h_mat_atom_density;

  for (int m = 0; m < n_mat; ++m) {
    const auto& mat = *materials[m];
    auto nuclide_span = mat.nuclides();
    auto density_span = mat.densities();
    int nn = static_cast<int>(nuclide_span.size());

    h_mat_n_nuclides[m] = nn;
    h_mat_nuclide_offset[m] = static_cast<int>(h_mat_nuclide_ids.size());

    for (int j = 0; j < nn; ++j) {
      h_mat_nuclide_ids.push_back(nuclide_span[j]);
      h_mat_atom_density.push_back(density_span[j]);
    }
  }

  device_xs.n_materials = n_mat;
  device_xs.d_mat_n_nuclides = hip_alloc_copy(h_mat_n_nuclides, bytes);
  device_xs.d_mat_nuclide_offset = hip_alloc_copy(h_mat_nuclide_offset, bytes);
  device_xs.d_mat_nuclide_ids = hip_alloc_copy(h_mat_nuclide_ids, bytes);
  device_xs.d_mat_atom_density = hip_alloc_copy(h_mat_atom_density, bytes);

  // Compute max nuclides per material
  int max_nucs = 0;
  for (int m = 0; m < n_mat; ++m) {
    if (h_mat_n_nuclides[m] > max_nucs)
      max_nucs = h_mat_n_nuclides[m];
  }
  device_xs.max_nucs_per_mat = max_nucs;

  device_xs.total_bytes = bytes;

  // Synchronize to ensure all copies complete
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    openmc::fatal_error(fmt::format(
      "hipDeviceSynchronize failed: {}", hipGetErrorString(err)));
  }

  // Print summary
  double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
  fmt::print(" Copied {} nuclides ({} temperature entries), {} materials to "
             "device ({:.1f} MB)\n",
    n_nuc, total_entries, n_mat, mb);

  // Pre-size persistent I/O buffer pool based on expected event-mode demand.
  // 2-slot allocation: each slot gets ceil(max/2) capacity for streamed
  // double-buffering (T2.1); existing single-slot wrapper still uses slot A.
  io_buf.ensure_capacity_2slot(
    static_cast<int>(settings::max_particles_in_flight),
    device_xs.max_nucs_per_mat);
}

//==============================================================================
// free_xs_data_on_device
//==============================================================================

static void hip_free_safe(void* ptr)
{
  if (ptr)
    hipFree(ptr);
}

void free_xs_data_on_device()
{
  hip_free_safe(device_xs.d_n_temps);
  hip_free_safe(device_xs.d_entry_start);
  hip_free_safe(device_xs.d_awr);
  hip_free_safe(device_xs.d_fissionable);
  hip_free_safe(device_xs.d_kTs);
  hip_free_safe(device_xs.d_kTs_offset);
  hip_free_safe(device_xs.d_energy_offset);
  hip_free_safe(device_xs.d_energy_size);
  hip_free_safe(device_xs.d_grid_index_offset);
  hip_free_safe(device_xs.d_grid_index_size);
  hip_free_safe(device_xs.d_xs_offset);
  hip_free_safe(device_xs.d_energy);
  hip_free_safe(device_xs.d_grid_index);
  hip_free_safe(device_xs.d_xs);
  hip_free_safe(device_xs.d_mat_n_nuclides);
  hip_free_safe(device_xs.d_mat_nuclide_offset);
  hip_free_safe(device_xs.d_mat_nuclide_ids);
  hip_free_safe(device_xs.d_mat_atom_density);

  device_xs = DeviceXSData {};

  io_buf.free_io_buffers();
  host_arena_free();

  fmt::print(" Freed device XS data\n");
}

//==============================================================================
// XS lookup kernel
//==============================================================================

// Device-side binary search matching openmc::lower_bound_index from search.h.
// Returns index i such that arr[i] <= val < arr[i+1].
// Equivalent to: std::lower_bound(arr, arr+n, val) - arr - 1
// with special case: if arr[0] == val, returns 0.
__device__ int device_lower_bound_index(const double* arr, int n, double val)
{
  if (arr[0] == val)
    return 0;

  // Binary search for first element >= val (equivalent to std::lower_bound)
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (arr[mid] < val)
      lo = mid + 1;
    else
      hi = mid;
  }
  // lo is now the index of the first element >= val
  return lo - 1;
}

__global__ void calculate_xs_kernel(int n_particles,
  const double* __restrict__ d_energy, const double* __restrict__ d_sqrtkT,
  const int* __restrict__ d_material, const int* __restrict__ d_i_log_union,
  double* __restrict__ d_macro_total, double* __restrict__ d_macro_absorption,
  double* __restrict__ d_macro_fission,
  double* __restrict__ d_macro_nu_fission,
  // XS data
  const int* __restrict__ nuc_n_temps, const int* __restrict__ nuc_entry_start,
  const int* __restrict__ nuc_fissionable, const double* __restrict__ nuc_kTs,
  const int* __restrict__ nuc_kTs_offset,
  const int64_t* __restrict__ entry_energy_offset,
  const int* __restrict__ entry_energy_size,
  const int64_t* __restrict__ entry_grid_index_offset,
  const double* __restrict__ all_energy,
  const int* __restrict__ all_grid_index, const double* __restrict__ all_xs,
  const int64_t* __restrict__ entry_xs_offset,
  // Material data
  const int* __restrict__ mat_n_nuclides,
  const int* __restrict__ mat_nuclide_offset,
  const int* __restrict__ mat_nuclide_ids,
  const double* __restrict__ mat_atom_density)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_particles)
    return;

  int mat_idx = d_material[tid];
  if (mat_idx < 0) {
    d_macro_total[tid] = 0.0;
    d_macro_absorption[tid] = 0.0;
    d_macro_fission[tid] = 0.0;
    d_macro_nu_fission[tid] = 0.0;
    return;
  }

  double E = d_energy[tid];
  double sqrtkT = d_sqrtkT[tid];
  double kT = sqrtkT * sqrtkT;
  int i_log_union = d_i_log_union[tid];

  double macro_total = 0.0;
  double macro_absorption = 0.0;
  double macro_fission = 0.0;
  double macro_nu_fission = 0.0;

  int n_nuc = mat_n_nuclides[mat_idx];
  int nuc_offset = mat_nuclide_offset[mat_idx];

  for (int j = 0; j < n_nuc; ++j) {
    int i_nuclide = mat_nuclide_ids[nuc_offset + j];
    double atom_dens = mat_atom_density[nuc_offset + j];

    // Temperature index (nearest method)
    int nt = nuc_n_temps[i_nuclide];
    int kts_off = nuc_kTs_offset[i_nuclide];
    int i_temp = 0;
    if (nt > 1) {
      double max_diff = 1.0e30;
      for (int t = 0; t < nt; ++t) {
        double diff = fabs(nuc_kTs[kts_off + t] - kT);
        if (diff < max_diff) {
          i_temp = t;
          max_diff = diff;
        }
      }
    }

    // Get entry index for (nuclide, i_temp)
    int entry = nuc_entry_start[i_nuclide] + i_temp;

    // Energy grid lookup
    int64_t e_off = entry_energy_offset[entry];
    int e_size = entry_energy_size[entry];
    const double* e_grid = &all_energy[e_off];

    int i_grid;
    if (E < e_grid[0]) {
      i_grid = 0;
    } else if (E > e_grid[e_size - 1]) {
      i_grid = e_size - 2;
    } else {
      // Log-grid acceleration
      int64_t gi_off = entry_grid_index_offset[entry];
      int i_low = all_grid_index[gi_off + i_log_union];
      int i_high = all_grid_index[gi_off + i_log_union + 1] + 1;
      if (i_low < 0)
        i_low = 0;
      if (i_high > e_size)
        i_high = e_size;

      // Binary search in reduced range
      const double* sub = &e_grid[i_low];
      int sub_n = i_high - i_low;
      i_grid = i_low + device_lower_bound_index(sub, sub_n, E);
    }

    // Check for rare case where two energy points are the same
    if (i_grid < e_size - 1 && e_grid[i_grid] == e_grid[i_grid + 1])
      ++i_grid;

    // Interpolation factor
    double f = (E - e_grid[i_grid]) / (e_grid[i_grid + 1] - e_grid[i_grid]);

    // XS interpolation (row-major: n_energy × 5)
    // Channels: 0=total, 1=absorption, 2=fission, 3=nu_fission, 4=photon_prod
    int64_t xs_off = entry_xs_offset[entry];
    const double* xs_lo = &all_xs[xs_off + i_grid * 5];
    const double* xs_hi = &all_xs[xs_off + (i_grid + 1) * 5];

    double micro_total = (1.0 - f) * xs_lo[0] + f * xs_hi[0];
    double micro_absorption = (1.0 - f) * xs_lo[1] + f * xs_hi[1];

    macro_total += atom_dens * micro_total;
    macro_absorption += atom_dens * micro_absorption;

    if (nuc_fissionable[i_nuclide]) {
      double micro_fission = (1.0 - f) * xs_lo[2] + f * xs_hi[2];
      double micro_nu_fission = (1.0 - f) * xs_lo[3] + f * xs_hi[3];
      macro_fission += atom_dens * micro_fission;
      macro_nu_fission += atom_dens * micro_nu_fission;
    }
  }

  d_macro_total[tid] = macro_total;
  d_macro_absorption[tid] = macro_absorption;
  d_macro_fission[tid] = macro_fission;
  d_macro_nu_fission[tid] = macro_nu_fission;
}

//==============================================================================
// launch_calculate_xs_kernel
//==============================================================================

void launch_calculate_xs_kernel(int n_particles, const double* d_energy,
  const double* d_sqrtkT, const int* d_material, const int* d_i_log_union,
  double* d_macro_total, double* d_macro_absorption, double* d_macro_fission,
  double* d_macro_nu_fission, hipStream_t stream)
{
  if (n_particles == 0)
    return;

  const int block_size = 256;
  int grid_size = (n_particles + block_size - 1) / block_size;

  calculate_xs_kernel<<<grid_size, block_size, 0, stream>>>(n_particles, d_energy,
    d_sqrtkT, d_material, d_i_log_union, d_macro_total, d_macro_absorption,
    d_macro_fission, d_macro_nu_fission,
    // XS data
    device_xs.d_n_temps, device_xs.d_entry_start, device_xs.d_fissionable,
    device_xs.d_kTs, device_xs.d_kTs_offset, device_xs.d_energy_offset,
    device_xs.d_energy_size, device_xs.d_grid_index_offset, device_xs.d_energy,
    device_xs.d_grid_index, device_xs.d_xs, device_xs.d_xs_offset,
    // Material data
    device_xs.d_mat_n_nuclides, device_xs.d_mat_nuclide_offset,
    device_xs.d_mat_nuclide_ids, device_xs.d_mat_atom_density);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    openmc::fatal_error(
      fmt::format("HIP kernel launch failed: {}", hipGetErrorString(err)));
  }
}

//==============================================================================
// calculate_xs_on_device (host-callable wrapper)
//==============================================================================

void calculate_xs_on_device(int n_particles, const double* h_energy,
  const double* h_sqrtkT, const int* h_material, const int* h_i_log_union,
  double* h_macro_total, double* h_macro_absorption, double* h_macro_fission,
  double* h_macro_nu_fission)
{
  if (n_particles == 0)
    return;

  io_buf.ensure_capacity(n_particles, device_xs.max_nucs_per_mat);

  size_t n_d = n_particles * sizeof(double);
  size_t n_i = n_particles * sizeof(int);

  // Copy inputs to device (buffers persist across calls)
  hipMemcpy(io_buf.d_energy, h_energy, n_d, hipMemcpyHostToDevice);
  hipMemcpy(io_buf.d_sqrtkT, h_sqrtkT, n_d, hipMemcpyHostToDevice);
  hipMemcpy(io_buf.d_material, h_material, n_i, hipMemcpyHostToDevice);
  hipMemcpy(io_buf.d_i_log_union, h_i_log_union, n_i, hipMemcpyHostToDevice);

  // Launch kernel
  launch_calculate_xs_kernel(n_particles, io_buf.d_energy, io_buf.d_sqrtkT,
    io_buf.d_material, io_buf.d_i_log_union, io_buf.d_macro_total,
    io_buf.d_macro_abs, io_buf.d_macro_fis, io_buf.d_macro_nufis);

  // Synchronize
  hipDeviceSynchronize();

  // Copy results back
  hipMemcpy(
    h_macro_total, io_buf.d_macro_total, n_d, hipMemcpyDeviceToHost);
  hipMemcpy(
    h_macro_absorption, io_buf.d_macro_abs, n_d, hipMemcpyDeviceToHost);
  hipMemcpy(h_macro_fission, io_buf.d_macro_fis, n_d, hipMemcpyDeviceToHost);
  hipMemcpy(
    h_macro_nu_fission, io_buf.d_macro_nufis, n_d, hipMemcpyDeviceToHost);
}

//==============================================================================
// Extended kernel: macro XS + per-nuclide micro XS + density_mult
//==============================================================================

__global__ void calculate_xs_kernel_full(int n_particles,
  const double* __restrict__ d_energy, const double* __restrict__ d_sqrtkT,
  const int* __restrict__ d_material, const int* __restrict__ d_i_log_union,
  const double* __restrict__ d_density_mult,
  double* __restrict__ d_macro_total, double* __restrict__ d_macro_absorption,
  double* __restrict__ d_macro_fission,
  double* __restrict__ d_macro_nu_fission,
  int max_nucs,
  double* __restrict__ d_micro_total, double* __restrict__ d_micro_abs,
  double* __restrict__ d_micro_fis, double* __restrict__ d_micro_nufis,
  double* __restrict__ d_micro_pprod, double* __restrict__ d_micro_interp,
  int* __restrict__ d_micro_igrid, int* __restrict__ d_micro_itemp,
  // XS data
  const int* __restrict__ nuc_n_temps, const int* __restrict__ nuc_entry_start,
  const int* __restrict__ nuc_fissionable, const double* __restrict__ nuc_kTs,
  const int* __restrict__ nuc_kTs_offset,
  const int64_t* __restrict__ entry_energy_offset,
  const int* __restrict__ entry_energy_size,
  const int64_t* __restrict__ entry_grid_index_offset,
  const double* __restrict__ all_energy,
  const int* __restrict__ all_grid_index, const double* __restrict__ all_xs,
  const int64_t* __restrict__ entry_xs_offset,
  // Material data
  const int* __restrict__ mat_n_nuclides,
  const int* __restrict__ mat_nuclide_offset,
  const int* __restrict__ mat_nuclide_ids,
  const double* __restrict__ mat_atom_density)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_particles)
    return;

  int mat_idx = d_material[tid];
  if (mat_idx < 0) {
    d_macro_total[tid] = 0.0;
    d_macro_absorption[tid] = 0.0;
    d_macro_fission[tid] = 0.0;
    d_macro_nu_fission[tid] = 0.0;
    return;
  }

  double E = d_energy[tid];
  double sqrtkT = d_sqrtkT[tid];
  double kT = sqrtkT * sqrtkT;
  int i_log_union = d_i_log_union[tid];
  double dens_mult = d_density_mult[tid];

  double macro_total = 0.0;
  double macro_absorption = 0.0;
  double macro_fission = 0.0;
  double macro_nu_fission = 0.0;

  int n_nuc = mat_n_nuclides[mat_idx];
  int nuc_offset = mat_nuclide_offset[mat_idx];
  int micro_base = tid * max_nucs;

  for (int j = 0; j < n_nuc; ++j) {
    int i_nuclide = mat_nuclide_ids[nuc_offset + j];
    double atom_dens = mat_atom_density[nuc_offset + j] * dens_mult;

    // Temperature index (nearest method)
    int nt = nuc_n_temps[i_nuclide];
    int kts_off = nuc_kTs_offset[i_nuclide];
    int i_temp = 0;
    if (nt > 1) {
      double max_diff = 1.0e30;
      for (int t = 0; t < nt; ++t) {
        double diff = fabs(nuc_kTs[kts_off + t] - kT);
        if (diff < max_diff) {
          i_temp = t;
          max_diff = diff;
        }
      }
    }

    // Get entry index for (nuclide, i_temp)
    int entry = nuc_entry_start[i_nuclide] + i_temp;

    // Energy grid lookup
    int64_t e_off = entry_energy_offset[entry];
    int e_size = entry_energy_size[entry];
    const double* e_grid = &all_energy[e_off];

    int i_grid;
    if (E < e_grid[0]) {
      i_grid = 0;
    } else if (E > e_grid[e_size - 1]) {
      i_grid = e_size - 2;
    } else {
      // Log-grid acceleration
      int64_t gi_off = entry_grid_index_offset[entry];
      int i_low = all_grid_index[gi_off + i_log_union];
      int i_high = all_grid_index[gi_off + i_log_union + 1] + 1;
      if (i_low < 0)
        i_low = 0;
      if (i_high > e_size)
        i_high = e_size;

      // Binary search in reduced range
      const double* sub = &e_grid[i_low];
      int sub_n = i_high - i_low;
      i_grid = i_low + device_lower_bound_index(sub, sub_n, E);
    }

    // Check for rare case where two energy points are the same
    if (i_grid < e_size - 1 && e_grid[i_grid] == e_grid[i_grid + 1])
      ++i_grid;

    // Interpolation factor
    double f = (E - e_grid[i_grid]) / (e_grid[i_grid + 1] - e_grid[i_grid]);

    // XS interpolation (row-major: n_energy × 5)
    int64_t xs_off = entry_xs_offset[entry];
    const double* xs_lo = &all_xs[xs_off + i_grid * 5];
    const double* xs_hi = &all_xs[xs_off + (i_grid + 1) * 5];

    double micro_total = (1.0 - f) * xs_lo[0] + f * xs_hi[0];
    double micro_absorption = (1.0 - f) * xs_lo[1] + f * xs_hi[1];
    double micro_fission = 0.0;
    double micro_nu_fission = 0.0;
    double micro_pprod = (1.0 - f) * xs_lo[4] + f * xs_hi[4];

    if (nuc_fissionable[i_nuclide]) {
      micro_fission = (1.0 - f) * xs_lo[2] + f * xs_hi[2];
      micro_nu_fission = (1.0 - f) * xs_lo[3] + f * xs_hi[3];
    }

    macro_total += atom_dens * micro_total;
    macro_absorption += atom_dens * micro_absorption;
    macro_fission += atom_dens * micro_fission;
    macro_nu_fission += atom_dens * micro_nu_fission;

    // Write per-nuclide micro XS output
    int mi = micro_base + j;
    d_micro_total[mi] = micro_total;
    d_micro_abs[mi] = micro_absorption;
    d_micro_fis[mi] = micro_fission;
    d_micro_nufis[mi] = micro_nu_fission;
    d_micro_pprod[mi] = micro_pprod;
    d_micro_interp[mi] = f;
    d_micro_igrid[mi] = i_grid;
    d_micro_itemp[mi] = i_temp;
  }

  d_macro_total[tid] = macro_total;
  d_macro_absorption[tid] = macro_absorption;
  d_macro_fission[tid] = macro_fission;
  d_macro_nu_fission[tid] = macro_nu_fission;
}

//==============================================================================
// calculate_xs_full_on_device (host-callable wrapper with micro XS output)
//==============================================================================

void calculate_xs_full_on_device(int n_particles, int max_nucs,
  const double* h_energy, const double* h_sqrtkT, const int* h_material,
  const int* h_i_log_union, const double* h_density_mult,
  double* h_macro_total, double* h_macro_absorption, double* h_macro_fission,
  double* h_macro_nu_fission, double* h_micro_total, double* h_micro_abs,
  double* h_micro_fis, double* h_micro_nufis, double* h_micro_pprod,
  double* h_micro_interp, int* h_micro_igrid, int* h_micro_itemp)
{
  if (n_particles == 0)
    return;

  // 2-slot: each slot gets ceil(n/2) capacity for streamed double-buffering
  io_buf.ensure_capacity_2slot(n_particles, max_nucs);

  int chunk_a = n_particles / 2;
  int chunk_b = n_particles - chunk_a;

  size_t n_d_a = chunk_a * sizeof(double);
  size_t n_i_a = chunk_a * sizeof(int);
  size_t n_d_b = chunk_b * sizeof(double);
  size_t n_i_b = chunk_b * sizeof(int);
  size_t micro_d_a = static_cast<size_t>(chunk_a) * max_nucs * sizeof(double);
  size_t micro_i_a = static_cast<size_t>(chunk_a) * max_nucs * sizeof(int);
  size_t micro_d_b = static_cast<size_t>(chunk_b) * max_nucs * sizeof(double);
  size_t micro_i_b = static_cast<size_t>(chunk_b) * max_nucs * sizeof(int);
  constexpr int kBlockSize = 256;

  // --- Chunk A (first half) → slot A, stream 0 ---
  if (chunk_a > 0) {
    hipMemcpyAsync(io_buf.d_energy,       h_energy,       n_d_a, hipMemcpyHostToDevice, io_buf.streams[0]);
    hipMemcpyAsync(io_buf.d_sqrtkT,       h_sqrtkT,       n_d_a, hipMemcpyHostToDevice, io_buf.streams[0]);
    hipMemcpyAsync(io_buf.d_density_mult, h_density_mult,  n_d_a, hipMemcpyHostToDevice, io_buf.streams[0]);
    hipMemcpyAsync(io_buf.d_material,     h_material,      n_i_a, hipMemcpyHostToDevice, io_buf.streams[0]);
    hipMemcpyAsync(io_buf.d_i_log_union,  h_i_log_union,   n_i_a, hipMemcpyHostToDevice, io_buf.streams[0]);

    int grid_a = (chunk_a + kBlockSize - 1) / kBlockSize;
    calculate_xs_kernel_full<<<grid_a, kBlockSize, 0, io_buf.streams[0]>>>(chunk_a,
      io_buf.d_energy, io_buf.d_sqrtkT, io_buf.d_material, io_buf.d_i_log_union,
      io_buf.d_density_mult, io_buf.d_macro_total, io_buf.d_macro_abs,
      io_buf.d_macro_fis, io_buf.d_macro_nufis, max_nucs, io_buf.d_mt,
      io_buf.d_ma, io_buf.d_mf, io_buf.d_mnf, io_buf.d_mpp, io_buf.d_mi,
      io_buf.d_mig, io_buf.d_mit,
      device_xs.d_n_temps, device_xs.d_entry_start, device_xs.d_fissionable,
      device_xs.d_kTs, device_xs.d_kTs_offset, device_xs.d_energy_offset,
      device_xs.d_energy_size, device_xs.d_grid_index_offset, device_xs.d_energy,
      device_xs.d_grid_index, device_xs.d_xs, device_xs.d_xs_offset,
      device_xs.d_mat_n_nuclides, device_xs.d_mat_nuclide_offset,
      device_xs.d_mat_nuclide_ids, device_xs.d_mat_atom_density);

    hipError_t err_launch_a = hipGetLastError();
    if (err_launch_a != hipSuccess) {
      openmc::fatal_error(
        fmt::format("HIP full kernel launch failed (chunk A): {}", hipGetErrorString(err_launch_a)));
    }

    hipMemcpyAsync(h_macro_total,       io_buf.d_macro_total,  n_d_a, hipMemcpyDeviceToHost, io_buf.streams[0]);
    hipMemcpyAsync(h_macro_absorption,  io_buf.d_macro_abs,    n_d_a, hipMemcpyDeviceToHost, io_buf.streams[0]);
    hipMemcpyAsync(h_macro_fission,     io_buf.d_macro_fis,    n_d_a, hipMemcpyDeviceToHost, io_buf.streams[0]);
    hipMemcpyAsync(h_macro_nu_fission,  io_buf.d_macro_nufis,  n_d_a, hipMemcpyDeviceToHost, io_buf.streams[0]);
    hipMemcpyAsync(h_micro_total,       io_buf.d_mt,           micro_d_a, hipMemcpyDeviceToHost, io_buf.streams[0]);
    hipMemcpyAsync(h_micro_abs,         io_buf.d_ma,           micro_d_a, hipMemcpyDeviceToHost, io_buf.streams[0]);
    hipMemcpyAsync(h_micro_fis,         io_buf.d_mf,           micro_d_a, hipMemcpyDeviceToHost, io_buf.streams[0]);
    hipMemcpyAsync(h_micro_nufis,       io_buf.d_mnf,          micro_d_a, hipMemcpyDeviceToHost, io_buf.streams[0]);
    hipMemcpyAsync(h_micro_pprod,       io_buf.d_mpp,          micro_d_a, hipMemcpyDeviceToHost, io_buf.streams[0]);
    hipMemcpyAsync(h_micro_interp,      io_buf.d_mi,           micro_d_a, hipMemcpyDeviceToHost, io_buf.streams[0]);
    hipMemcpyAsync(h_micro_igrid,       io_buf.d_mig,          micro_i_a, hipMemcpyDeviceToHost, io_buf.streams[0]);
    hipMemcpyAsync(h_micro_itemp,       io_buf.d_mit,          micro_i_a, hipMemcpyDeviceToHost, io_buf.streams[0]);
  }

  // --- Chunk B (second half) → slot B, stream 1 ---
  if (chunk_b > 0) {
    hipMemcpyAsync(io_buf.d_energy_b,       h_energy + chunk_a,       n_d_b, hipMemcpyHostToDevice, io_buf.streams[1]);
    hipMemcpyAsync(io_buf.d_sqrtkT_b,       h_sqrtkT + chunk_a,       n_d_b, hipMemcpyHostToDevice, io_buf.streams[1]);
    hipMemcpyAsync(io_buf.d_density_mult_b, h_density_mult + chunk_a,  n_d_b, hipMemcpyHostToDevice, io_buf.streams[1]);
    hipMemcpyAsync(io_buf.d_material_b,     h_material + chunk_a,      n_i_b, hipMemcpyHostToDevice, io_buf.streams[1]);
    hipMemcpyAsync(io_buf.d_i_log_union_b,  h_i_log_union + chunk_a,   n_i_b, hipMemcpyHostToDevice, io_buf.streams[1]);

    int grid_b = (chunk_b + kBlockSize - 1) / kBlockSize;
    calculate_xs_kernel_full<<<grid_b, kBlockSize, 0, io_buf.streams[1]>>>(chunk_b,
      io_buf.d_energy_b, io_buf.d_sqrtkT_b, io_buf.d_material_b, io_buf.d_i_log_union_b,
      io_buf.d_density_mult_b, io_buf.d_macro_total_b, io_buf.d_macro_abs_b,
      io_buf.d_macro_fis_b, io_buf.d_macro_nufis_b, max_nucs, io_buf.d_mt_b,
      io_buf.d_ma_b, io_buf.d_mf_b, io_buf.d_mnf_b, io_buf.d_mpp_b, io_buf.d_mi_b,
      io_buf.d_mig_b, io_buf.d_mit_b,
      device_xs.d_n_temps, device_xs.d_entry_start, device_xs.d_fissionable,
      device_xs.d_kTs, device_xs.d_kTs_offset, device_xs.d_energy_offset,
      device_xs.d_energy_size, device_xs.d_grid_index_offset, device_xs.d_energy,
      device_xs.d_grid_index, device_xs.d_xs, device_xs.d_xs_offset,
      device_xs.d_mat_n_nuclides, device_xs.d_mat_nuclide_offset,
      device_xs.d_mat_nuclide_ids, device_xs.d_mat_atom_density);

    hipError_t err_launch_b = hipGetLastError();
    if (err_launch_b != hipSuccess) {
      openmc::fatal_error(
        fmt::format("HIP full kernel launch failed (chunk B): {}", hipGetErrorString(err_launch_b)));
    }

    // D2H output — destination host pointers offset by chunk_a for macro,
    // by chunk_a * max_nucs for micro (particle-major layout).
    hipMemcpyAsync(h_macro_total + chunk_a,      io_buf.d_macro_total_b, n_d_b, hipMemcpyDeviceToHost, io_buf.streams[1]);
    hipMemcpyAsync(h_macro_absorption + chunk_a, io_buf.d_macro_abs_b,   n_d_b, hipMemcpyDeviceToHost, io_buf.streams[1]);
    hipMemcpyAsync(h_macro_fission + chunk_a,    io_buf.d_macro_fis_b,   n_d_b, hipMemcpyDeviceToHost, io_buf.streams[1]);
    hipMemcpyAsync(h_macro_nu_fission + chunk_a, io_buf.d_macro_nufis_b, n_d_b, hipMemcpyDeviceToHost, io_buf.streams[1]);
    hipMemcpyAsync(h_micro_total + chunk_a * max_nucs,  io_buf.d_mt_b,  micro_d_b, hipMemcpyDeviceToHost, io_buf.streams[1]);
    hipMemcpyAsync(h_micro_abs + chunk_a * max_nucs,    io_buf.d_ma_b,  micro_d_b, hipMemcpyDeviceToHost, io_buf.streams[1]);
    hipMemcpyAsync(h_micro_fis + chunk_a * max_nucs,    io_buf.d_mf_b,  micro_d_b, hipMemcpyDeviceToHost, io_buf.streams[1]);
    hipMemcpyAsync(h_micro_nufis + chunk_a * max_nucs,  io_buf.d_mnf_b, micro_d_b, hipMemcpyDeviceToHost, io_buf.streams[1]);
    hipMemcpyAsync(h_micro_pprod + chunk_a * max_nucs,  io_buf.d_mpp_b, micro_d_b, hipMemcpyDeviceToHost, io_buf.streams[1]);
    hipMemcpyAsync(h_micro_interp + chunk_a * max_nucs, io_buf.d_mi_b,  micro_d_b, hipMemcpyDeviceToHost, io_buf.streams[1]);
    hipMemcpyAsync(h_micro_igrid + chunk_a * max_nucs,  io_buf.d_mig_b, micro_i_b, hipMemcpyDeviceToHost, io_buf.streams[1]);
    hipMemcpyAsync(h_micro_itemp + chunk_a * max_nucs,  io_buf.d_mit_b, micro_i_b, hipMemcpyDeviceToHost, io_buf.streams[1]);
  }

  // Sync both streams and check for errors
  if (chunk_a > 0) {
    hipError_t sync_err = hipStreamSynchronize(io_buf.streams[0]);
    if (sync_err != hipSuccess) {
      openmc::fatal_error(
        fmt::format("hipStreamSynchronize failed (stream 0): {}", hipGetErrorString(sync_err)));
    }
  }
  if (chunk_b > 0) {
    hipError_t sync_err = hipStreamSynchronize(io_buf.streams[1]);
    if (sync_err != hipSuccess) {
      openmc::fatal_error(
        fmt::format("hipStreamSynchronize failed (stream 1): {}", hipGetErrorString(sync_err)));
    }
  }
}

//==============================================================================
// get_max_nuclides_per_material
//==============================================================================

int get_max_nuclides_per_material()
{
  return device_xs.max_nucs_per_mat;
}

} // namespace hip
} // namespace openmc
