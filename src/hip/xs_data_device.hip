#include "openmc/hip/xs_data_device.h"

#include <hip/hip_runtime.h>

#include "openmc/error.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"

#include <fmt/format.h>

#include <cstdint>
#include <numeric>
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

struct DeviceXSLookupData {
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

  // Total bytes allocated on device
  size_t total_bytes = 0;
};

// Global device data instance
static DeviceXSLookupData device_xs;

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
    fatal_error(fmt::format(
      "hipMalloc failed ({} bytes): {}", sz, hipGetErrorString(err)));
  }
  err = hipMemcpy(d_ptr, host_data, sz, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fatal_error(fmt::format(
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

  device_xs.total_bytes = bytes;

  // Synchronize to ensure all copies complete
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    fatal_error(fmt::format(
      "hipDeviceSynchronize failed: {}", hipGetErrorString(err)));
  }

  // Print summary
  double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
  fmt::print(" Copied {} nuclides ({} temperature entries) to device "
             "({:.1f} MB)\n",
    n_nuc, total_entries, mb);
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

  device_xs = DeviceXSLookupData {};

  fmt::print(" Freed device XS data\n");
}

} // namespace hip
} // namespace openmc
