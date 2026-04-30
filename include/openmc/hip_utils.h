#ifndef OPENMC_HIP_UTILS_H
#define OPENMC_HIP_UTILS_H

#ifdef OPENMC_USE_HIP

// Only include HIP runtime and use __host__/__device__ annotations when
// compiled by hipcc (clang with HIP support). GCC-compiled .cpp files get
// empty macros even when OPENMC_USE_HIP is defined.
#if defined(__HIPCC__) || defined(__HIP__)

#include <hip/hip_runtime.h>

#include "openmc/error.h"

#include <fmt/core.h>

#define OPENMC_HOST_DEVICE __host__ __device__
#define OPENMC_DEVICE __device__
#define OPENMC_GLOBAL __global__

// HIP error checking macro (host-only)
#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      openmc::fatal_error(fmt::format(                                         \
        "HIP error: {} at {}:{}", hipGetErrorString(err), __FILE__,            \
        __LINE__));                                                            \
    }                                                                          \
  } while (0)

// RAII wrapper for device memory. Use in .hip files only.
// Provides typed hipMalloc/hipFree with copy helpers.
template<typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t count) : size_(count)
  {
    if (count > 0) {
      HIP_CHECK(hipMalloc(&ptr_, count * sizeof(T)));
    }
  }

  ~DeviceBuffer()
  {
    if (ptr_) {
      hipFree(ptr_);
    }
  }

  // Non-copyable
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Movable
  DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_)
  {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
  {
    if (this != &other) {
      if (ptr_) hipFree(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void copy_from_host(const T* host_data, size_t count)
  {
    HIP_CHECK(hipMemcpy(ptr_, host_data, count * sizeof(T),
      hipMemcpyHostToDevice));
  }

  void copy_to_host(T* host_data, size_t count) const
  {
    HIP_CHECK(hipMemcpy(host_data, ptr_, count * sizeof(T),
      hipMemcpyDeviceToHost));
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t size() const { return size_; }

private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
};

#else // OPENMC_USE_HIP but compiled by GCC (not hipcc)

#define OPENMC_HOST_DEVICE
#define OPENMC_DEVICE
#define OPENMC_GLOBAL

#endif // __HIPCC__ || __HIP__

#else // !OPENMC_USE_HIP

#define OPENMC_HOST_DEVICE
#define OPENMC_DEVICE
#define OPENMC_GLOBAL

#endif // OPENMC_USE_HIP

#endif // OPENMC_HIP_UTILS_H
