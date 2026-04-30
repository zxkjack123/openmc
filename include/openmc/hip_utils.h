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
      fatal_error(fmt::format(                                                 \
        "HIP error: {} at {}:{}", hipGetErrorString(err), __FILE__,            \
        __LINE__));                                                            \
    }                                                                          \
  } while (0)

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
