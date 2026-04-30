#ifndef OPENMC_HIP_XS_DATA_DEVICE_H
#define OPENMC_HIP_XS_DATA_DEVICE_H

//! \file xs_data_device.h
//! Device-side cross section data management for HIP acceleration.
//!
//! This header is included from GCC-compiled .cpp files, so it must NOT
//! include any HIP runtime headers. All HIP API calls are in the .hip file.

#ifdef OPENMC_USE_HIP

namespace openmc {
namespace hip {

//! Copy nuclear cross section data from host to device memory.
//! Called from openmc_simulation_init() after all nuclide data is loaded.
//! Prints summary of copied data (nuclide count, memory usage).
void copy_xs_data_to_device();

//! Free device-side cross section data.
//! Called from openmc_simulation_finalize() or at program exit.
void free_xs_data_on_device();

} // namespace hip
} // namespace openmc

#endif // OPENMC_USE_HIP
#endif // OPENMC_HIP_XS_DATA_DEVICE_H
