#ifndef OPENMC_HIP_CALCULATE_XS_KERNEL_H
#define OPENMC_HIP_CALCULATE_XS_KERNEL_H

#ifdef OPENMC_USE_HIP

namespace openmc {
namespace hip {

//! \brief Batch cross-section lookup on device for particles in the XS queue.
//!
//! Each thread handles one particle: determines temperature index, performs
//! binary search on the energy grid, interpolates microscopic XS for each
//! nuclide in the particle's material, and accumulates macroscopic XS.
//!
//! Limitations (Phase 3):
//!   - Nearest temperature method only (no stochastic interpolation)
//!   - No S(a,b) thermal scattering correction
//!   - No URR probability table treatment
//!   - No multipole evaluation
//!   - No depletion reaction cross sections
//!   - No NCrystal cross sections
//!
//! \param n_particles      Number of particles to process
//! \param d_energy         Particle energies [eV]
//! \param d_sqrtkT         Particle sqrt(k*T) values
//! \param d_material       Material index for each particle (-1 = void)
//! \param d_i_log_union    Log-union grid index for each particle
//! \param d_macro_total    [out] Macroscopic total XS per particle
//! \param d_macro_absorption [out] Macroscopic absorption XS per particle
//! \param d_macro_fission  [out] Macroscopic fission XS per particle
//! \param d_macro_nu_fission [out] Macroscopic nu-fission XS per particle
void launch_calculate_xs_kernel(int n_particles, const double* d_energy,
  const double* d_sqrtkT, const int* d_material, const int* d_i_log_union,
  double* d_macro_total, double* d_macro_absorption, double* d_macro_fission,
  double* d_macro_nu_fission);

//! \brief Host-callable wrapper: allocate device memory, launch kernel, copy
//! results back. Takes host arrays in, returns host arrays out.
//!
//! \param n_particles      Number of particles
//! \param h_energy         [in]  Host array of particle energies
//! \param h_sqrtkT         [in]  Host array of sqrt(kT) values
//! \param h_material       [in]  Host array of material indices (-1 = void)
//! \param h_i_log_union    [in]  Host array of log-union grid indices
//! \param h_macro_total    [out] Host array for macroscopic total XS
//! \param h_macro_absorption [out] Host array for macroscopic absorption XS
//! \param h_macro_fission  [out] Host array for macroscopic fission XS
//! \param h_macro_nu_fission [out] Host array for macroscopic nu-fission XS
void calculate_xs_on_device(int n_particles, const double* h_energy,
  const double* h_sqrtkT, const int* h_material, const int* h_i_log_union,
  double* h_macro_total, double* h_macro_absorption, double* h_macro_fission,
  double* h_macro_nu_fission);

//! \brief Extended host-callable wrapper: computes both macroscopic AND
//! per-nuclide microscopic cross sections on device.
//!
//! Output micro XS arrays are indexed as [particle * max_nucs + nuclide_j].
//! For each particle, nuclide_j corresponds to the j-th nuclide in its
//! material (same order as Material::nuclide_).
//!
//! \param n_particles        Number of particles
//! \param max_nucs           Max nuclides per material (stride for micro arrays)
//! \param h_energy           [in]  Particle energies
//! \param h_sqrtkT           [in]  sqrt(kT) values
//! \param h_material         [in]  Material indices (-1 = void)
//! \param h_i_log_union      [in]  Log-union grid indices
//! \param h_density_mult     [in]  Per-particle density multiplier
//! \param h_macro_total      [out] Macroscopic total XS
//! \param h_macro_absorption [out] Macroscopic absorption XS
//! \param h_macro_fission    [out] Macroscopic fission XS
//! \param h_macro_nu_fission [out] Macroscopic nu-fission XS
//! \param h_micro_total      [out] Per-nuclide total [n * max_nucs]
//! \param h_micro_abs        [out] Per-nuclide absorption [n * max_nucs]
//! \param h_micro_fis        [out] Per-nuclide fission [n * max_nucs]
//! \param h_micro_nufis      [out] Per-nuclide nu-fission [n * max_nucs]
//! \param h_micro_pprod      [out] Per-nuclide photon production [n * max_nucs]
//! \param h_micro_interp     [out] Per-nuclide interpolation factor [n * max_nucs]
//! \param h_micro_igrid      [out] Per-nuclide grid index [n * max_nucs]
//! \param h_micro_itemp      [out] Per-nuclide temperature index [n * max_nucs]
void calculate_xs_full_on_device(int n_particles, int max_nucs,
  const double* h_energy, const double* h_sqrtkT, const int* h_material,
  const int* h_i_log_union, const double* h_density_mult,
  double* h_macro_total, double* h_macro_absorption, double* h_macro_fission,
  double* h_macro_nu_fission, double* h_micro_total, double* h_micro_abs,
  double* h_micro_fis, double* h_micro_nufis, double* h_micro_pprod,
  double* h_micro_interp, int* h_micro_igrid, int* h_micro_itemp);

//! \brief Return the maximum number of nuclides in any material.
//! Computed during copy_xs_data_to_device().
int get_max_nuclides_per_material();

} // namespace hip
} // namespace openmc

#endif // OPENMC_USE_HIP
#endif // OPENMC_HIP_CALCULATE_XS_KERNEL_H
