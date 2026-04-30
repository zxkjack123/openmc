#include "openmc/event.h"

#include "openmc/material.h"
#include "openmc/simulation.h"
#include "openmc/timer.h"

#ifdef OPENMC_USE_HIP
#include "openmc/hip/calculate_xs_kernel.h"
#include "openmc/mgxs_interface.h"
#include "openmc/nuclide.h"
#include "openmc/particle_data.h"
#include "openmc/settings.h"
#include <cmath>
#include <vector>
#include <fmt/format.h>
#endif

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

SharedArray<EventQueueItem> calculate_fuel_xs_queue;
SharedArray<EventQueueItem> calculate_nonfuel_xs_queue;
SharedArray<EventQueueItem> advance_particle_queue;
SharedArray<EventQueueItem> surface_crossing_queue;
SharedArray<EventQueueItem> collision_queue;

vector<Particle> particles;

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

void init_event_queues(int64_t n_particles)
{
  simulation::calculate_fuel_xs_queue.reserve(n_particles);
  simulation::calculate_nonfuel_xs_queue.reserve(n_particles);
  simulation::advance_particle_queue.reserve(n_particles);
  simulation::surface_crossing_queue.reserve(n_particles);
  simulation::collision_queue.reserve(n_particles);

  simulation::particles.resize(n_particles);
}

void free_event_queues(void)
{
  simulation::calculate_fuel_xs_queue.clear();
  simulation::calculate_nonfuel_xs_queue.clear();
  simulation::advance_particle_queue.clear();
  simulation::surface_crossing_queue.clear();
  simulation::collision_queue.clear();

  simulation::particles.clear();
}

void dispatch_xs_event(int64_t buffer_idx)
{
  Particle& p = simulation::particles[buffer_idx];
  if (p.material() == MATERIAL_VOID ||
      !model::materials[p.material()]->fissionable()) {
    simulation::calculate_nonfuel_xs_queue.thread_safe_append({p, buffer_idx});
  } else {
    simulation::calculate_fuel_xs_queue.thread_safe_append({p, buffer_idx});
  }
}

void process_init_events(int64_t n_particles, int64_t source_offset)
{
  simulation::time_event_init.start();
#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < n_particles; i++) {
    initialize_history(simulation::particles[i], source_offset + i + 1);
    dispatch_xs_event(i);
  }
  simulation::time_event_init.stop();
}

void process_calculate_xs_events(SharedArray<EventQueueItem>& queue)
{
  simulation::time_event_calculate_xs.start();

  int64_t offset = simulation::advance_particle_queue.size();

#ifdef OPENMC_USE_HIP
  // =========================================================================
  // HIP path: CPU preamble + GPU XS calculation + CPU fallback for S(a,b)/URR
  // =========================================================================

  // Phase 1: CPU preamble — cell search, state storage, track writing.
  // Void materials get zeroed macro_xs. XS calculation deferred to GPU.
#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < queue.size(); i++) {
    Particle* p = &simulation::particles[queue[i].idx];
    p->event_xs_preamble();
    simulation::advance_particle_queue[offset + i] = queue[i];
  }

  // Phase 2: Classify particles and batch GPU XS calculation
  if (settings::run_CE && queue.size() > 0) {
    int n = static_cast<int>(queue.size());
    int neutron = ParticleType::neutron().transport_index();
    int max_nucs = hip::get_max_nuclides_per_material();

    // Separate GPU-eligible vs CPU-fallback particles
    std::vector<int> gpu_batch_idx;  // queue indices for GPU
    std::vector<int> cpu_fallback_idx; // queue indices for CPU

    std::vector<double> h_energy, h_sqrtkT, h_density_mult;
    std::vector<int> h_material, h_i_log_union;

    for (int i = 0; i < n; i++) {
      const Particle& p = simulation::particles[queue[i].idx];
      int mat = p.material();
      if (mat < 0)
        continue; // void — already handled by preamble

      // Check if particle needs CPU fallback (S(a,b) or URR)
      bool needs_cpu = false;
      if (!model::materials[mat]->thermal_tables_.empty()) {
        needs_cpu = true;
      }
      if (!needs_cpu) {
        double E = p.E();
        for (int nuc_idx : model::materials[mat]->nuclide_) {
          const auto& nuc = *data::nuclides[nuc_idx];
          if (nuc.urr_present_ && !nuc.urr_data_.empty()) {
            if (E >= nuc.urr_data_[0].energy_.front() &&
                E <= nuc.urr_data_[0].energy_.back()) {
              needs_cpu = true;
              break;
            }
          }
        }
      }

      if (needs_cpu) {
        cpu_fallback_idx.push_back(i);
      } else {
        gpu_batch_idx.push_back(i);
        h_energy.push_back(p.E());
        h_sqrtkT.push_back(p.sqrtkT());
        h_material.push_back(mat);
        h_density_mult.push_back(p.density_mult());
        if (p.E() > 0.0) {
          h_i_log_union.push_back(static_cast<int>(
            std::log(p.E() / data::energy_min[neutron]) /
            simulation::log_spacing));
        } else {
          h_i_log_union.push_back(0);
        }
      }
    }

    int n_gpu = static_cast<int>(gpu_batch_idx.size());

    // Phase 2a: GPU XS for eligible particles
    if (n_gpu > 0) {
      int micro_size = n_gpu * max_nucs;

      // Output arrays
      std::vector<double> h_macro_total(n_gpu), h_macro_abs(n_gpu),
        h_macro_fis(n_gpu), h_macro_nufis(n_gpu);
      std::vector<double> h_micro_total(micro_size), h_micro_abs(micro_size),
        h_micro_fis(micro_size), h_micro_nufis(micro_size),
        h_micro_pprod(micro_size), h_micro_interp(micro_size);
      std::vector<int> h_micro_igrid(micro_size), h_micro_itemp(micro_size);

      hip::calculate_xs_full_on_device(n_gpu, max_nucs, h_energy.data(),
        h_sqrtkT.data(), h_material.data(), h_i_log_union.data(),
        h_density_mult.data(), h_macro_total.data(), h_macro_abs.data(),
        h_macro_fis.data(), h_macro_nufis.data(), h_micro_total.data(),
        h_micro_abs.data(), h_micro_fis.data(), h_micro_nufis.data(),
        h_micro_pprod.data(), h_micro_interp.data(), h_micro_igrid.data(),
        h_micro_itemp.data());

      // Write GPU results back to particles
      for (int b = 0; b < n_gpu; b++) {
        int qi = gpu_batch_idx[b];
        Particle& p = simulation::particles[queue[qi].idx];

        // Set macroscopic XS
        p.macro_xs().total = h_macro_total[b];
        p.macro_xs().absorption = h_macro_abs[b];
        p.macro_xs().fission = h_macro_fis[b];
        p.macro_xs().nu_fission = h_macro_nufis[b];

        // Set per-nuclide microscopic XS
        int mat = p.material();
        const auto& mat_nuclides = model::materials[mat]->nuclide_;
        int n_nuc = static_cast<int>(mat_nuclides.size());
        for (int j = 0; j < n_nuc; j++) {
          int i_nuclide = mat_nuclides[j];
          auto& micro = p.neutron_xs(i_nuclide);
          int mi = b * max_nucs + j;

          micro.total = h_micro_total[mi];
          micro.absorption = h_micro_abs[mi];
          micro.fission = h_micro_fis[mi];
          micro.nu_fission = h_micro_nufis[mi];
          micro.photon_prod = h_micro_pprod[mi];
          micro.elastic = CACHE_INVALID;
          micro.thermal = 0.0;
          micro.thermal_elastic = 0.0;
          micro.index_grid = h_micro_igrid[mi];
          micro.index_temp = h_micro_itemp[mi];
          micro.interp_factor = h_micro_interp[mi];
          micro.index_sab = C_NONE;
          micro.sab_frac = 0.0;
          micro.use_ptable = false;
          micro.last_E = p.E();
          micro.last_sqrtkT = p.sqrtkT();
          micro.ncrystal_xs = -1.0;
          for (auto& rx : micro.reaction)
            rx = 0.0;
        }
      }
    }

    // Phase 2b: CPU fallback for S(a,b)/URR particles
    for (int qi : cpu_fallback_idx) {
      Particle& p = simulation::particles[queue[qi].idx];
      model::materials[p.material()]->calculate_xs(p);
    }

  } else if (!settings::run_CE && queue.size() > 0) {
    // Multi-group mode: CPU fallback for all particles
    for (int64_t i = 0; i < queue.size(); i++) {
      Particle& p = simulation::particles[queue[i].idx];
      if (p.material() != MATERIAL_VOID) {
        data::mg.macro_xs_[p.material()].calculate_xs(p);
        p.g_last() = p.g();
      }
    }
  }

#else
  // =========================================================================
  // CPU-only path (original OpenMP implementation)
  // =========================================================================

#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < queue.size(); i++) {
    Particle* p = &simulation::particles[queue[i].idx];
    p->event_calculate_xs();

    // After executing a calculate_xs event, particles will
    // always require an advance event. Therefore, we don't need to use
    // the protected enqueuing function.
    simulation::advance_particle_queue[offset + i] = queue[i];
  }
#endif

  simulation::advance_particle_queue.resize(offset + queue.size());

  queue.resize(0);

  simulation::time_event_calculate_xs.stop();
}

void process_advance_particle_events()
{
  simulation::time_event_advance_particle.start();

#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < simulation::advance_particle_queue.size(); i++) {
    int64_t buffer_idx = simulation::advance_particle_queue[i].idx;
    Particle& p = simulation::particles[buffer_idx];
    p.event_advance();
    if (!p.alive())
      continue;
    if (p.collision_distance() > p.boundary().distance()) {
      simulation::surface_crossing_queue.thread_safe_append({p, buffer_idx});
    } else {
      simulation::collision_queue.thread_safe_append({p, buffer_idx});
    }
  }

  simulation::advance_particle_queue.resize(0);

  simulation::time_event_advance_particle.stop();
}

void process_surface_crossing_events()
{
  simulation::time_event_surface_crossing.start();

#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < simulation::surface_crossing_queue.size(); i++) {
    int64_t buffer_idx = simulation::surface_crossing_queue[i].idx;
    Particle& p = simulation::particles[buffer_idx];
    p.event_cross_surface();
    p.event_revive_from_secondary();
    if (p.alive())
      dispatch_xs_event(buffer_idx);
  }

  simulation::surface_crossing_queue.resize(0);

  simulation::time_event_surface_crossing.stop();
}

void process_collision_events()
{
  simulation::time_event_collision.start();

#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < simulation::collision_queue.size(); i++) {
    int64_t buffer_idx = simulation::collision_queue[i].idx;
    Particle& p = simulation::particles[buffer_idx];
    p.event_collide();
    p.event_revive_from_secondary();
    if (p.alive())
      dispatch_xs_event(buffer_idx);
  }

  simulation::collision_queue.resize(0);

  simulation::time_event_collision.stop();
}

void process_death_events(int64_t n_particles)
{
  simulation::time_event_death.start();
#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < n_particles; i++) {
    Particle& p = simulation::particles[i];
    p.event_death();
  }
  simulation::time_event_death.stop();
}

} // namespace openmc
