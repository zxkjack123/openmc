#include "openmc/event.h"

#include "openmc/material.h"
#include "openmc/simulation.h"
#include "openmc/timer.h"

#ifdef OPENMC_USE_HIP
#include "openmc/hip/calculate_xs_kernel.h"
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

#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < queue.size(); i++) {
    Particle* p = &simulation::particles[queue[i].idx];
    p->event_calculate_xs();

    // After executing a calculate_xs event, particles will
    // always require an advance event. Therefore, we don't need to use
    // the protected enqueuing function.
    simulation::advance_particle_queue[offset + i] = queue[i];
  }

#ifdef OPENMC_USE_HIP
  // GPU validation pass: run the XS kernel on device with the same inputs
  // and compare results against the CPU-computed values. This runs once
  // per simulation (on the first batch with enough particles) to validate
  // the kernel produces correct results.
  //
  // Note: The GPU kernel does not handle S(a,b) thermal scattering, so
  // particles in materials with thermal scattering tables are excluded
  // from the comparison.
  {
    static bool validated = false;
    if (!validated && queue.size() > 0 && settings::run_CE) {
      validated = true;
      int n = static_cast<int>(queue.size());
      int n_validate = std::min(n, 200);

      // Extract particle data for GPU
      std::vector<double> h_energy(n), h_sqrtkT(n);
      std::vector<int> h_material(n), h_i_log_union(n);
      int neutron = ParticleType::neutron().transport_index();
      for (int i = 0; i < n; i++) {
        const Particle& p = simulation::particles[queue[i].idx];
        h_energy[i] = p.E();
        h_sqrtkT[i] = p.sqrtkT();
        h_material[i] = p.material();
        if (p.E() > 0.0 && p.material() >= 0) {
          h_i_log_union[i] = static_cast<int>(
            std::log(p.E() / data::energy_min[neutron]) /
            simulation::log_spacing);
        } else {
          h_i_log_union[i] = 0;
        }
      }

      // Run XS calculation on device
      std::vector<double> gpu_total(n), gpu_abs(n), gpu_fis(n), gpu_nufis(n);
      hip::calculate_xs_on_device(n, h_energy.data(), h_sqrtkT.data(),
        h_material.data(), h_i_log_union.data(), gpu_total.data(),
        gpu_abs.data(), gpu_fis.data(), gpu_nufis.data());

      // Compare GPU vs CPU results, excluding S(a,b) and URR particles
      int n_compared = 0;
      int n_match = 0;
      int n_sab_skip = 0;
      int n_urr_skip = 0;
      double max_rel_err = 0.0;
      int worst_idx = -1;

      for (int i = 0; i < n_validate; i++) {
        int mat = h_material[i];
        if (mat < 0)
          continue; // skip void

        // Skip materials with S(a,b) thermal tables
        if (!model::materials[mat]->thermal_tables_.empty()) {
          ++n_sab_skip;
          continue;
        }

        // Skip particles in URR range for any nuclide in this material
        bool in_urr = false;
        for (int nuc_idx : model::materials[mat]->nuclide_) {
          const auto& nuc = *data::nuclides[nuc_idx];
          if (nuc.urr_present_ && !nuc.urr_data_.empty()) {
            double E = h_energy[i];
            double urr_emin = nuc.urr_data_[0].energy_.front();
            double urr_emax = nuc.urr_data_[0].energy_.back();
            if (E >= urr_emin && E <= urr_emax) {
              in_urr = true;
              break;
            }
          }
        }
        if (in_urr) {
          ++n_urr_skip;
          continue;
        }

        const Particle& p = simulation::particles[queue[i].idx];
        double cpu_total = p.macro_xs().total;
        double gpu_tot = gpu_total[i];
        ++n_compared;

        if (cpu_total > 0.0) {
          double rel = std::abs(gpu_tot - cpu_total) / cpu_total;
          if (rel > max_rel_err) {
            max_rel_err = rel;
            worst_idx = i;
          }
          if (rel < 1.0e-10)
            ++n_match;
        } else if (gpu_tot == 0.0) {
          ++n_match;
        }
      }

      fmt::print(
        " HIP XS kernel validation: {}/{} particles match (max rel err = "
        "{:.2e}, {} S(a,b) skipped, {} URR skipped)\n",
        n_match, n_compared, max_rel_err, n_sab_skip, n_urr_skip);

      // Print details for up to 5 mismatched non-S(a,b)/non-URR particles
      if (n_match < n_compared) {
        int n_printed = 0;
        for (int i = 0; i < n_validate && n_printed < 5; i++) {
          int mat = h_material[i];
          if (mat < 0 || !model::materials[mat]->thermal_tables_.empty())
            continue;
          // Also skip URR particles for mismatch reporting
          bool in_urr = false;
          for (int nuc_idx : model::materials[mat]->nuclide_) {
            const auto& nuc = *data::nuclides[nuc_idx];
            if (nuc.urr_present_ && !nuc.urr_data_.empty()) {
              double E = h_energy[i];
              if (E >= nuc.urr_data_[0].energy_.front() &&
                  E <= nuc.urr_data_[0].energy_.back()) {
                in_urr = true;
                break;
              }
            }
          }
          if (in_urr)
            continue;
          const Particle& p = simulation::particles[queue[i].idx];
          double cpu_t = p.macro_xs().total;
          double gpu_t = gpu_total[i];
          double rel = (cpu_t > 0) ? std::abs(gpu_t - cpu_t) / cpu_t : 0.0;
          if (rel >= 1.0e-10) {
            fmt::print(
              "   mismatch[{}]: mat={} E={:.6e} cpu_total={:.10e} "
              "gpu_total={:.10e} rel={:.2e}\n",
              i, mat, h_energy[i], cpu_t, gpu_t, rel);
            fmt::print(
              "     cpu: abs={:.10e} fis={:.10e} nufis={:.10e}\n",
              p.macro_xs().absorption, p.macro_xs().fission,
              p.macro_xs().nu_fission);
            fmt::print(
              "     gpu: abs={:.10e} fis={:.10e} nufis={:.10e}\n",
              gpu_abs[i], gpu_fis[i], gpu_nufis[i]);
            ++n_printed;
          }
        }
      }
    }
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
