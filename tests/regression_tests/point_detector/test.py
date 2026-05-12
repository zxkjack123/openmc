"""Regression test for point detector (next-event estimator) tallies.

A fixed isotropic source at the center of a 5 cm water sphere surrounded
by a 30 cm void region (vacuum outer boundary).  Two point detectors sit
in the void at (0, 0, 20) and (20, 0, 0), each with exclusion radius
r0 = 1 cm.  A flux tally with PointFilter + EnergyFilter verifies that
the next-event estimator produces consistent, non-zero results.

Additional tests verify that point detectors correctly reject
multi-group mode and non-vacuum boundary conditions.
"""

import pytest
import openmc
import pandas as pd

from tests.testing_harness import PyAPITestHarness
from tests.regression_tests import config


class PointDetectorTestHarness(PyAPITestHarness):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- Materials ---
        water = openmc.Material(name='water')
        water.add_nuclide('H1', 2.0)
        water.add_nuclide('O16', 1.0)
        water.set_density('g/cm3', 1.0)

        self._model.materials = openmc.Materials([water])

        # --- Geometry ---
        # Inner water sphere + outer void shell so that detectors are
        # inside the model domain (required for ray-tracing to reach them).
        inner = openmc.Sphere(r=5.0)
        outer = openmc.Sphere(r=30.0, boundary_type='vacuum')
        water_cell = openmc.Cell(fill=water, region=-inner)
        void_cell = openmc.Cell(region=+inner & -outer)
        root = openmc.Universe(cells=[water_cell, void_cell])
        self._model.geometry = openmc.Geometry(root)

        # --- Settings ---
        settings = openmc.Settings()
        settings.run_mode = 'fixed source'
        settings.batches = 10
        settings.particles = 1000
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Point((0.0, 0.0, 0.0)),
            energy=openmc.stats.Discrete([1.0e6], [1.0]),
        )
        self._model.settings = settings

        # --- Tallies ---
        point_filter = openmc.PointFilter([
            ((0.0, 0.0, 20.0), 1.0),
            ((20.0, 0.0, 0.0), 1.0),
        ])
        energy_filter = openmc.EnergyFilter([0.0, 1.0e5, 2.0e7])

        tally = openmc.Tally(name='point_det_flux')
        tally.filters = [point_filter, energy_filter]
        tally.scores = ['flux']

        self._model.tallies = openmc.Tallies([tally])

    def _get_results(self):
        """Digest info in the statepoint and return as a string."""
        sp = openmc.StatePoint(self._sp_name)
        tally_dfs = [t.get_pandas_dataframe() for t in sp.tallies.values()]
        df = pd.concat(tally_dfs, ignore_index=True)
        cols = ('mean', 'std. dev.')
        return df.to_csv(None, columns=cols, index=False, float_format='%.7e')


def test_point_detector():
    harness = PointDetectorTestHarness(
        'statepoint.10.h5', model=openmc.Model())
    harness.main()


def test_point_detector_mg_error(run_in_tmpdir):
    """Point detectors must reject multi-group mode at runtime."""
    import numpy as np

    # Create a minimal 1-group MG cross section library
    groups = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    xs = openmc.XSdata('mat_1', groups)
    xs.order = 0
    xs.set_total([1.0])
    xs.set_scatter_matrix(np.array([[[1.0]]]))
    xs.set_absorption([0.0])
    mg_lib = openmc.MGXSLibrary(groups)
    mg_lib.add_xsdata(xs)
    mg_lib.export_to_hdf5('mgxs.h5')

    # Build a simple MG model
    mat = openmc.Material(name='mat_1')
    mat.set_density('macro', 1.0)
    mat.add_macroscopic('mat_1')

    sphere = openmc.Sphere(r=10.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=mat, region=-sphere)

    settings = openmc.Settings()
    settings.energy_mode = 'multi-group'
    settings.run_mode = 'fixed source'
    settings.batches = 2
    settings.particles = 100
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0.0, 0.0, 0.0)),
    )

    pf = openmc.PointFilter([((0.0, 0.0, 5.0), 1.0)])
    t = openmc.Tally()
    t.filters = [pf]
    t.scores = ['flux']

    model = openmc.Model()
    mats = openmc.Materials([mat])
    mats.cross_sections = 'mgxs.h5'
    model.materials = mats
    model.geometry = openmc.Geometry(openmc.Universe(cells=[cell]))
    model.settings = settings
    model.tallies = openmc.Tallies([t])

    with pytest.raises(RuntimeError, match='multi-group'):
        model.run(openmc_exec=config['exe'])


def test_point_detector_nonvacuum_error(run_in_tmpdir):
    """Point detectors must reject non-vacuum boundary conditions."""
    water = openmc.Material()
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)

    # Use reflective boundary (non-vacuum)
    sphere = openmc.Sphere(r=10.0, boundary_type='reflective')
    cell = openmc.Cell(fill=water, region=-sphere)
    root = openmc.Universe(cells=[cell])

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.batches = 2
    settings.particles = 100
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0.0, 0.0, 0.0)),
        energy=openmc.stats.Discrete([1.0e6], [1.0]),
    )

    pf = openmc.PointFilter([((0.0, 0.0, 5.0), 1.0)])
    t = openmc.Tally()
    t.filters = [pf]
    t.scores = ['flux']

    model = openmc.Model()
    model.materials = openmc.Materials([water])
    model.geometry = openmc.Geometry(root)
    model.settings = settings
    model.tallies = openmc.Tallies([t])

    with pytest.raises(RuntimeError, match='non-vacuum'):
        model.run(openmc_exec=config['exe'])
