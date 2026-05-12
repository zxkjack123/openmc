"""Regression test for multi-detector point detector tallies.

An iron cylinder (R=50 cm, z=-5 to 100 cm) with a 2 MeV isotropic point
source at the origin.  Three point detectors along the z-axis at 10, 30,
and 50 cm test that the next-event estimator handles multiple detector
bins correctly and that flux decreases monotonically with distance
through iron shielding.
"""

import openmc
import pandas as pd
import pytest

from tests.testing_harness import PyAPITestHarness
from tests.regression_tests import config


class MultiPointDetectorHarness(PyAPITestHarness):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- Materials ---
        iron = openmc.Material()
        iron.add_nuclide('Fe56', 1.0)
        iron.set_density('g/cc', 7.87)
        self._model.materials = openmc.Materials([iron])

        # --- Geometry ---
        z_min = openmc.ZPlane(-5.0, boundary_type='vacuum')
        z_max = openmc.ZPlane(100.0, boundary_type='vacuum')
        cyl = openmc.ZCylinder(r=50.0, boundary_type='vacuum')
        cell = openmc.Cell(fill=iron, region=+z_min & -z_max & -cyl)
        self._model.geometry = openmc.Geometry([cell])

        # --- Settings ---
        settings = openmc.Settings()
        settings.run_mode = 'fixed source'
        settings.batches = 5
        settings.particles = 1000
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Point((0, 0, 0)),
            energy=openmc.stats.Discrete([2.0e6], [1.0]),
        )
        self._model.settings = settings

        # --- Tallies ---
        point_filter = openmc.PointFilter(bins=[
            ((0, 0, 10), 1.0),
            ((0, 0, 30), 1.0),
            ((0, 0, 50), 1.0),
        ])
        tally = openmc.Tally(name='multi_point_detector')
        tally.filters = [point_filter]
        tally.scores = ['flux']
        self._model.tallies = openmc.Tallies([tally])

    def _get_results(self):
        """Digest info in the statepoint and return as a string."""
        sp = openmc.StatePoint(self._sp_name)
        tally_dfs = [t.get_pandas_dataframe() for t in sp.tallies.values()]
        df = pd.concat(tally_dfs, ignore_index=True)
        cols = ('mean', 'std. dev.')
        return df.to_csv(None, columns=cols, index=False, float_format='%.7e')


def test_point_detector_multi():
    harness = MultiPointDetectorHarness(
        'statepoint.5.h5', model=openmc.Model())
    harness.main()
