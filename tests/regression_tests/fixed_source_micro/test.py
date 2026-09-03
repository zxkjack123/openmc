import openmc
import openmc.stats

from tests.testing_harness import PyAPITestHarness


def test_fixed_source_micro():
    """Minimal fixed-source sphere for smoke testing and regression drift."""
    mat = openmc.Material()
    mat.add_nuclide("O16", 1.0)
    mat.add_nuclide("U238", 0.0001)
    mat.set_density("g/cc", 7.5)

    surf = openmc.Sphere(r=5.0, boundary_type="vacuum")
    cell = openmc.Cell(fill=mat, region=-surf)

    model = openmc.model.Model()
    model.geometry.root_universe = openmc.Universe(cells=[cell])
    model.materials.append(mat)

    model.settings.run_mode = "fixed source"
    model.settings.batches = 5
    model.settings.particles = 50
    model.settings.temperature = {"default": 300}
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(), strength=10.0
    )

    tally = openmc.Tally()
    tally.scores = ["flux"]
    model.tallies.append(tally)

    harness = PyAPITestHarness("statepoint.5.h5", model)
    harness.main()
