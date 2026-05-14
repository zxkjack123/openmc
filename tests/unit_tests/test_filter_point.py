import numpy as np
import openmc
import pytest


def test_point_filter_creation():
    """Test PointFilter can be created with valid bins."""
    pf = openmc.PointFilter(bins=[((1, 2, 3), 0.5)])
    assert pf.num_bins == 1

    pf = openmc.PointFilter(bins=[
        ((0, 0, 0), 1.0),
        ((10, 0, 0), 0.5),
        ((0, 10, 0), 0.01),
    ])
    assert pf.num_bins == 3

    # Make sure __repr__ works
    repr(pf)


def test_point_filter_invalid_bins():
    """Test PointFilter rejects invalid bin specifications."""
    with pytest.raises((TypeError, ValueError)):
        openmc.PointFilter(bins=[((1, 2), 0.5)])  # 2D position

    with pytest.raises((TypeError, ValueError)):
        openmc.PointFilter(bins=[(1, 2, 3, 0.5)])  # flat tuple


def test_point_filter_xml_roundtrip():
    """Test PointFilter serializes and deserializes correctly."""
    pf = openmc.PointFilter(bins=[
        ((1.0, 2.0, 3.0), 0.5),
        ((4.0, 5.0, 6.0), 1.0),
    ])
    elem = pf.to_xml_element()
    # Check XML structure
    assert elem.tag == 'filter'
    assert elem.get('type') == 'point'
    bins_text = elem.find('bins').text
    values = [float(x) for x in bins_text.split()]
    assert len(values) == 8  # 2 detectors × 4 values each
    assert values == [1.0, 2.0, 3.0, 0.5, 4.0, 5.0, 6.0, 1.0]

    # from_xml_element round-trip
    new_pf = openmc.Filter.from_xml_element(elem)
    assert isinstance(new_pf, openmc.PointFilter)
    assert new_pf.id == pf.id
    assert new_pf == pf


def test_point_filter_in_tally():
    """Test PointFilter can be used in a Tally."""
    pf = openmc.PointFilter(bins=[((0, 0, 100), 1.0)])
    tally = openmc.Tally()
    tally.filters = [pf]
    tally.scores = ['flux']
    # Should not raise
    tallies = openmc.Tallies([tally])
    elem = tallies.to_xml_element()
    assert elem is not None


def test_point_filter_hdf5_roundtrip(tmp_path):
    """Test PointFilter HDF5 read via from_hdf5."""
    import h5py

    bins = [((10.0, 20.0, 30.0), 2.0), ((0.0, 0.0, 0.0), 0.1)]
    f = openmc.PointFilter(bins)

    # Simulate statepoint HDF5 structure
    path = tmp_path / "test_point_filter.h5"
    with h5py.File(path, 'w') as h5:
        grp = h5.create_group(f"filter {f.id}")
        flat = []
        for pos, r0 in bins:
            flat.extend([*pos, r0])
        grp.create_dataset('bins', data=flat)
        grp.create_dataset('n_bins', data=len(bins))
        grp.attrs['type'] = np.bytes_('point')

    with h5py.File(path, 'r') as h5:
        grp = h5[f"filter {f.id}"]
        new_f = openmc.PointFilter.from_hdf5(grp)

    assert new_f.id == f.id
    assert new_f.num_bins == 2
    assert new_f.bins[0] == ((10.0, 20.0, 30.0), 2.0)
    assert new_f.bins[1] == ((0.0, 0.0, 0.0), 0.1)


def test_point_filter_equality():
    """Test PointFilter.__eq__ with equal and unequal instances."""
    f1 = openmc.PointFilter([((1.0, 2.0, 3.0), 0.5)])
    f2 = openmc.PointFilter([((1.0, 2.0, 3.0), 0.5)])
    f3 = openmc.PointFilter([((1.0, 2.0, 3.0), 1.0)])

    assert f1 == f2
    assert f1 != f3
    assert f1 != "not a filter"


def test_point_filter_pandas():
    """Test PointFilter.get_pandas_dataframe output."""
    bins = [((1.0, 2.0, 3.0), 0.5), ((4.0, 5.0, 6.0), 1.0)]
    f = openmc.PointFilter(bins)
    df = f.get_pandas_dataframe(2, 1)
    assert len(df) == 2
    assert 'point' in df.columns
    assert 'R0=0.5' in df['point'].iloc[0]
    assert 'R0=1.0' in df['point'].iloc[1]
