"""Tests for voxel-aware plotting helpers."""

import matplotlib
import numpy as np
import pytest

from multi_pinhole import Voxel
from multi_pinhole.utils.plot import plot_voxel_slice, plot_voxel_volume

matplotlib.use("Agg")


@pytest.fixture
def voxel():
    """Return a small non-cubic grid that exposes axis-order mistakes."""
    return Voxel.uniform_voxel(
        ranges=[[0, 2], [10, 16], [-2, 2]],
        shape=[2, 3, 4],
    )


def test_plot_voxel_volume_accepts_voxel_shape_and_omits_nan(voxel):
    emission = np.arange(voxel.N_voxel, dtype=float).reshape(voxel.shape)
    emission[0, 0, 0] = np.nan

    figure = plot_voxel_volume(voxel, emission, length_unit="mm")

    trace = figure.data[0]
    assert len(trace.value) == voxel.N_voxel - 1
    assert np.asarray(trace.value).min() == 1
    assert trace.isomin == 1
    assert trace.isomax == voxel.N_voxel - 1
    assert figure.layout.scene.xaxis.title.text == "x [mm]"
    assert figure.layout.scene.aspectmode == "data"


def test_plot_voxel_volume_validates_shape_and_mask(voxel):
    with pytest.raises(ValueError, match="emission must have shape"):
        plot_voxel_volume(voxel, np.ones(voxel.N_voxel - 1))
    with pytest.raises(ValueError, match="mask must have shape"):
        plot_voxel_volume(voxel, np.ones(voxel.N_voxel), mask=np.ones(3))


def test_plot_voxel_slice_uses_voxel_edges_and_nearest_coordinate(voxel):
    emission = np.arange(voxel.N_voxel).reshape(voxel.shape)

    ax, mesh = plot_voxel_slice(
        voxel,
        emission,
        axis="z",
        coordinate=0.4,
        colorbar=False,
    )

    assert mesh.get_array().size == voxel.shape[0] * voxel.shape[1]
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_title() == "z = 0.5"
