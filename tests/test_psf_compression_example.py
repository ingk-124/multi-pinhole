"""Regression tests for the subvoxel-PSF compression toy model."""

from importlib import util
from pathlib import Path
import sys

import numpy as np

from multi_pinhole.projection import build_psf_group_matrix, factorize_psf_columns


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "evaluate_subvoxel_psf_compression.py"
SPEC = util.spec_from_file_location("evaluate_subvoxel_psf_compression", EXAMPLE)
MODULE = util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _small_problem(resolution=2):
    world = MODULE.build_toy_world(
        axial_distance=300.0, voxel_shape=(3, 3, 2),
        pixel_shape=(10, 10), with_aperture=False,
    )
    return MODULE.build_problem(world, resolution, max_chunk_size=64)


def test_explicit_i_s_matches_existing_projection_matrix():
    problem = _small_problem(resolution=2)

    difference = problem.P_manual - problem.P_existing

    np.testing.assert_allclose(difference.data, 0.0, rtol=0.0, atol=1e-14)
    assert problem.P_manual.shape == problem.P_existing.shape


def test_optical_work_chunks_rebuild_the_uncompressed_projection():
    problem = _small_problem(resolution=2)

    rebuilt = MODULE.project_by_optical_work_chunks(problem)
    difference = rebuilt - problem.P_existing

    np.testing.assert_allclose(difference.data, 0.0, rtol=0.0, atol=1e-14)
    sample_indices = np.concatenate(problem.work_chunks)
    np.testing.assert_array_equal(np.sort(sample_indices),
                                  np.arange(problem.active_points.shape[0]))


def test_work_chunk_packing_does_not_change_projection_or_cross_bin_grouping():
    problem = _small_problem(resolution=2)
    for max_size in (7, 19, 64):
        work_chunks = MODULE.pack_optical_chunks(problem.optical_chunks, max_size)
        rebuilt = MODULE.project_by_optical_work_chunks(problem, work_chunks)
        difference = rebuilt - problem.P_existing
        np.testing.assert_allclose(difference.data, 0.0, rtol=0.0, atol=1e-14)

    compression = MODULE.compress_problem(
        problem, tolerance=0.1, metric="relative_l2", algorithm="recursive",
    )
    for members in compression.group_members:
        assert np.unique(problem.optical_bin_id[members]).size == 1


def test_recursive_and_leader_psf_groups_keep_bounds_and_signed_sum():
    problem = _small_problem(resolution=2)
    signed = MODULE.emission_profiles(problem.world.voxel.gravity_center)["signed"]
    exact = np.asarray(problem.P_existing @ signed).ravel()

    recursive = MODULE.compress_problem(
        problem, tolerance=0.1, metric="relative_l2", algorithm="recursive",
    )
    leader = MODULE.compress_problem(
        problem, tolerance=0.1, metric="l1", algorithm="leader",
    )

    assert recursive.group_max_relative_l2.max(initial=0.0) <= 0.1 + 1e-12
    assert leader.group_max_l1.max(initial=0.0) <= 0.1 + 1e-12
    for compression in (recursive, leader):
        approximate = np.asarray(compression.Q @ (compression.A @ signed)).ravel()
        np.testing.assert_allclose(approximate.sum(), exact.sum(), rtol=2e-6, atol=1e-12)
        reference_sum = np.asarray(problem.P_existing.sum(axis=0)).ravel()
        approximate_sum = np.asarray(compression.P_approx.sum(axis=0)).ravel()
        np.testing.assert_allclose(approximate_sum, reference_sum, rtol=2e-6, atol=1e-12)


def test_library_psf_factorization_builds_q_a_on_toy_problem():
    problem = _small_problem(resolution=2)
    order = np.concatenate(problem.optical_chunks)
    scope_offsets = np.concatenate((
        [0], np.cumsum([chunk.size for chunk in problem.optical_chunks]),
    ))
    factorization = factorize_psf_columns(
        problem.I[:, order], scope_offsets, tolerance=0.1,
    )
    assert factorization is not None

    A = build_psf_group_matrix(factorization, problem.S[order])
    approximation = factorization.Q @ A
    reference_sum = np.asarray(problem.P_existing.sum(axis=0)).ravel()
    approximate_sum = np.asarray(approximation.sum(axis=0)).ravel()

    assert factorization.group_max_relative_l2.max(initial=0.0) <= 0.1 + 1e-12
    np.testing.assert_allclose(approximate_sum, reference_sum, rtol=2e-6, atol=1e-12)


def test_production_hybrid_sweep_records_bytes_timings_and_errors(tmp_path):
    result = MODULE.run_production_sweep(
        tmp_path,
        axial_distances=(100.0,), resolutions=(1,), bin_widths=(1.0,),
        tolerances=(0.1,), metrics=("relative_l2",),
        algorithms=("recursive",), max_group_fractions=(0.8,),
        voxel_shape=(4, 4, 3), pixel_shape=(8, 8), timing_repeats=2,
    )

    assert result["csv_path"].is_file()
    assert result["figure_path"].is_file()
    assert result["rows"]
    row = next(item for item in result["rows"] if item["profile"] == "gaussian")
    assert row["scope_count"] > 0
    assert row["active_sample_count"] > 0
    assert row["direct_bytes"] > 0
    assert row["hybrid_bytes"] > 0
    assert row["storage_compression"] > 0.0
    assert row["direct_build_seconds"] > 0.0
    assert row["hybrid_build_seconds"] > 0.0
    assert row["max_column_sum_relative"] < 2e-6
    assert np.isfinite(row["relative_l2"])
