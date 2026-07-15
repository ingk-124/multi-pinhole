"""Benchmark partial-voxel visibility during projection construction.

Run the reference and optimized implementations in separate processes, then
compare ``projection_sha256``.  Preflight is always completed first on the same
World instance so ``projection_seconds`` measures the cache-hit projection path
including the fresh visibility tests for partial sub-voxel centers.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from benchmarks.benchmark_projection import build_mst_world
from benchmarks.profile_wall_visibility import _fingerprint, _git_head, _measure
from multi_pinhole.utils import stl_utils


def _projection_digest(matrix) -> str:
    matrix = matrix.tocsr()
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(matrix.indptr.tobytes())
    digest.update(matrix.indices.tobytes())
    digest.update(matrix.data.tobytes())
    return digest.hexdigest()


def run(voxel_shape=(8, 6, 4), partial_res=2, implementation="optimized",
        max_working_memory=200_000_000):
    world = build_mst_world(tuple(voxel_shape), detector_res=1)
    with contextlib.redirect_stdout(io.StringIO()):
        world.remove_camera("right")
    production = stl_utils.check_visible
    selected = production if implementation == "optimized" else stl_utils._check_visible_reference
    stl_utils.check_visible = selected
    try:
        np.random.seed(124)
        report, preflight = _measure(
            lambda: world.preflight_projection(res=1, partial_res=partial_res),
        )
        state = world.visible_voxels["left"][0]
        cached_vertices = world._visible_vertices["left"]
        cached_voxels = world._visible_voxels["left"]

        partial_visibility_calls = []
        original_find_visible_points = world.find_visible_points

        def measured_find_visible_points(points, *args, **kwargs):
            started = time.perf_counter()
            result = original_find_visible_points(points, *args, **kwargs)
            partial_visibility_calls.append({
                "points": int(len(points)),
                "seconds": time.perf_counter() - started,
            })
            return result

        world.find_visible_points = measured_find_visible_points
        np.random.seed(124)
        _, projection_metrics = _measure(
            lambda: world.set_projection_matrix(
                res=1, partial_res=partial_res, parallel=1, verbose=0,
                force=False, max_working_memory=max_working_memory,
            ),
        )
        projection = world.P_matrix["left"].tocsr()
    finally:
        stl_utils.check_visible = production

    return {
        "implementation": implementation,
        "git_commit": _git_head(),
        "geometry_fingerprint_sha256": _fingerprint(world),
        "voxel_shape": list(world.voxel.shape),
        "voxel_count": int(world.voxel.N_voxel),
        "partial_resolution": int(partial_res),
        "partial_voxels": int(np.count_nonzero(state == 1)),
        "full_voxels": int(np.count_nonzero(state == 2)),
        "preflight": preflight,
        "projection": projection_metrics,
        "preflight_rows": len(report.eyes),
        "vertex_cache_reused": world._visible_vertices["left"] is cached_vertices,
        "voxel_cache_reused": world._visible_voxels["left"] is cached_voxels,
        "partial_visibility_calls": len(partial_visibility_calls),
        "partial_visibility_points": sum(call["points"] for call in partial_visibility_calls),
        "partial_visibility_seconds": sum(call["seconds"] for call in partial_visibility_calls),
        "projection_shape": list(projection.shape),
        "projection_nnz": int(projection.nnz),
        "projection_sum": float(projection.sum()),
        "projection_sha256": _projection_digest(projection),
    }


def _triplet(value: str):
    result = tuple(int(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxel-shape", type=_triplet, default=(8, 6, 4))
    parser.add_argument("--partial-res", type=int, default=2)
    parser.add_argument("--implementation", choices=("optimized", "reference"), default="optimized")
    parser.add_argument("--max-working-memory-mb", type=float, default=200.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(
        args.voxel_shape, args.partial_res, args.implementation,
        int(args.max_working_memory_mb * 1_000_000),
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
