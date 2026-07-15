"""Profile the visibility part of projection preflight on bounded scenes.

The benchmark deliberately stops before projection construction.  It records
the cold visibility build, the cache-hit preflight, and the expensive helpers
inside ``stl_utils.check_visible``.  Run each scene in a fresh process when
comparing peak memory because ``ru_maxrss`` is process-wide.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time
import tracemalloc

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "multi_pinhole_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "multi_pinhole_cache"))

import numpy as np

from multi_pinhole.utils import stl_utils
from benchmarks.benchmark_projection import build_mst_world, build_world


def _rss_peak_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


@contextmanager
def _instrument_visibility_helpers():
    """Collect timings and candidate sizes without changing production code."""
    original_delta = stl_utils.delta_cone_apply
    original_intersection = stl_utils.check_intersection
    calls = {"delta_cone_apply": [], "check_intersection": []}

    def measured_delta(*args, **kwargs):
        started = time.perf_counter()
        result = original_delta(*args, **kwargs)
        matrix = result[0] if isinstance(result, tuple) else result
        calls["delta_cone_apply"].append({
            "seconds": time.perf_counter() - started,
            "shape": list(matrix.shape),
            "candidate_pairs": int(matrix.nnz),
            "csr_bytes": int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes),
        })
        return result

    def measured_intersection(*args, **kwargs):
        started = time.perf_counter()
        result = original_intersection(*args, **kwargs)
        points = args[2] if len(args) > 2 else kwargs["end_points"]
        calls["check_intersection"].append({
            "seconds": time.perf_counter() - started,
            "candidate_points": int(len(points)),
            "intersections": int(np.count_nonzero(result)),
        })
        return result

    stl_utils.delta_cone_apply = measured_delta
    stl_utils.check_intersection = measured_intersection
    try:
        yield calls
    finally:
        stl_utils.delta_cone_apply = original_delta
        stl_utils.check_intersection = original_intersection


def _measure(function):
    tracemalloc.start()
    rss_before = _rss_peak_bytes()
    started = time.perf_counter()
    value = function()
    seconds = time.perf_counter() - started
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, {
        "seconds": seconds,
        "tracemalloc_peak_bytes": int(traced_peak),
        "process_peak_rss_before_bytes": rss_before,
        "process_peak_rss_after_bytes": _rss_peak_bytes(),
    }


def _add_partial_plane_wall(world):
    vertices = np.array([
        [-10.0, -10.0, -50.0], [0.0, -10.0, -50.0],
        [0.0, 10.0, -50.0], [-10.0, 10.0, -50.0],
    ])
    world.walls = [stl_utils.make_stl(vertices, np.array([[0, 1, 2], [0, 2, 3]]))]
    return world


def _build_scene(scene: str, voxel_shape: tuple[int, int, int], mst_spacing: float | None):
    if scene == "toy":
        return build_world(voxel_shape, (8, 8), detector_res=1)
    if scene == "plane":
        return _add_partial_plane_wall(build_world(voxel_shape, (8, 8), detector_res=1))
    return build_mst_world(voxel_shape, voxel_spacing=mst_spacing, detector_res=1)


def _fingerprint(world) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(world.voxel.ranges, dtype=np.float64).tobytes())
    digest.update(np.asarray(world.voxel.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(world.inside_vertices, dtype=bool).tobytes())
    for key, camera in world.cameras.items():
        digest.update(repr(key).encode())
        digest.update(np.asarray(camera.camera_position, dtype=np.float64).tobytes())
        digest.update(np.asarray(camera.rotation_matrix, dtype=np.float64).tobytes())
        for eye in camera.eyes:
            digest.update(np.asarray(eye.position, dtype=np.float64).tobytes())
    for wall in world.walls:
        digest.update(np.asarray(wall.vectors, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True,
    ).stdout.strip()


def run(scene="toy", voxel_shape=(24, 16, 12), mst_spacing=None, batch_points=65536):
    world = _build_scene(scene, tuple(voxel_shape), mst_spacing)
    inside_function = world._inside_function
    inside_kwargs = dict(world._inside_kwargs)
    if inside_function is None:
        inside_function = lambda x, y, z: np.ones_like(x, dtype=bool)

    _, inside_metrics = _measure(
        lambda: world.set_inside_vertices(inside_function, **inside_kwargs),
    )

    original_check_visible = stl_utils.check_visible
    check_visible_calls = []

    def configured_check_visible(*args, **kwargs):
        kwargs["batch_points"] = batch_points
        mesh_obj = kwargs.get("mesh_obj", args[0] if args else None)
        points = kwargs.get("grid_points", args[2] if len(args) > 2 else None)
        started = time.perf_counter()
        result = original_check_visible(*args, **kwargs)
        check_visible_calls.append({
            "seconds": time.perf_counter() - started,
            "triangles": int(len(mesh_obj.vectors)),
            "points": int(len(points)),
            "visible_points": int(np.count_nonzero(result)),
        })
        return result

    stl_utils.check_visible = configured_check_visible
    try:
        with _instrument_visibility_helpers() as helper_calls:
            report, cold = _measure(lambda: world.preflight_projection(res=1, force_visibility=True))
        _, cache_hit = _measure(lambda: world.preflight_projection(res=1, force_visibility=False))
    finally:
        stl_utils.check_visible = original_check_visible

    visible_vertices = {
        repr(key): list(value.shape) for key, value in world._visible_vertices.items()
    }
    visible_voxels = {
        repr(key): list(value.shape) for key, value in world._visible_voxels.items()
    }
    intersection_calls = helper_calls["check_intersection"]
    intersection_summary = {
        "calls": len(intersection_calls),
        "seconds": sum(call["seconds"] for call in intersection_calls),
        "candidate_points": sum(call["candidate_points"] for call in intersection_calls),
        "intersections": sum(call["intersections"] for call in intersection_calls),
        "max_candidate_points_per_call": max(
            (call["candidate_points"] for call in intersection_calls), default=0,
        ),
    }
    return {
        "scene": scene,
        "git_commit": _git_head(),
        "geometry_fingerprint_sha256": _fingerprint(world),
        "voxel_shape": list(world.voxel.shape),
        "grid_shape": list(world.voxel.grid_shape),
        "voxel_count": int(world.voxel.N_voxel),
        "grid_count": int(world.voxel.N_grid),
        "inside_vertex_count": int(np.count_nonzero(world.inside_vertices)),
        "camera_eyes": {repr(key): len(camera.eyes) for key, camera in world.cameras.items()},
        "wall_triangles": [int(len(wall.vectors)) for wall in world.walls],
        "batch_points": int(batch_points),
        "visible_vertices_shapes": visible_vertices,
        "visible_voxels_shapes": visible_voxels,
        "inside": inside_metrics,
        "cold_preflight": cold,
        "cache_hit_preflight": cache_hit,
        "check_visible_calls": check_visible_calls,
        "delta_cone_calls": helper_calls["delta_cone_apply"],
        "check_intersection_summary": intersection_summary,
        "preflight_rows": len(report.eyes),
    }


def _triplet(value: str):
    result = tuple(int(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("toy", "plane", "mst"), default="toy")
    parser.add_argument("--voxel-shape", type=_triplet, default=(24, 16, 12))
    parser.add_argument("--mst-spacing", type=float)
    parser.add_argument("--batch-points", type=int, default=65536)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.scene, args.voxel_shape, args.mst_spacing, args.batch_points)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
