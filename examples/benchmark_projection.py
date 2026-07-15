"""Benchmark projection-matrix construction on a configurable small scene.

The default case is intentionally much smaller than the MST workflow so it can
be used while iterating on projection performance.  Use the same command line
before and after a change to compare elapsed time and sparse-matrix size.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "multi_pinhole_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "multi_pinhole_cache"))

import numpy as np
from stl import mesh

from multi_pinhole import Aperture, Camera, Eye, Screen, Voxel, World
from multi_pinhole.utils import stl_utils


def build_world(voxel_shape: tuple[int, int, int], pixel_shape: tuple[int, int],
                detector_res: int = 2) -> World:
    """Build a deterministic projection benchmark without wall occlusion."""
    voxel = Voxel.uniform_voxel(
        ranges=((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)),
        shape=voxel_shape,
    )
    camera = Camera(
        eyes=[Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.6,
                  focal_length=25.0, position=(0.0, 0.0))],
        screen=Screen(screen_shape="rectangle", screen_size=(10.0, 10.0),
                      pixel_shape=pixel_shape, subpixel_resolution=detector_res),
        apertures=Aperture(shape="circle", size=12.0, position=(0.0, 0.0, 40.0),
                           resolution=24, max_size=30.0),
        camera_position=(0.0, 0.0, -150.0),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    return world


def _center_ranges_for_spacing(bounds, spacing):
    """Return center-coordinate ranges and shapes covering axis bounds."""
    ranges, shape = [], []
    for lower, upper in bounds:
        count = int(np.ceil((upper - lower) / spacing))
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (count - 1) * spacing
        ranges.append((midpoint - half_width, midpoint + half_width))
        shape.append(count)
    return tuple(ranges), tuple(shape)


def build_mst_world(voxel_shape: tuple[int, int, int], voxel_spacing=None,
                    detector_res: int = 5, voxel_bounds=None) -> World:
    """Build a reduced-grid version of the 2026 MST tangential SXR case."""
    camera_center = np.array([1550.7, -1522.4, 210.8])
    forward_point = np.array([1525.9, -1521.1, 207.3])
    right_point = np.array([1551.9, -1511.2, 206.9])
    aperture_model = stl_utils.generate_aperture_stl(
        shape="circle", size=1.8, resolution=40, max_size=15,
    )

    def make_camera(offset):
        aperture = Aperture(stl_model=aperture_model, position=(0.0, 0.0, 13.0))
        return Camera.single_pinhole(
            focal_length=25.0,
            eye_size=1.0,
            screen_size=7.5,
            pixel_shape=(61, 61),
            subpixel_resolution=detector_res,
            apertures=aperture,
        ).set_camera_position(camera_center).set_orientation_from_points(
            look_point=forward_point,
            right_point=right_point,
        ).translate_camera((offset, 0.0, 0.0))

    if voxel_spacing is None:
        ranges = ((-2012.5, 2012.5), (-2012.5, 2012.5), (-512.5, 512.5))
    else:
        bounds = (((-2020.0, 2020.0), (-2020.0, 2020.0), (-520.0, 520.0))
                  if voxel_bounds is None else voxel_bounds)
        ranges, voxel_shape = _center_ranges_for_spacing(
            bounds, voxel_spacing,
        )
    voxel = Voxel.uniform_voxel_from_centers(
        ranges=ranges,
        shape=voxel_shape,
        coordinate_type="torus_inverse",
        coordinate_parameters={"major_radius": 1500, "minor_radius": 520},
    )
    wall_path = Path(__file__).resolve().parent / "mst" / "MST_wall-mesh.stl"
    world = World(
        voxel=voxel,
        cameras={"left": make_camera(-4.15), "right": make_camera(4.15)},
        walls=mesh.Mesh.from_file(wall_path),
        verbose=0,
    )

    def inside_condition(x, y, z):
        radius = np.sqrt((np.sqrt(x ** 2 + y ** 2) - 1500) ** 2 + z ** 2)
        return radius <= 500

    world.set_inside_vertices(inside_condition)
    return world


def run_benchmark(voxel_shape=(16, 16, 16), pixel_shape=(24, 24), res=3, parallel=4,
                  scene="simple", max_working_memory=1_000_000_000,
                  mst_spacing=None, detector_res=None):
    """Construct a projection matrix and return stable benchmark metrics."""
    if scene == "mst":
        detector_res = 5 if detector_res is None else detector_res
        world = build_mst_world(tuple(voxel_shape), voxel_spacing=mst_spacing,
                                detector_res=detector_res)
    elif scene == "simple":
        detector_res = 2 if detector_res is None else detector_res
        world = build_world(tuple(voxel_shape), tuple(pixel_shape),
                            detector_res=detector_res)
    else:
        raise ValueError(f"unknown benchmark scene: {scene!r}")
    start = time.perf_counter()
    world.set_projection_matrix(res=res, verbose=0, parallel=parallel, force=True,
                                max_working_memory=max_working_memory)
    elapsed = time.perf_counter() - start
    projection = next(iter(world.P_matrix.values()))
    camera = next(iter(world.cameras.values()))
    return {
        "scene": scene,
        "elapsed_seconds": elapsed,
        "voxel_shape": tuple(world.voxel.shape),
        "voxel_count": world.voxel.N,
        "pixel_shape": tuple(camera.screen.pixel_shape),
        "subvoxel_resolution": res,
        "detector_subpixel_resolution": detector_res,
        "parallel": parallel,
        "max_working_memory": max_working_memory,
        "projection_shape": projection.shape,
        "projection_nnz": projection.nnz,
        "projection_sum": float(projection.sum()),
    }


def _parse_triplet(value: str) -> tuple[int, int, int]:
    values = tuple(int(item) for item in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return values


def _parse_pair(value: str) -> tuple[int, int]:
    values = tuple(int(item) for item in value.split(","))
    if len(values) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated integers")
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("simple", "mst"), default="simple")
    parser.add_argument("--voxel-shape", type=_parse_triplet, default=(16, 16, 16))
    parser.add_argument("--pixel-shape", type=_parse_pair, default=(24, 24))
    parser.add_argument("--res", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-working-memory-mb", type=float, default=1000.0)
    parser.add_argument("--mst-spacing", type=float)
    parser.add_argument("--detector-res", type=int)
    args = parser.parse_args()

    metrics = run_benchmark(args.voxel_shape, args.pixel_shape, args.res, args.parallel,
                            scene=args.scene,
                            max_working_memory=int(args.max_working_memory_mb * 1_000_000),
                            mst_spacing=args.mst_spacing,
                            detector_res=args.detector_res)
    for key, value in metrics.items():
        print(f"{key}: {value}")
