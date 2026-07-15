"""Evaluate camera-coordinate chunking and local projection-column compression.

This is deliberately a post-processing prototype: it first constructs the
fine projection matrix, partitions voxel centres by their camera-space
direction (equivalently, projected detector pixel), and compresses only
columns that occur in the same bounded work chunk.  It therefore measures
whether the proposed grouping is accurate and useful without yet changing the
production projection builder.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "multi_pinhole_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "multi_pinhole_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

from multi_pinhole import Camera, Eye, Screen, Voxel, World


def optical_work_chunks(camera: Camera, eye_index: int, points: np.ndarray,
                        chunk_size: int) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Partition points by projected detector pixel, then bound each work chunk.

    The normalized coordinates are exactly the pinhole-coordinate ratios
    ``(X-X_eye)/(Z-f)`` and ``(Y-Y_eye)/(Z-f)``.  Multiplication by ``-f`` and
    addition of the principal point convert them to the same detector-plane
    coordinates used by :meth:`Eye.calc_rays`.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    points_in_camera = camera.world2camera(np.asarray(points, dtype=float))
    eye = camera.eyes[eye_index]
    points_in_eye = eye.camera2eye(points_in_camera)
    axial_distance = points_in_eye[:, 2]
    direction = np.full((points.shape[0], 2), np.nan, dtype=float)
    front = axial_distance > 0.0
    direction[front] = points_in_eye[front, :2] / axial_distance[front, None]

    projected_xy = np.full_like(direction, np.nan)
    projected_xy[front] = (-eye.focal_length * direction[front]
                           + eye.principal_point[None, :2])
    projected_uv = camera.screen.xy2uv(projected_xy)

    # The projected centre may lie just outside the detector while a finite
    # Eye footprint still overlaps it.  Clipping assigns such points to an
    # edge tile; genuinely empty columns are removed later by the compressor.
    pixel_indices = np.zeros((points.shape[0], 2), dtype=np.int64)
    pixel_indices[front] = np.floor(
        projected_uv[front] / camera.screen.pixel_size[None, :]
    ).astype(np.int64)
    pixel_indices[:, 0] = np.clip(pixel_indices[:, 0], 0, camera.screen.pixel_shape[0] - 1)
    pixel_indices[:, 1] = np.clip(pixel_indices[:, 1], 0, camera.screen.pixel_shape[1] - 1)

    # Within one detector-pixel tile, keep nearby direction and depth samples
    # adjacent before applying the memory-bound work chunk size.
    zoom_rate = np.full(points.shape[0], np.inf, dtype=float)
    zoom_rate[front] = 1.0 + eye.focal_length / axial_distance[front]
    order = np.lexsort((zoom_rate, direction[:, 1], direction[:, 0],
                        pixel_indices[:, 1], pixel_indices[:, 0], ~front))

    chunks = []
    start = 0
    while start < order.size:
        first = order[start]
        if not front[first]:
            break
        key = pixel_indices[first]
        stop = start + 1
        while stop < order.size and np.array_equal(pixel_indices[order[stop]], key):
            stop += 1
        tile = order[start:stop]
        chunks.extend(tile[offset:offset + chunk_size]
                      for offset in range(0, tile.size, chunk_size))
        start = stop

    return chunks, {
        "points_in_camera": points_in_camera,
        "direction": direction,
        "axial_distance": axial_distance,
        "zoom_rate": zoom_rate,
        "projected_xy": projected_xy,
        "projected_uv": projected_uv,
        "pixel_indices": pixel_indices,
        "front": front,
    }


def compress_projection_in_chunks(projection, chunks: list[np.ndarray], tolerance: float):
    """Construct ``P ~= Q A`` using local proportional-column groups.

    A candidate chunk is accepted as one group when every normalized column is
    within ``tolerance`` in L1 distance of the chunk response.  Failed chunks
    are recursively bisected in their optical ordering.  ``A`` is represented
    by one group index and one sensitivity weight per fine column.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    projection = projection.tocsc()
    sensitivity = np.asarray(projection.sum(axis=0)).ravel()
    group_index = np.full(projection.shape[1], -1, dtype=np.int64)
    weights = np.zeros(projection.shape[1], dtype=float)
    groups: list[np.ndarray] = []
    group_errors: list[float] = []

    def split_or_accept(indices):
        indices = np.asarray(indices, dtype=np.int64)
        indices = indices[sensitivity[indices] > 0.0]
        if indices.size == 0:
            return
        block = projection[:, indices].toarray()
        block_sensitivity = sensitivity[indices]
        response_sum = block.sum(axis=1)
        total_sensitivity = block_sensitivity.sum()
        normalized = block / block_sensitivity[None, :]
        representative = response_sum / total_sensitivity
        errors = np.abs(normalized - representative[:, None]).sum(axis=0)
        max_error = float(errors.max(initial=0.0))
        if max_error <= tolerance or indices.size == 1:
            group_number = len(groups)
            groups.append(indices)
            group_errors.append(max_error)
            group_index[indices] = group_number
            weights[indices] = block_sensitivity / total_sensitivity
            return
        middle = indices.size // 2
        split_or_accept(indices[:middle])
        split_or_accept(indices[middle:])

    for chunk in chunks:
        split_or_accept(chunk)

    # A positive column whose centre was not assigned to a front-facing work
    # chunk is kept as a singleton, so the approximation never drops signal.
    remaining = np.flatnonzero((sensitivity > 0.0) & (group_index < 0))
    for index in remaining:
        split_or_accept(np.array([index]))

    if groups:
        membership = sparse.csr_matrix(
            (np.ones(sum(group.size for group in groups), dtype=float),
             (np.concatenate(groups),
              np.repeat(np.arange(len(groups)), [group.size for group in groups]))),
            shape=(projection.shape[1], len(groups)),
        )
        compressed = (projection @ membership).tocsr()
    else:
        compressed = sparse.csr_matrix((projection.shape[0], 0), dtype=projection.dtype)

    return {
        "projection": compressed,
        "group_index": group_index,
        "weights": weights,
        "groups": groups,
        "group_errors": np.asarray(group_errors),
        "sensitivity": sensitivity,
    }


def project_compressed(compression, emission):
    """Apply the implicit restriction ``A`` followed by compressed ``Q``."""
    emission = np.asarray(emission)
    active = compression["group_index"] >= 0
    reduced = np.bincount(
        compression["group_index"][active],
        weights=compression["weights"][active] * emission[active],
        minlength=compression["projection"].shape[1],
    )
    return np.asarray(compression["projection"] @ reduced).ravel()


def build_simple_world(rotation_degrees: float = 0.0,
                       axial_range=(60.0, 140.0),
                       voxel_shape=(24, 24, 32), pixel_shape=(12, 12)) -> World:
    """Create a wall-free source box viewed head-on or from 30 degrees above."""
    rotation = Rotation.from_euler("x", rotation_degrees, degrees=True).as_matrix()
    focal_length = 20.0
    axial_min, axial_max = map(float, axial_range)
    if axial_min <= 0.0 or axial_max <= axial_min:
        raise ValueError("axial_range must be positive and increasing")
    centre_in_camera = np.array([
        0.0, 0.0, focal_length + 0.5 * (axial_min + axial_max),
    ])
    half_width_in_camera = np.array([18.0, 18.0, 0.5 * (axial_max - axial_min)])
    centre = rotation.T @ centre_in_camera
    # Voxel is world-axis aligned.  Use the world AABB of the desired
    # camera-aligned source box; actual camera-space Z/f is recorded below.
    half_width = np.abs(rotation.T) @ half_width_in_camera
    ranges = tuple((float(value - width), float(value + width))
                   for value, width in zip(centre, half_width))

    voxel = Voxel.uniform_voxel(ranges=ranges, shape=voxel_shape)
    eye = Eye(position=(0.0, 0.0), focal_length=focal_length, eye_size=1.2,
              eye_shape="circle")
    screen = Screen(screen_shape="rectangle", screen_size=(12.0, 12.0),
                    pixel_shape=pixel_shape, subpixel_resolution=1)
    camera = Camera(eyes=[eye], apertures=[], screen=screen,
                    camera_position=(0.0, 0.0, 0.0), rotation_matrix=rotation)
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    return world


def emission_profiles(points):
    """Non-negative profiles used to test the compressed forward operator."""
    centred = points - points.mean(axis=0)
    scales = np.maximum(np.ptp(points, axis=0), 1.0)
    normalized = centred / scales
    radius2 = np.sum((centred / (0.28 * scales)) ** 2, axis=1)
    return {
        "constant": np.ones(points.shape[0]),
        "linear": 1.0 + 0.7 * normalized[:, 0] - 0.35 * normalized[:, 1],
        "square": normalized[:, 0] ** 2 + 0.4 * normalized[:, 2] ** 2,
        "gaussian": np.exp(-0.5 * radius2),
    }


def run_evaluation(output_dir: Path, rotations=(0.0, 30.0),
                   axial_ranges=((60.0, 140.0), (200.0, 400.0), (800.0, 1200.0)),
                   tolerances=(0.01, 0.03, 0.1, 0.2, 0.5, 1.0),
                   chunk_size=256, voxel_shape=(24, 24, 32), pixel_shape=(12, 12)):
    """Run the simple-model comparison and save a metrics table and figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.perf_counter()

    for rotation in rotations:
        for requested_axial_range in axial_ranges:
            world = build_simple_world(
                rotation, axial_range=requested_axial_range,
                voxel_shape=voxel_shape, pixel_shape=pixel_shape,
            )
            world.set_projection_matrix(res=1, partial_res=1, verbose=0,
                                        parallel=1, force=True)
            projection = world.P_matrix[0].tocsr()
            points = world.voxel.gravity_center
            chunks, optical = optical_work_chunks(
                world.cameras[0], 0, points, chunk_size=chunk_size,
            )
            # Verify that the proposed normalized coordinates reproduce the actual
            # ray projection before using them to define chunks.
            rays = world.cameras[0].eyes[0].calc_rays(optical["points_in_camera"])
            if not np.allclose(optical["projected_xy"], rays.XY, equal_nan=True):
                raise AssertionError("camera-space chunk coordinates disagree with Eye.calc_rays")

            profiles = emission_profiles(points)
            reference = {name: np.asarray(projection @ values).ravel()
                         for name, values in profiles.items()}
            active_columns = int(np.count_nonzero(np.asarray(projection.sum(axis=0)).ravel()))
            z_over_f = optical["axial_distance"][optical["front"]] \
                / world.cameras[0].eyes[0].focal_length

            for tolerance in tolerances:
                compression = compress_projection_in_chunks(projection, chunks, tolerance)
                group_count = len(compression["groups"])
                for name, values in profiles.items():
                    image = project_compressed(compression, values)
                    ref = reference[name]
                    rows.append({
                        "rotation_degrees": rotation,
                        "requested_axial_min": requested_axial_range[0],
                        "requested_axial_max": requested_axial_range[1],
                        "minimum_z_over_f": float(z_over_f.min()),
                        "median_z_over_f": float(np.median(z_over_f)),
                        "maximum_z_over_f": float(z_over_f.max()),
                        "tolerance": tolerance,
                        "profile": name,
                        "voxel_count": world.voxel.N_voxel,
                        "active_columns": active_columns,
                        "work_chunk_count": len(chunks),
                        "group_count": group_count,
                        "column_compression_ratio": active_columns / max(group_count, 1),
                        "projection_nnz": projection.nnz,
                        "compressed_nnz": compression["projection"].nnz,
                        "nnz_compression_ratio": projection.nnz / max(compression["projection"].nnz, 1),
                        "maximum_group_error": float(compression["group_errors"].max(initial=0.0)),
                        "relative_l1": float(np.linalg.norm(image - ref, ord=1)
                                             / np.linalg.norm(ref, ord=1)),
                        "relative_l2": float(np.linalg.norm(image - ref)
                                             / np.linalg.norm(ref)),
                        "relative_flux": float(abs(image.sum() - ref.sum()) / abs(ref.sum())),
                    })

    csv_path = output_dir / "projection_column_compression.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    case_keys = [(rotation, tuple(axial_range))
                 for rotation in rotations for axial_range in axial_ranges]
    for rotation, axial_range in case_keys:
        selected = [row for row in rows
                    if row["rotation_degrees"] == rotation
                    and row["requested_axial_min"] == axial_range[0]
                    and row["requested_axial_max"] == axial_range[1]
                    and row["profile"] == "gaussian"]
        label = (f"rot={rotation:g}°, Z/f≈"
                 f"{selected[0]['median_z_over_f']:.1f}")
        axes[0, 0].plot([row["tolerance"] for row in selected],
                        [row["column_compression_ratio"] for row in selected],
                        marker="o", label=label)
        axes[0, 1].plot([row["tolerance"] for row in selected],
                        [row["nnz_compression_ratio"] for row in selected],
                        marker="o", label=label)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("maximum normalized-column L1 error")
    axes[0, 0].set_ylabel("active columns / compressed groups")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("maximum normalized-column L1 error")
    axes[0, 1].set_ylabel("projection nnz / compressed nnz")

    for rotation, axial_range in case_keys:
        selected = [row for row in rows
                    if row["rotation_degrees"] == rotation
                    and row["requested_axial_min"] == axial_range[0]
                    and row["requested_axial_max"] == axial_range[1]
                    and row["profile"] == "gaussian"]
        label = (f"rot={rotation:g}°, Z/f≈"
                 f"{selected[0]['median_z_over_f']:.1f}")
        axes[1, 0].plot(
            [row["column_compression_ratio"] for row in selected],
            [max(row["relative_l2"], 1e-16) for row in selected],
            marker="o", label=label,
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("column compression ratio")
    axes[1, 0].set_ylabel("Gaussian image relative L2")

    depth_tolerances = tuple(tolerances[-3:])
    for tolerance in depth_tolerances:
        selected = sorted(
            (row for row in rows
             if row["rotation_degrees"] == 0.0
             and row["profile"] == "gaussian"
             and row["tolerance"] == tolerance),
            key=lambda row: row["median_z_over_f"],
        )
        axes[1, 1].plot(
            [row["median_z_over_f"] for row in selected],
            [row["column_compression_ratio"] for row in selected],
            marker="o", label=f"tolerance={tolerance:g}",
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("median Z/f (rotation=0°)")
    axes[1, 1].set_ylabel("column compression ratio")
    for axis in axes.ravel():
        axis.grid(which="both", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    figure_path = output_dir / "projection_column_compression.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    return {
        "rows": rows,
        "csv_path": csv_path,
        "figure_path": figure_path,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _parse_floats(value):
    return tuple(float(item) for item in value.split(","))


def _parse_ints(value):
    return tuple(int(item) for item in value.split(","))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("benchmarks/output/projection_column_compression"))
    parser.add_argument("--rotations", type=_parse_floats, default=(0.0, 30.0))
    parser.add_argument("--axial-ranges", type=str, default="60:140,200:400,800:1200")
    parser.add_argument("--tolerances", type=_parse_floats,
                        default=(0.01, 0.03, 0.1, 0.2, 0.5, 1.0))
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--voxel-shape", type=_parse_ints, default=(24, 24, 32))
    parser.add_argument("--pixel-shape", type=_parse_ints, default=(12, 12))
    args = parser.parse_args()
    axial_ranges = tuple(
        tuple(float(bound) for bound in item.split(":"))
        for item in args.axial_ranges.split(",")
    )
    result = run_evaluation(args.output_dir, rotations=args.rotations,
                            axial_ranges=axial_ranges,
                            tolerances=args.tolerances, chunk_size=args.chunk_size,
                            voxel_shape=args.voxel_shape, pixel_shape=args.pixel_shape)
    print(f"elapsed_seconds: {result['elapsed_seconds']:.3f}")
    print(f"csv: {result['csv_path']}")
    print(f"figure: {result['figure_path']}")
