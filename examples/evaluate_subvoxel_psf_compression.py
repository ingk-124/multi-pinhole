"""Step-by-step toy evaluation of subvoxel PSF compression.

The experiment exposes the production factorization

    P = I S ~= Q R S = Q A,

where ``I`` contains one detector PSF per visible subvoxel sample and ``S``
contains trilinear interpolation plus the subvoxel volume weight.  Samples are
bucketed in Eye angular coordinates X/Z and Y/Z before their normalized PSFs
are clustered.  The script compares the reconstructed ``I @ S`` and ``Q @ A``
against the existing ``World.set_projection_matrix`` result and writes a set
of diagnostic pyplot figures in calculation order.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time
import tracemalloc

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "multi_pinhole_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "multi_pinhole_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from multi_pinhole import Aperture, Camera, Eye, Screen, Voxel, World
from multi_pinhole.projection import make_optical_binning


FOCAL_LENGTH = 20.0
EYE_DIAMETER = 1.2
SCREEN_SIZE = (12.0, 12.0)
APERTURE_Z = 50.0
APERTURE_SIZE = 8.0
APERTURE_MAX_SIZE = 42.0


@dataclass
class ToyProblem:
    """Matrices and geometry for one source resolution."""

    world: World
    resolution: int
    points: np.ndarray
    owner_voxel: np.ndarray
    visibility_state: np.ndarray
    point_visible: np.ndarray
    active: np.ndarray
    active_points: np.ndarray
    S: sparse.csr_matrix
    I: sparse.csr_matrix
    P_existing: sparse.csr_matrix
    P_manual: sparse.csr_matrix
    eye_coordinates: np.ndarray
    xi_eta: np.ndarray
    projected_uv: np.ndarray
    optical_chunks: list[np.ndarray]
    work_chunks: list[np.ndarray]
    optical_bin_id: np.ndarray


@dataclass
class Compression:
    """One factorization I ~= Q R and P ~= Q A."""

    Q: sparse.csr_matrix
    R: sparse.csr_matrix
    A: sparse.csr_matrix
    P_approx: sparse.csr_matrix
    group_index: np.ndarray
    group_members: list[np.ndarray]
    group_max_l1: np.ndarray
    group_max_relative_l2: np.ndarray
    sensitivity: np.ndarray
    elapsed_seconds: float
    metric: str
    tolerance: float
    algorithm: str


def build_toy_world(axial_distance: float = 100.0,
                    voxel_shape: tuple[int, int, int] = (10, 10, 6),
                    pixel_shape: tuple[int, int] = (24, 24),
                    with_aperture: bool = False,
                    detector_resolution: int = 1) -> World:
    """Build a camera-aligned source box with an optional circular aperture."""
    source_center_z = FOCAL_LENGTH + axial_distance
    depth_half_width = min(0.35 * axial_distance, 40.0)
    voxel = Voxel.uniform_voxel(
        ranges=((-20.0, 20.0), (-20.0, 20.0),
                (source_center_z - depth_half_width,
                 source_center_z + depth_half_width)),
        shape=voxel_shape,
    )
    eye = Eye(
        position=(0.0, 0.0), focal_length=FOCAL_LENGTH,
        eye_type="pinhole", eye_shape="circle", eye_size=EYE_DIAMETER,
    )
    screen = Screen(
        screen_shape="rectangle", screen_size=SCREEN_SIZE,
        pixel_shape=pixel_shape, subpixel_resolution=detector_resolution,
    )
    apertures = []
    if with_aperture:
        apertures = [Aperture(
            shape="circle", size=APERTURE_SIZE,
            position=(0.0, 0.0, APERTURE_Z),
            resolution=48, max_size=APERTURE_MAX_SIZE,
        )]
    camera = Camera(
        eyes=[eye], apertures=apertures, screen=screen,
        camera_position=(0.0, 0.0, 0.0),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    return world


def _sparse_nbytes(matrix) -> int:
    matrix = matrix.tocsr()
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def _optical_coordinates(camera: Camera, eye_index: int, points: np.ndarray):
    points_camera = camera.world2camera(points)
    points_eye = camera.eyes[eye_index].camera2eye(points_camera)
    if np.any(points_eye[:, 2] <= 0.0):
        raise ValueError("toy source points must all lie in front of the Eye")
    xi_eta = points_eye[:, :2] / points_eye[:, 2, None]
    projected_xy = (-camera.eyes[eye_index].focal_length * xi_eta
                    + camera.eyes[eye_index].principal_point[None, :2])
    projected_uv = camera.screen.xy2uv(projected_xy)
    rays = camera.eyes[eye_index].calc_rays(points_camera)
    if not np.allclose(projected_xy, rays.XY, rtol=1e-12, atol=1e-12):
        raise AssertionError("X/Z optical coordinates disagree with Eye.calc_rays")
    return points_eye, xi_eta, projected_uv


def make_optical_chunks(camera: Camera, eye_index: int, points: np.ndarray,
                        max_chunk_size: int = 256,
                        optical_bin_width_pixels=1.0):
    """Bucket arbitrary point indices in Eye X/Z, Y/Z coordinates."""
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be positive")
    points_eye, xi_eta, projected_uv = _optical_coordinates(camera, eye_index, points)
    binning = make_optical_binning(
        camera, eye_index, points,
        bin_width_pixels=optical_bin_width_pixels,
        max_scope_samples=max_chunk_size,
    )
    chunks = binning.scopes()
    chunk_bin_ids: list[int] = []
    sample_bin_id = np.empty(points.shape[0], dtype=np.int64)
    next_bin_id = -1
    previous_key = None
    for chunk, key in zip(chunks, binning.scope_keys):
        if previous_key is None or not np.array_equal(key, previous_key):
            next_bin_id += 1
            previous_key = key
        sample_bin_id[chunk] = next_bin_id
        chunk_bin_ids.append(next_bin_id)
    return chunks, sample_bin_id, np.asarray(chunk_bin_ids), points_eye, xi_eta, projected_uv


def pack_optical_chunks(optical_chunks: list[np.ndarray],
                        max_work_chunk_size: int) -> list[np.ndarray]:
    """Pack complete optical compression scopes into memory work chunks.

    ``optical_chunks`` remain the independent grouping scopes.  Several of
    them may share one PSF calculation, but samples from different scopes are
    never clustered together.  Oversized optical bins have already been split
    by :func:`make_optical_chunks`.
    """
    if max_work_chunk_size < 1:
        raise ValueError("max_work_chunk_size must be positive")
    work_chunks: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    pending_size = 0
    for optical_chunk in optical_chunks:
        optical_chunk = np.asarray(optical_chunk, dtype=np.int64)
        if pending and pending_size + optical_chunk.size > max_work_chunk_size:
            work_chunks.append(np.concatenate(pending))
            pending = []
            pending_size = 0
        pending.append(optical_chunk)
        pending_size += optical_chunk.size
        if pending_size >= max_work_chunk_size:
            work_chunks.append(np.concatenate(pending))
            pending = []
            pending_size = 0
    if pending:
        work_chunks.append(np.concatenate(pending))
    return work_chunks


def build_problem(world: World, resolution: int, max_chunk_size: int = 256,
                  optical_bin_width_pixels=1.0) -> ToyProblem:
    """Construct production P plus explicit visible I and S for one resolution."""
    if resolution < 1:
        raise ValueError("resolution must be positive")
    voxel_indices = np.arange(world.voxel.N)
    samples_per_voxel = resolution ** 3
    points = world.voxel.get_sub_voxel_centers(voxel_indices, res=resolution)
    owner_voxel = np.repeat(voxel_indices, samples_per_voxel)
    S_all = world.voxel.sub_voxel_interpolator_from_centers(
        voxel_indices, res=resolution, points=points,
    ).tocsr()

    world.find_visible_voxels(force=True, verbose=0)
    visibility_state = world.visible_voxels[0][0].astype(np.int8)
    point_visible = world.find_visible_points(
        points, camera_idx=0, eye_idx=0, verbose=0,
    )[0]
    owner_state = visibility_state[owner_voxel]
    active = (owner_state == 2) | ((owner_state == 1) & point_visible)
    active_points = points[active]
    S = S_all[active].tocsr()

    camera = world.cameras[0]
    I_subpixel = camera.calc_image_vec(
        0, points=active_points, verbose=0, check_visibility=False,
    ).tocsr()
    I = (camera.screen.transform_matrix @ I_subpixel).tocsr()
    P_manual = (I @ S).tocsr()
    world.set_projection_matrix(
        res=resolution, partial_res=resolution, verbose=0,
        parallel=1, force=True,
    )
    P_existing = world.P_matrix[0].tocsr()

    chunks, bin_id, _, eye_coordinates, xi_eta, projected_uv = make_optical_chunks(
        camera, 0, active_points, max_chunk_size=max_chunk_size,
        optical_bin_width_pixels=optical_bin_width_pixels,
    )
    work_chunks = pack_optical_chunks(chunks, max_work_chunk_size=max_chunk_size)
    return ToyProblem(
        world=world, resolution=resolution, points=points,
        owner_voxel=owner_voxel, visibility_state=visibility_state,
        point_visible=point_visible, active=active,
        active_points=active_points, S=S, I=I,
        P_existing=P_existing, P_manual=P_manual,
        eye_coordinates=eye_coordinates, xi_eta=xi_eta,
        projected_uv=projected_uv, optical_chunks=chunks,
        work_chunks=work_chunks,
        optical_bin_id=bin_id,
    )


def project_by_optical_work_chunks(problem: ToyProblem,
                                   work_chunks: list[np.ndarray] | None = None):
    """Rebuild the ordinary sparse P after optical-bin reordering.

    This deliberately performs no PSF grouping: every work chunk evaluates
    ``I_chunk @ S_chunk`` and the contributions are summed.  It isolates the
    correctness of optical ordering and batching from compression error.
    """
    chunks = problem.work_chunks if work_chunks is None else work_chunks
    camera = problem.world.cameras[0]
    result = sparse.csr_matrix(problem.P_existing.shape)
    for chunk in chunks:
        chunk = np.asarray(chunk, dtype=np.int64)
        if chunk.size == 0:
            continue
        I_subpixel_chunk = camera.calc_image_vec(
            0, points=problem.active_points[chunk], verbose=0,
            check_visibility=False,
        ).tocsr()
        I_chunk = camera.screen.transform_matrix @ I_subpixel_chunk
        result += I_chunk @ problem.S[chunk]
    return result.tocsr()


def _distances(H: np.ndarray, q: np.ndarray):
    difference = H - q[:, None]
    l1 = np.abs(difference).sum(axis=0)
    h_norm = np.linalg.norm(H, axis=0)
    relative_l2 = np.linalg.norm(difference, axis=0) / np.maximum(h_norm, 1e-30)
    return l1, relative_l2


def _metric_values(H: np.ndarray, q: np.ndarray, metric: str):
    l1, relative_l2 = _distances(H, q)
    if metric == "l1":
        return l1
    if metric == "relative_l2":
        return relative_l2
    raise ValueError(f"unknown metric {metric!r}")


def _recursive_groups(H: np.ndarray, sensitivity: np.ndarray,
                      tolerance: float, metric: str):
    groups: list[np.ndarray] = []

    def split(indices):
        indices = np.asarray(indices, dtype=np.int64)
        block = H[:, indices]
        weights = sensitivity[indices]
        q = (block * weights[None, :]).sum(axis=1) / weights.sum()
        errors = _metric_values(block, q, metric)
        if errors.max(initial=0.0) <= tolerance or indices.size == 1:
            groups.append(indices)
            return

        first_local = int(np.argmax(errors))
        first = block[:, first_local]
        first_distances = _metric_values(block, first, metric)
        second_local = int(np.argmax(first_distances))
        second = block[:, second_local]
        distance_first = _metric_values(block, first, metric)
        distance_second = _metric_values(block, second, metric)
        left_mask = distance_first <= distance_second
        if np.all(left_mask) or not np.any(left_mask):
            order = np.argsort(errors, kind="stable")
            left_mask = np.zeros(indices.size, dtype=bool)
            left_mask[order[:indices.size // 2]] = True
        split(indices[left_mask])
        split(indices[~left_mask])

    split(np.arange(H.shape[1]))
    return groups


def _leader_groups(H: np.ndarray, sensitivity: np.ndarray,
                   tolerance: float, metric: str):
    """One input scan with exact member rechecks after each mean update."""
    groups: list[list[int]] = []
    representatives: list[np.ndarray] = []
    for point in range(H.shape[1]):
        if not groups:
            groups.append([point])
            representatives.append(H[:, point].copy())
            continue
        candidate_distance = np.asarray([
            _metric_values(H[:, point:point + 1], q, metric)[0]
            for q in representatives
        ])
        accepted = False
        for group_number in np.argsort(candidate_distance, kind="stable"):
            proposed = np.asarray(groups[group_number] + [point], dtype=np.int64)
            weights = sensitivity[proposed]
            q = (H[:, proposed] * weights[None, :]).sum(axis=1) / weights.sum()
            if _metric_values(H[:, proposed], q, metric).max(initial=0.0) <= tolerance:
                groups[group_number].append(point)
                representatives[group_number] = q
                accepted = True
                break
        if not accepted:
            groups.append([point])
            representatives.append(H[:, point].copy())
    return [np.asarray(group, dtype=np.int64) for group in groups]


def compress_problem(problem: ToyProblem, tolerance: float = 0.1,
                     metric: str = "relative_l2",
                     algorithm: str = "recursive") -> Compression:
    """Compress normalized columns of I independently inside optical chunks."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    start = time.perf_counter()
    I_csc = problem.I.tocsc()
    sensitivity = np.asarray(I_csc.sum(axis=0)).ravel()
    positive = sensitivity > 0.0
    group_index = np.full(I_csc.shape[1], -1, dtype=np.int64)
    group_members: list[np.ndarray] = []
    q_columns: list[sparse.csc_matrix] = []
    group_l1: list[float] = []
    group_l2: list[float] = []

    for chunk in problem.optical_chunks:
        chunk = np.asarray(chunk, dtype=np.int64)
        chunk = chunk[positive[chunk]]
        if chunk.size == 0:
            continue
        block = I_csc[:, chunk].toarray()
        H = block / sensitivity[chunk][None, :]
        if algorithm == "recursive":
            local_groups = _recursive_groups(H, sensitivity[chunk], tolerance, metric)
        elif algorithm == "leader":
            local_groups = _leader_groups(H, sensitivity[chunk], tolerance, metric)
        else:
            raise ValueError(f"unknown algorithm {algorithm!r}")

        for local in local_groups:
            members = chunk[local]
            total_sensitivity = sensitivity[members].sum()
            q = np.asarray(I_csc[:, members].sum(axis=1)).ravel() / total_sensitivity
            local_H = H[:, local]
            l1, relative_l2 = _distances(local_H, q)
            number = len(group_members)
            group_members.append(members)
            group_index[members] = number
            q_columns.append(sparse.csc_matrix(q[:, None]))
            group_l1.append(float(l1.max(initial=0.0)))
            group_l2.append(float(relative_l2.max(initial=0.0)))

    if q_columns:
        Q = sparse.hstack(q_columns, format="csr")
        member_columns = np.concatenate(group_members)
        member_groups = np.repeat(
            np.arange(len(group_members)), [group.size for group in group_members],
        )
        R = sparse.csr_matrix(
            (sensitivity[member_columns], (member_groups, member_columns)),
            shape=(len(group_members), I_csc.shape[1]),
        )
    else:
        Q = sparse.csr_matrix((I_csc.shape[0], 0))
        R = sparse.csr_matrix((0, I_csc.shape[1]))
    A = (R @ problem.S).tocsr()
    P_approx = (Q @ A).tocsr()
    return Compression(
        Q=Q, R=R, A=A, P_approx=P_approx,
        group_index=group_index, group_members=group_members,
        group_max_l1=np.asarray(group_l1),
        group_max_relative_l2=np.asarray(group_l2),
        sensitivity=sensitivity,
        elapsed_seconds=time.perf_counter() - start,
        metric=metric, tolerance=tolerance, algorithm=algorithm,
    )


def emission_profiles(points: np.ndarray):
    centered = points - points.mean(axis=0)
    scale = np.maximum(np.ptp(points, axis=0), 1.0)
    normalized = centered / scale
    radius2 = np.sum((centered / (0.28 * scale)) ** 2, axis=1)
    return {
        "constant": np.ones(points.shape[0]),
        "linear": 1.0 + 0.8 * normalized[:, 0] - 0.4 * normalized[:, 1],
        "square": normalized[:, 0] ** 2 + 0.5 * normalized[:, 2] ** 2,
        "gaussian": np.exp(-0.5 * radius2),
        "signed": normalized[:, 0] - 0.6 * normalized[:, 1] + 0.3 * normalized[:, 2],
    }


def _matrix_metrics(reference, approximation):
    reference = reference.tocsc()
    approximation = approximation.tocsc()
    difference = (approximation - reference).tocsc()
    reference_norm = sparse.linalg.norm(reference)
    sensitivity = np.asarray(reference.sum(axis=0)).ravel()
    approximate_sensitivity = np.asarray(approximation.sum(axis=0)).ravel()
    positive = sensitivity > 0.0
    column_l1 = []
    column_relative_l2 = []
    for column in np.flatnonzero(positive):
        ref = reference[:, column].toarray().ravel()
        app = approximation[:, column].toarray().ravel()
        ref_shape = ref / sensitivity[column]
        app_shape = app / max(approximate_sensitivity[column], 1e-30)
        column_l1.append(np.abs(app_shape - ref_shape).sum())
        column_relative_l2.append(
            np.linalg.norm(app - ref) / max(np.linalg.norm(ref), 1e-30)
        )
    sensitivity_scale = max(float(np.max(np.abs(sensitivity), initial=0.0)), 1e-30)
    return {
        "relative_frobenius": float(sparse.linalg.norm(difference) / max(reference_norm, 1e-30)),
        "max_column_shape_l1": float(np.max(column_l1, initial=0.0)),
        "max_column_relative_l2": float(np.max(column_relative_l2, initial=0.0)),
        "max_column_sum_error_scaled": float(
            np.max(np.abs(approximate_sensitivity - sensitivity), initial=0.0) / sensitivity_scale
        ),
    }


def evaluate_profiles(problem: ToyProblem, compression: Compression):
    rows = []
    profiles = emission_profiles(problem.world.voxel.gravity_center)
    subvoxel_values_by_profile = {}
    for name, values in profiles.items():
        exact = np.asarray(problem.P_existing @ values).ravel()
        approximate = np.asarray(compression.Q @ (compression.A @ values)).ravel()
        difference = approximate - exact
        subvoxel_values = np.asarray(problem.S @ values).ravel()
        subvoxel_values_by_profile[name] = subvoxel_values
        absolute_source_scale = float(np.sum(
            compression.sensitivity * np.abs(subvoxel_values)
        ))
        rows.append({
            "profile": name,
            "relative_l1": float(np.linalg.norm(difference, ord=1)
                                 / max(np.linalg.norm(exact, ord=1), 1e-30)),
            "relative_l2": float(np.linalg.norm(difference)
                                 / max(np.linalg.norm(exact), 1e-30)),
            "max_pixel_error_scaled": float(np.max(np.abs(difference), initial=0.0)
                                            / max(np.max(np.abs(exact), initial=0.0), 1e-30)),
            "signed_sum_error": float(abs(approximate.sum() - exact.sum())),
            "absolute_weighted_l1": float(np.linalg.norm(difference, ord=1)
                                          / max(absolute_source_scale, 1e-30)),
        })
    return rows, profiles, subvoxel_values_by_profile


def _box_edges(ranges):
    lower = np.array([axis[0] for axis in ranges])
    upper = np.array([axis[1] for axis in ranges])
    corners = np.array([
        [x, y, z] for x in (lower[0], upper[0])
        for y in (lower[1], upper[1]) for z in (lower[2], upper[2])
    ])
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(corners[i] != corners[j]) == 1:
                edges.append((corners[i], corners[j]))
    return corners, edges


def plot_geometry(problem: ToyProblem, output: Path):
    world = problem.world
    camera = world.cameras[0]
    eye = camera.eyes[0]
    ranges = tuple((axis[0], axis[-1]) for axis in
                   (world.voxel.x_axis, world.voxel.y_axis, world.voxel.z_axis))
    corners, edges = _box_edges(ranges)
    fig = plt.figure(figsize=(13, 4.2))
    ax3d = fig.add_subplot(131, projection="3d")
    ax_xz = fig.add_subplot(132)
    ax_yz = fig.add_subplot(133)
    for p0, p1 in edges:
        ax3d.plot(*np.vstack([p0, p1]).T, color="tab:blue", lw=1)
        ax_xz.plot([p0[2], p1[2]], [p0[0], p1[0]], color="tab:blue", lw=1)
        ax_yz.plot([p0[2], p1[2]], [p0[1], p1[1]], color="tab:blue", lw=1)
    ax3d.scatter(0, 0, 0, marker="s", color="black", label="camera/screen")
    ax3d.scatter(*eye.position, marker="o", color="tab:red", label="Eye")
    ax_xz.scatter([0, eye.position[2]], [0, eye.position[0]], c=["black", "tab:red"])
    ax_yz.scatter([0, eye.position[2]], [0, eye.position[1]], c=["black", "tab:red"])
    for corner in corners:
        ax3d.plot([eye.position[0], corner[0]], [eye.position[1], corner[1]],
                  [eye.position[2], corner[2]], color="0.65", lw=0.5, alpha=0.6)
        ax_xz.plot([eye.position[2], corner[2]], [eye.position[0], corner[0]],
                   color="0.7", lw=0.5)
        ax_yz.plot([eye.position[2], corner[2]], [eye.position[1], corner[1]],
                   color="0.7", lw=0.5)
    if camera.apertures:
        z = camera.apertures[0].position[2]
        half = APERTURE_MAX_SIZE / 2
        radius = APERTURE_SIZE / 2
        ax3d.plot([-half, half, half, -half, -half],
                  [-half, -half, half, half, -half], [z] * 5,
                  color="tab:orange", lw=1.2, label="aperture plate")
        angle = np.linspace(0.0, 2.0 * np.pi, 128)
        ax3d.plot(radius * np.cos(angle), radius * np.sin(angle),
                  np.full(angle.size, z), color="tab:red", lw=1.2,
                  label="aperture hole")
        for axis in (ax_xz, ax_yz):
            axis.plot([z, z], [-half, -radius], color="tab:orange", lw=2)
            axis.plot([z, z], [radius, half], color="tab:orange", lw=2)
    ax3d.set_xlabel("camera X [mm]")
    ax3d.set_ylabel("camera Y [mm]")
    ax3d.set_zlabel("camera Z [mm]")
    ax3d.set_title("camera, Eye, voxel bounds")
    ax3d.legend(fontsize=8)
    ax_xz.set(xlabel="camera Z [mm]", ylabel="camera X [mm]", title="X-Z side view")
    ax_yz.set(xlabel="camera Z [mm]", ylabel="camera Y [mm]", title="Y-Z side view")
    for axis in (ax_xz, ax_yz):
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_optical_distribution(problem: ToyProblem, output: Path):
    xi_eta = problem.xi_eta
    depth = problem.eye_coordinates[:, 2] / FOCAL_LENGTH
    _, all_xi_eta, _ = _optical_coordinates(
        problem.world.cameras[0], 0, problem.points,
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].scatter(all_xi_eta[~problem.active, 0], all_xi_eta[~problem.active, 1],
                    color="0.75", s=2, alpha=0.35, label="culled")
    scatter = axes[0].scatter(xi_eta[:, 0], xi_eta[:, 1], c=depth,
                              s=4, cmap="viridis", alpha=0.65)
    axes[0].set(xlabel=r"$X_e/Z_e$", ylabel=r"$Y_e/Z_e$",
                title="visible subvoxel samples")
    axes[0].legend(fontsize=8)
    fig.colorbar(scatter, ax=axes[0], label=r"$Z_e/f$")
    axes[1].hexbin(xi_eta[:, 0], xi_eta[:, 1], gridsize=30,
                   bins="log", mincnt=1, cmap="magma")
    axes[1].set(xlabel=r"$X_e/Z_e$", ylabel=r"$Y_e/Z_e$",
                title="optical-coordinate density")
    counts = np.bincount(problem.optical_bin_id)
    axes[2].hist(counts, bins=min(30, max(5, counts.size)), color="tab:blue", alpha=0.8)
    axes[2].axvline(max(chunk.size for chunk in problem.work_chunks),
                    color="tab:red", label="largest work chunk")
    axes[2].set(xlabel="samples per optical bin", ylabel="bin count",
                title="X/Z, Y/Z bucket occupancy")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_visibility(problem: ToyProblem, output: Path):
    voxel = problem.world.voxel
    state = problem.visibility_state.reshape(voxel.shape)
    samples_per_voxel = problem.resolution ** 3
    visible_fraction = problem.point_visible.reshape(
        *voxel.shape, samples_per_voxel,
    ).mean(axis=-1)
    middle_y = voxel.shape[1] // 2
    middle_z = voxel.shape[2] // 2
    state_cmap = matplotlib.colors.ListedColormap(["0.65", "tab:orange", "tab:green"])
    state_norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], state_cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    extent_xz = [voxel.z_axis[0], voxel.z_axis[-1], voxel.x_axis[0], voxel.x_axis[-1]]
    extent_xy = [voxel.y_axis[0], voxel.y_axis[-1], voxel.x_axis[0], voxel.x_axis[-1]]
    im0 = axes[0].imshow(state[:, middle_y, :], origin="lower", aspect="auto",
                         extent=extent_xz, cmap=state_cmap, norm=state_norm)
    axes[0].set(xlabel="camera Z [mm]", ylabel="camera X [mm]",
                title=f"voxel state, Y index={middle_y}")
    colorbar = fig.colorbar(im0, ax=axes[0], ticks=(0, 1, 2))
    colorbar.ax.set_yticklabels(("invisible", "partial", "full"))
    axes[1].imshow(state[:, :, middle_z], origin="lower", aspect="equal",
                   extent=extent_xy, cmap=state_cmap, norm=state_norm)
    axes[1].set(xlabel="camera Y [mm]", ylabel="camera X [mm]",
                title=f"voxel state, Z index={middle_z}")
    im2 = axes[2].imshow(visible_fraction[:, middle_y, :], origin="lower", aspect="auto",
                         extent=extent_xz, cmap="viridis", vmin=0, vmax=1)
    axes[2].set(xlabel="camera Z [mm]", ylabel="camera X [mm]",
                title=f"subvoxel visible fraction, Y index={middle_y}")
    fig.colorbar(im2, ax=axes[2], label="visible fraction")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_compression_detail(problem: ToyProblem, compression: Compression, output: Path):
    screen_shape = tuple(problem.world.cameras[0].screen.pixel_shape)
    largest = int(np.argmax([group.size for group in compression.group_members]))
    members = compression.group_members[largest]
    q = compression.Q[:, largest].toarray().ravel()
    H = problem.I[:, members].toarray() / compression.sensitivity[members][None, :]
    l1, relative_l2 = _distances(H, q)
    farthest = int(np.argmax(relative_l2))
    difference = H[:, farthest] - q
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    images = (q, H[:, farthest], difference)
    titles = ("representative q", "farthest normalized PSF",
              "farthest - representative")
    cmaps = ("viridis", "viridis", "coolwarm")
    for axis, image, title, cmap in zip(axes.ravel()[:3], images, titles, cmaps):
        vmax = None if cmap != "coolwarm" else np.max(np.abs(image))
        im = axis.imshow(image.reshape(screen_shape), origin="upper", cmap=cmap,
                         vmin=None if vmax is None else -vmax, vmax=vmax)
        axis.set_title(title)
        fig.colorbar(im, ax=axis, fraction=0.046)
    axes[1, 1].scatter(problem.xi_eta[members, 0], problem.xi_eta[members, 1],
                       c=relative_l2, s=14, cmap="magma")
    axes[1, 1].set(xlabel=r"$X_e/Z_e$", ylabel=r"$Y_e/Z_e$",
                   title=(f"largest group: n={members.size}\n"
                          f"max L1={l1.max():.3g}, max rel.L2={relative_l2.max():.3g}"))
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_matrix_comparison(reference, approximation, output: Path):
    reference = reference.tocsr()
    approximation = approximation.tocsr()
    difference = (approximation - reference).tocsr()
    ref_sensitivity = np.asarray(reference.sum(axis=0)).ravel()
    app_sensitivity = np.asarray(approximation.sum(axis=0)).ravel()
    column_relative_l2 = np.zeros(reference.shape[1])
    for column in range(reference.shape[1]):
        ref = reference[:, column].toarray().ravel()
        app = approximation[:, column].toarray().ravel()
        column_relative_l2[column] = np.linalg.norm(app - ref) / max(np.linalg.norm(ref), 1e-30)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].spy(reference, markersize=0.35, color="tab:blue")
    axes[0].set_title(f"existing P: nnz={reference.nnz}")
    axes[1].spy(approximation, markersize=0.35, color="tab:orange")
    axes[1].set_title(f"Q A: nnz={approximation.nnz}")
    axes[2].scatter(ref_sensitivity, column_relative_l2, s=8, alpha=0.65)
    axes[2].set(xlabel="P column sum", ylabel="column relative L2",
                title="final voxel-column error")
    axes[2].set_yscale("log")
    axes[2].grid(which="both", alpha=0.25)
    max_sum_error = np.max(np.abs(app_sensitivity - ref_sensitivity), initial=0.0)
    axes[2].text(0.03, 0.97, f"max column-sum error={max_sum_error:.3e}",
                 transform=axes[2].transAxes, va="top")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_case(output_dir: Path, axial_distance=100.0, resolution=2,
             voxel_shape=(10, 10, 6), pixel_shape=(24, 24),
             with_aperture=True, max_chunk_size=256,
             tolerance=0.1, metric="relative_l2", algorithm="recursive"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    world = build_toy_world(axial_distance, voxel_shape, pixel_shape, with_aperture)
    problem = build_problem(world, resolution, max_chunk_size=max_chunk_size)
    compression = compress_problem(problem, tolerance, metric, algorithm)
    production_metrics = _matrix_metrics(problem.P_existing, problem.P_manual)
    compression_metrics = _matrix_metrics(problem.P_existing, compression.P_approx)
    profile_rows, _, _ = evaluate_profiles(problem, compression)

    paths = {
        "geometry": output_dir / "01_geometry.png",
        "optical_distribution": output_dir / "02_optical_distribution.png",
        "visibility": output_dir / "03_visibility.png",
        "compression_detail": output_dir / "04_compression_detail.png",
        "matrix_comparison": output_dir / "05_matrix_comparison.png",
        "matrix_metrics": output_dir / "case_matrix_metrics.csv",
        "profile_metrics": output_dir / "case_profile_metrics.csv",
    }
    plot_geometry(problem, paths["geometry"])
    plot_optical_distribution(problem, paths["optical_distribution"])
    plot_visibility(problem, paths["visibility"])
    plot_compression_detail(problem, compression, paths["compression_detail"])
    plot_matrix_comparison(problem.P_existing, compression.P_approx,
                           paths["matrix_comparison"])
    with paths["matrix_metrics"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("comparison", *production_metrics.keys()))
        writer.writeheader()
        writer.writerow({"comparison": "manual_I_S_vs_existing_P", **production_metrics})
        writer.writerow({"comparison": "Q_A_vs_existing_P", **compression_metrics})
    with paths["profile_metrics"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)
    return {
        "problem": problem,
        "compression": compression,
        "production_metrics": production_metrics,
        "compression_metrics": compression_metrics,
        "profile_rows": profile_rows,
        "paths": paths,
    }


def run_sweep(output_dir: Path, axial_distances=(100.0, 300.0, 1000.0),
              resolutions=(1, 2, 4), tolerances=(0.03, 0.1, 0.2),
              metrics=("relative_l2", "l1"), algorithms=("recursive", "leader"),
              voxel_shape=(8, 8, 4), pixel_shape=(20, 20), max_chunk_size=256):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.perf_counter()
    for axial_distance in axial_distances:
        for resolution in resolutions:
            problem = build_problem(
                build_toy_world(axial_distance, voxel_shape, pixel_shape, False),
                resolution, max_chunk_size=max_chunk_size,
            )
            production_metrics = _matrix_metrics(problem.P_existing, problem.P_manual)
            for metric in metrics:
                for algorithm in algorithms:
                    for tolerance in tolerances:
                        compression = compress_problem(problem, tolerance, metric, algorithm)
                        matrix_metrics = _matrix_metrics(problem.P_existing,
                                                         compression.P_approx)
                        profile_rows, _, _ = evaluate_profiles(problem, compression)
                        for profile in profile_rows:
                            rows.append({
                                "axial_distance": axial_distance,
                                "z_over_f": axial_distance / FOCAL_LENGTH,
                                "resolution": resolution,
                                "subvoxel_count": problem.I.shape[1],
                                "metric": metric,
                                "algorithm": algorithm,
                                "tolerance": tolerance,
                                "group_count": compression.Q.shape[1],
                                "optical_chunk_count": len(problem.optical_chunks),
                                "psf_count_compression": (
                                    problem.I.shape[1] / max(compression.Q.shape[1], 1)
                                ),
                                "max_group_l1": float(
                                    compression.group_max_l1.max(initial=0.0)
                                ),
                                "max_group_relative_l2": float(
                                    compression.group_max_relative_l2.max(initial=0.0)
                                ),
                                "I_nnz": problem.I.nnz,
                                "Q_nnz": compression.Q.nnz,
                                "A_nnz": compression.A.nnz,
                                "factor_nnz_compression": (
                                    problem.I.nnz / max(compression.Q.nnz + compression.A.nnz, 1)
                                ),
                                "I_bytes": _sparse_nbytes(problem.I),
                                "factor_bytes": (_sparse_nbytes(compression.Q)
                                                 + _sparse_nbytes(compression.A)),
                                "compression_seconds": compression.elapsed_seconds,
                                "production_manual_relative_frobenius": (
                                    production_metrics["relative_frobenius"]
                                ),
                                **matrix_metrics,
                                **profile,
                            })

    csv_path = output_dir / "subvoxel_psf_compression_sweep.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    selected_profile = "gaussian"
    selected_metric = "relative_l2"
    selected_algorithm = "recursive"
    for distance in axial_distances:
        selected = [row for row in rows
                    if row["axial_distance"] == distance
                    and row["metric"] == selected_metric
                    and row["algorithm"] == selected_algorithm
                    and row["tolerance"] == tolerances[1]
                    and row["profile"] == selected_profile]
        selected.sort(key=lambda row: row["resolution"])
        label = f"Z/f={distance / FOCAL_LENGTH:g}"
        axes[0, 0].plot([row["subvoxel_count"] for row in selected],
                        [row["group_count"] for row in selected], marker="o", label=label)
        axes[0, 1].plot([row["resolution"] for row in selected],
                        [row["factor_nnz_compression"] for row in selected], marker="o", label=label)
        axes[1, 0].plot([row["resolution"] for row in selected],
                        [max(row["relative_l2"], 1e-16) for row in selected], marker="o", label=label)
    compare = [row for row in rows
               if row["axial_distance"] == axial_distances[0]
               and row["resolution"] == resolutions[-1]
               and row["profile"] == selected_profile]
    for metric in metrics:
        for algorithm in algorithms:
            selected = sorted((row for row in compare
                               if row["metric"] == metric and row["algorithm"] == algorithm),
                              key=lambda row: row["tolerance"])
            axes[1, 1].plot([row["psf_count_compression"] for row in selected],
                            [max(row["relative_l2"], 1e-16) for row in selected],
                            marker="o", label=f"{algorithm}, {metric}")
    axes[0, 0].set(xlabel="visible subvoxel PSFs Ns", ylabel="representative PSFs Ng",
                   title="does Ng saturate as res³ grows?")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set(xlabel="source res", ylabel="nnz(I) / [nnz(Q)+nnz(A)]",
                   title="factor storage compression")
    axes[1, 0].set(xlabel="source res", ylabel="Gaussian image relative L2",
                   title="image error versus source res")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set(xlabel="Ns / Ng", ylabel="Gaussian image relative L2",
                   title=f"algorithm/metric comparison, Z/f={axial_distances[0]/FOCAL_LENGTH:g}")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    for axis in axes.ravel():
        axis.grid(which="both", alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    figure_path = output_dir / "06_compression_sweep.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return {
        "rows": rows, "csv_path": csv_path, "figure_path": figure_path,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _median_elapsed(operation, repeats: int) -> float:
    """Return a warm median for a small matrix-vector operation."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    operation()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed))


def run_production_sweep(
        output_dir: Path,
        axial_distances=(100.0, 300.0, 1000.0),
        resolutions=(1, 2),
        bin_widths=(1.0,),
        tolerances=(0.1,),
        metrics=("relative_l2",),
        algorithms=("recursive",),
        max_group_fractions=(0.6, 0.8, 1.0, None),
        voxel_shape=(6, 6, 4),
        pixel_shape=(16, 16),
        detector_resolution=1,
        timing_repeats=5,
        max_working_memory=256 * 2 ** 20,
):
    """Benchmark the production World sparse and hybrid projection paths.

    Exact ``P`` is built once per geometry/resolution/bin combination for
    validation.  Production hybrid construction never uses that matrix for its
    direct/factorized decision; it is retained here only to measure errors and
    the actual CSR byte ratio.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    benchmark_start = time.perf_counter()

    for axial_distance in axial_distances:
        for resolution in resolutions:
            for bin_width in bin_widths:
                world = build_toy_world(
                    axial_distance=axial_distance,
                    voxel_shape=voxel_shape,
                    pixel_shape=pixel_shape,
                    with_aperture=False,
                    detector_resolution=detector_resolution,
                )
                # Keep visibility initialization and first-call library setup
                # out of the direct-vs-hybrid construction comparison.
                world.find_visible_voxels(force=True, verbose=0)
                world.set_projection_matrix(
                    res=resolution, partial_res=resolution,
                    verbose=0, parallel=1, force=True,
                    max_working_memory=max_working_memory,
                    chunk_strategy="optical",
                    optical_bin_width_pixels=bin_width,
                    projection_representation="sparse",
                )
                tracemalloc.start()
                direct_start = time.perf_counter()
                world.set_projection_matrix(
                    res=resolution, partial_res=resolution,
                    verbose=0, parallel=1, force=True,
                    max_working_memory=max_working_memory,
                    chunk_strategy="optical",
                    optical_bin_width_pixels=bin_width,
                    projection_representation="sparse",
                )
                direct_build_seconds = time.perf_counter() - direct_start
                _, direct_peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                reference = world.P_matrix[0].tocsr()
                direct_bytes = _sparse_nbytes(reference)
                profiles = emission_profiles(world.voxel.gravity_center)
                timing_emission = profiles["gaussian"]
                timing_detector = np.linspace(-1.0, 1.0, reference.shape[0])
                direct_project_seconds = _median_elapsed(
                    lambda: reference @ timing_emission, timing_repeats,
                )
                direct_backproject_seconds = _median_elapsed(
                    lambda: reference.T @ timing_detector, timing_repeats,
                )

                for metric in metrics:
                    for algorithm in algorithms:
                        for tolerance in tolerances:
                            for max_group_fraction in max_group_fractions:
                                tracemalloc.start()
                                hybrid_start = time.perf_counter()
                                world.set_projection_matrix(
                                    res=resolution, partial_res=resolution,
                                    verbose=0, parallel=1, force=True,
                                    max_working_memory=max_working_memory,
                                    chunk_strategy="optical",
                                    optical_bin_width_pixels=bin_width,
                                    projection_representation="hybrid",
                                    psf_tolerance=tolerance,
                                    psf_metric=metric,
                                    psf_grouping=algorithm,
                                    max_group_fraction=max_group_fraction,
                                )
                                hybrid_build_seconds = time.perf_counter() - hybrid_start
                                _, hybrid_peak_bytes = tracemalloc.get_traced_memory()
                                tracemalloc.stop()
                                operator = world.P_matrix[0]
                                stats = operator.compression_stats
                                hybrid_bytes = operator.storage_nbytes
                                hybrid_project_seconds = _median_elapsed(
                                    lambda: operator.project(timing_emission),
                                    timing_repeats,
                                )
                                hybrid_backproject_seconds = _median_elapsed(
                                    lambda: operator.backproject(timing_detector),
                                    timing_repeats,
                                )

                                # Explicit materialization belongs to this
                                # benchmark only. It validates indexing and
                                # approximation error after timing production.
                                approximation = operator.to_sparse()
                                difference = approximation - reference
                                reference_norm = sparse.linalg.norm(reference)
                                reference_sum = np.asarray(reference.sum(axis=0)).ravel()
                                approximate_sum = np.asarray(
                                    approximation.sum(axis=0)
                                ).ravel()
                                sum_scale = max(
                                    float(np.max(np.abs(reference_sum), initial=0.0)),
                                    1e-30,
                                )
                                matrix_relative_frobenius = float(
                                    sparse.linalg.norm(difference)
                                    / max(reference_norm, 1e-30)
                                )
                                max_column_sum_relative = float(
                                    np.max(
                                        np.abs(approximate_sum - reference_sum),
                                        initial=0.0,
                                    ) / sum_scale
                                )

                                common = {
                                    "axial_distance": axial_distance,
                                    "z_over_f": axial_distance / FOCAL_LENGTH,
                                    "resolution": resolution,
                                    "detector_resolution": detector_resolution,
                                    "bin_width_pixels": bin_width,
                                    "metric": metric,
                                    "algorithm": algorithm,
                                    "tolerance": tolerance,
                                    "max_group_fraction": max_group_fraction,
                                    "scope_count": len(stats),
                                    "factorized_scope_count": sum(
                                        item.used_factorization for item in stats
                                    ),
                                    "sample_count": sum(item.n_samples for item in stats),
                                    "active_sample_count": sum(
                                        item.n_active_samples for item in stats
                                    ),
                                    "stored_group_count": sum(
                                        item.n_groups for item in stats
                                        if item.used_factorization
                                    ),
                                    "median_active_pixel_rows": float(np.median([
                                        item.n_active_pixel_rows for item in stats
                                    ])) if stats else 0.0,
                                    "max_active_pixel_rows": max(
                                        (item.n_active_pixel_rows for item in stats),
                                        default=0,
                                    ),
                                    "grouping_seconds": sum(
                                        item.grouping_seconds for item in stats
                                    ),
                                    "assembly_seconds": sum(
                                        item.assembly_seconds for item in stats
                                    ),
                                    "direct_build_seconds": direct_build_seconds,
                                    "hybrid_build_seconds": hybrid_build_seconds,
                                    "build_speed_ratio": (
                                        direct_build_seconds
                                        / max(hybrid_build_seconds, 1e-30)
                                    ),
                                    "direct_peak_python_bytes": direct_peak_bytes,
                                    "hybrid_peak_python_bytes": hybrid_peak_bytes,
                                    "direct_bytes": direct_bytes,
                                    "hybrid_bytes": hybrid_bytes,
                                    "storage_compression": (
                                        direct_bytes / max(hybrid_bytes, 1)
                                    ),
                                    "direct_project_seconds": direct_project_seconds,
                                    "hybrid_project_seconds": hybrid_project_seconds,
                                    "project_speed_ratio": (
                                        direct_project_seconds
                                        / max(hybrid_project_seconds, 1e-30)
                                    ),
                                    "direct_backproject_seconds": direct_backproject_seconds,
                                    "hybrid_backproject_seconds": hybrid_backproject_seconds,
                                    "backproject_speed_ratio": (
                                        direct_backproject_seconds
                                        / max(hybrid_backproject_seconds, 1e-30)
                                    ),
                                    "matrix_relative_frobenius": matrix_relative_frobenius,
                                    "max_column_sum_relative": max_column_sum_relative,
                                }
                                for profile_name, emission in profiles.items():
                                    exact = np.asarray(reference @ emission).ravel()
                                    approximate = np.asarray(
                                        operator.project(emission)
                                    ).ravel()
                                    error = approximate - exact
                                    rows.append({
                                        **common,
                                        "profile": profile_name,
                                        "relative_l1": float(
                                            np.linalg.norm(error, ord=1)
                                            / max(np.linalg.norm(exact, ord=1), 1e-30)
                                        ),
                                        "relative_l2": float(
                                            np.linalg.norm(error)
                                            / max(np.linalg.norm(exact), 1e-30)
                                        ),
                                        "relative_flux": float(
                                            abs(approximate.sum() - exact.sum())
                                            / max(abs(exact.sum()), 1e-30)
                                        ),
                                    })

    csv_path = output_dir / "production_hybrid_projection_sweep.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gaussian_rows = [row for row in rows if row["profile"] == "gaussian"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for resolution in resolutions:
        for fraction in max_group_fractions:
            selected = [
                row for row in gaussian_rows
                if row["resolution"] == resolution
                and row["max_group_fraction"] == fraction
                and row["bin_width_pixels"] == bin_widths[0]
                and row["metric"] == metrics[0]
                and row["algorithm"] == algorithms[0]
                and row["tolerance"] == tolerances[0]
            ]
            selected.sort(key=lambda row: row["z_over_f"])
            if not selected:
                continue
            fraction_label = "always" if fraction is None else f"<{fraction:g}"
            label = (
                f"source res={resolution}, detector res={detector_resolution}, "
                f"Ng/Ns {fraction_label}"
            )
            z = [row["z_over_f"] for row in selected]
            axes[0, 0].plot(z, [row["storage_compression"] for row in selected],
                            marker="o", label=label)
            axes[0, 1].plot(z, [row["build_speed_ratio"] for row in selected],
                            marker="o", label=label)
            axes[1, 0].plot(z, [row["project_speed_ratio"] for row in selected],
                            marker="o", label=label)
            axes[1, 1].plot(
                [row["storage_compression"] for row in selected],
                [max(row["relative_l2"], 1e-16) for row in selected],
                marker="o", label=label,
            )
    axes[0, 0].axhline(1.0, color="0.4", lw=1)
    axes[0, 0].set(xlabel="Z/f", ylabel="CSR P bytes / hybrid bytes",
                   title="stored-byte compression")
    axes[0, 1].axhline(1.0, color="0.4", lw=1)
    axes[0, 1].set(xlabel="Z/f", ylabel="sparse build / hybrid build",
                   title="construction speed ratio")
    axes[1, 0].axhline(1.0, color="0.4", lw=1)
    axes[1, 0].set(xlabel="Z/f", ylabel="sparse project / hybrid project",
                   title="forward speed ratio")
    axes[1, 1].set(xlabel="stored-byte compression",
                   ylabel="Gaussian image relative L2",
                   title="accuracy versus storage")
    axes[1, 1].set_yscale("log")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    figure_path = output_dir / "07_production_hybrid_sweep.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return {
        "rows": rows,
        "csv_path": csv_path,
        "figure_path": figure_path,
        "elapsed_seconds": time.perf_counter() - benchmark_start,
    }


def _parse_int_triplet(value):
    result = tuple(int(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return result


def _parse_int_pair(value):
    result = tuple(int(item) for item in value.split(","))
    if len(result) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated integers")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("examples/output/subvoxel_psf_compression"))
    parser.add_argument("--axial-distance", type=float, default=100.0)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--voxel-shape", type=_parse_int_triplet, default=(10, 10, 6))
    parser.add_argument("--pixel-shape", type=_parse_int_pair, default=(24, 24))
    parser.add_argument("--max-chunk-size", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--metric", choices=("l1", "relative_l2"), default="relative_l2")
    parser.add_argument("--algorithm", choices=("recursive", "leader"), default="recursive")
    parser.add_argument("--no-aperture", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--production-sweep", action="store_true")
    args = parser.parse_args()

    case = run_case(
        args.output_dir, axial_distance=args.axial_distance,
        resolution=args.resolution, voxel_shape=args.voxel_shape,
        pixel_shape=args.pixel_shape, with_aperture=not args.no_aperture,
        max_chunk_size=args.max_chunk_size, tolerance=args.tolerance,
        metric=args.metric, algorithm=args.algorithm,
    )
    print("production/manual:", case["production_metrics"])
    print("compression:", case["compression_metrics"])
    for row in case["profile_rows"]:
        print(row)
    for name, path in case["paths"].items():
        print(f"{name}: {path}")
    if args.sweep:
        sweep = run_sweep(args.output_dir)
        print(f"sweep CSV: {sweep['csv_path']}")
        print(f"sweep figure: {sweep['figure_path']}")
        print(f"sweep elapsed_seconds: {sweep['elapsed_seconds']}")
    if args.production_sweep:
        production_sweep = run_production_sweep(args.output_dir)
        print(f"production sweep CSV: {production_sweep['csv_path']}")
        print(f"production sweep figure: {production_sweep['figure_path']}")
        print(f"production sweep elapsed_seconds: {production_sweep['elapsed_seconds']}")
