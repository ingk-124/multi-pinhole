# Benchmarks

This directory contains reproducible performance and numerical-accuracy
experiments. User-facing examples remain in `examples/`; generated figures,
CSV files, and JSON summaries belong in `benchmarks/output/` and are not
versioned.

Run scripts as modules from the repository root, for example:

```bash
python -m benchmarks.benchmark_projection --help
python -m benchmarks.evaluate_adaptive_cap --help
python -m benchmarks.evaluate_partial_resolution --help
```

## Current validation scripts

| Script | Purpose |
|---|---|
| `benchmark_projection.py` | Projection construction time and sparse-matrix size for a small scene or reduced MST geometry |
| `benchmark_partial_visibility.py` | Reference/optimized comparison of partial sub-voxel visibility after a cache-building preflight |
| `profile_wall_visibility.py` | Stage timings, peak-memory indicators, candidate counts, and cache-hit timing for visibility preflight |
| `evaluate_adaptive_cap.py` | Uncapped ideal source resolution versus capped adaptive resolution |
| `evaluate_mst_adaptive_resolution.py` | Fixed, adaptive, and ideal resolution on the MST model |
| `evaluate_partial_resolution.py` | Convergence of partial-cell integration across plane and spherical boundaries |
| `evaluate_small_voxel_depth_sweep.py` | Resolution selected versus voxel size and source depth |
| `evaluate_subvoxel_resolution.py` | Source quadrature convergence versus voxel, pixel, and PSF scale |
| `evaluate_adaptive_projection.py` | Earlier wall-free adaptive-resolution comparison |
| `evaluate_adaptive_threshold_sweep.py` | Point-source threshold sweep |
| `evaluate_adaptive_source_resolution.py` | Earlier projected-span diagnostic retained for comparison |

## Exploratory compression scripts

`evaluate_projection_column_compression.py` and
`evaluate_subvoxel_psf_compression.py` preserve the QA/PSF factorization
experiments. They do not implement the production projection path. See
`docs/ja/projection-compression-future.md` before extending them.

## Accepted reference results

- In the wall-free d=10 toy model, adaptive cap 5 differed from uncapped ideal
  by about 0.2% or less for smooth profiles, about 0.4% for the near square
  profile, and about 0.9% for the nearest single-voxel impulse.
- In the restricted MST d=25 region, adaptive cap 5 with a common fixed
  partial resolution differed from ideal-full by 0.044--0.112% in image L2
  and 0.0022--0.0055% in total flux. It reduced the estimated source samples
  by 44% relative to ideal and 24% relative to fixed resolution 5; measured
  construction time changed from 189.5 s to 155.5 s versus ideal.
- Partial-cell error is controlled mainly by the phase between a discontinuous
  boundary and the regular sample centers. Small fixed `partial_res` therefore
  has no geometry-independent error guarantee.

These numbers are engineering reference points, not API accuracy guarantees.
Re-run the corresponding script when projection geometry or quadrature changes.

## Visibility profiling

Run each case in a fresh process so the process peak-RSS value belongs to that
case.  Start with the wall-free toy and two-triangle plane before increasing the
reduced MST grid.  The output includes a geometry fingerprint and generating
commit so before/after results can be paired safely.

```bash
python -m benchmarks.profile_wall_visibility --scene toy --voxel-shape 24,16,12
python -m benchmarks.profile_wall_visibility --scene plane --voxel-shape 24,16,12
python -m benchmarks.profile_wall_visibility --scene mst --voxel-shape 24,16,12
python -m benchmarks.profile_wall_visibility --scene mst --voxel-shape 24,16,12 --implementation reference
```

`tracemalloc_peak_bytes` is a phase-local allocation indicator.  The
`process_peak_rss_*` fields are process-wide high-water marks and should not be
subtracted; use them only when the benchmark starts in a fresh process.

On the restricted MST `shape=(24,16,12)` case, the bounded implementation and
the retained reference path produced the same visibility SHA-256 and voxel
state counts.  In one local run, cold preflight changed from 7.14 s to 0.130 s,
phase-local traced peak allocation from 284 MB to 9.9 MB, and process peak RSS
from about 585 MB to 204 MB.  These are small-grid development measurements,
not the final d=10 acceptance result.

Partial sub-voxel visibility uses the same optimized wall-intersection helper.
Run the following commands in separate processes and require matching
`projection_sha256` values:

```bash
python -m benchmarks.benchmark_partial_visibility --implementation reference
python -m benchmarks.benchmark_partial_visibility --implementation optimized
```
