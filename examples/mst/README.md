# MST 2025 SXR example

This directory contains the MST-specific soft X-ray camera example and its
simplified vessel-wall geometry.

## Contents

- `MST_2025_SXR_imaging.py`: builds two independent 61×61 single-pinhole cameras and runs the voxel projection workflow.
- `MST_wall-mesh.stl`: simplified MST wall mesh used for visibility checks and geometry plots.
- `outputs/`: generated projection caches and figures; intentionally excluded from Git.

The two cameras share the CAD reference pose and are translated by ±4.15 mm
along the camera X axis. Each camera has a 7.5×7.5 mm screen, a centered
single-hole aperture, and one centered pinhole eye.

Run from the repository root:

```bash
python examples/mst/MST_2025_SXR_imaging.py
```

The full projection is computationally expensive. Set `force_rebuild = False`
after generating the cached world in `outputs/` if you want to reuse it locally.

