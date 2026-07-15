# Changelog

## 0.7.0

- Added fixed, capped adaptive, and uncapped ideal source-resolution policies.
- Added projection preflight estimates and optical chunk ordering.
- Integrated detector cell-area overlap and local finite-Eye etendue evaluation.
- Preserved trilinear source interpolation with per-sample voxel-volume weights.
- Moved reproducible performance and accuracy experiments to `benchmarks/`.
- Updated the projection roadmap and archived the deferred QA/PSF compression design.

This is a breaking pre-1.0 release: projection construction now requires an
explicit source resolution and uses `res_mode` to select the resolution policy.
