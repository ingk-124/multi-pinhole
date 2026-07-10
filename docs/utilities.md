# Utilities Overview

The `multi_pinhole.utils` package (imported as `from multi_pinhole.utils import ...` or `from multi_pinhole.utils import stl_utils`, etc. -- it was moved under `multi_pinhole` to avoid colliding with unrelated top-level `utils` packages) collects helper functions that support the optical simulation without cluttering the core camera logic. These utilities cover input validation, progress-aware logging, STL mesh processing, and filter-transmission lookup.

## Collection Helpers
`multi_pinhole.utils.type_check_and_list` normalizes optional constructor arguments into lists while validating element types. It also accepts a default fallback when `None` is provided, allowing world and camera constructors to treat single objects and lists uniformly.【F:multi_pinhole/utils/__init__.py†L24-L54】

## Console Wrappers
`multi_pinhole.utils.my_stdio` wraps common iteration primitives with optional progress displays. `my_print` gates log messages on a `show` flag, `my_range` and `my_tqdm` defer to `tqdm` versions when verbosity is enabled, and `my_zip` pairs iterables with progress bars via `tzip`. These adapters let long-running geometry routines expose progress feedback without hard-coding dependencies on `tqdm` in calling code.【F:multi_pinhole/utils/my_stdio.py†L1-L97】

## STL Geometry Toolkit
`multi_pinhole.utils.stl_utils` provides extensive support for constructing and analyzing STL meshes used by apertures and world walls. Highlights include:

- `shape_check` and `generate_aperture_stl`, which validate analytic aperture dimensions and synthesize 2D STL meshes for circles, ellipses, and rectangles.【F:multi_pinhole/utils/stl_utils.py†L49-L173】
- `rotate_model` and `copy_model`, convenience functions for duplicating meshes and applying translations or Euler-angle rotations without mutating the originals.【F:multi_pinhole/utils/stl_utils.py†L174-L234】
- Visibility primitives such as `check_intersection` and `check_visible`, combining fast cone tests with Möller–Trumbore ray-triangle intersection to determine whether rays between an eye and grid points are occluded by geometry.【F:multi_pinhole/utils/stl_utils.py†L235-L671】

These routines are consumed by `World` and `Camera` to test aperture and wall occlusion efficiently during projection.

## Filter Transmission Data
`multi_pinhole.utils.filter` automates retrieval and interpolation of X-ray filter transparency curves. `get_data_from_CXRO` drives the CXRO website via Selenium to download tabulated transmission data, while `get_data` caches the results locally for repeat runs. The `exp_fit` helper fits an exponential attenuation law, and `characteristic` returns a callable that predicts transparency across energies for arbitrary film thicknesses using interpolated coefficients.【F:multi_pinhole/utils/filter.py†L37-L225】

Note: there is currently no `Filter` class in `multi_pinhole.core` that wraps these functions (an earlier version of this document, and of `docs/core.md`, described one, but it was removed from the code without being removed from the docs). Callers that need filter transmission today should call `multi_pinhole.utils.filter.get_data`/`characteristic` directly.
