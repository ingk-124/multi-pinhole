# Core Module Reference

This document summarizes the key classes defined in `multi_pinhole.core` and how they collaborate to simulate a multi-eye imaging system.

## Rays
`Rays` is an immutable dataclass, defined in `multi_pinhole.rays` and re-exported through `multi_pinhole.core`/`multi_pinhole`, that carries the geometric result of tracing scene points through an eye. It records:

* `Z`: the optical-axis distance from the eye to each point, preserving the sign so callers can distinguish front- and back-facing samples.【F:multi_pinhole/rays.py†L8-L28】
* `XY`: projected impact locations on the screen in camera Cartesian coordinates, with samples behind the eye left as `NaN` to simplify masking.【F:multi_pinhole/rays.py†L8-L28】【F:multi_pinhole/core.py†L316-L333】
* `zoom_rate`: the magnification factor `(1 + f/Z)` needed to dilate the eye footprint before rasterization.【F:multi_pinhole/rays.py†L8-L28】【F:multi_pinhole/core.py†L316-L333】
* `front_and_visible`: Boolean flags that encode both geometric visibility and aperture occlusion checks.【F:multi_pinhole/rays.py†L8-L28】【F:multi_pinhole/core.py†L316-L333】

Light-weight helpers (`n`, `n_visible`) expose counts of the total rays created and the subset that will contribute to the final image, making it easy for screening code to size sparse buffers or short-circuit empty jobs.【F:multi_pinhole/rays.py†L30-L38】 As a pure data object, `Rays` sits between `Eye.calc_rays` and the `Screen` rasterizers.

## Filter transmission (no `Filter` class)
There is currently **no `Filter` class** in `multi_pinhole.core` (an earlier version of this document described one, but it does not exist in the current codebase; the only remaining reference is a leftover, unreachable `Filter(...)` call at the end of `core.py`'s `if __name__ == "__main__"` demo block). Filter/transmission calculations are instead exposed as plain functions in `multi_pinhole.utils.filter`: `get_data`/`get_data_from_CXRO` retrieve tabulated CXRO transmission curves (with local caching), and `characteristic(material, d)` returns a callable that evaluates transparency at arbitrary thickness and photon energy via a fitted exponential attenuation model. See `docs/utilities.md` for details.

## Eye
An `Eye` models a single pinhole or concave lens, encapsulating its placement, field-of-view, and spectral band. During initialization it:

* Normalizes the `eye_type` (pinhole vs. concave lens) while enforcing consistent focal-length sign conventions and converting the 2D mount location into a full camera-frame position vector.【F:multi_pinhole/core.py†L178-L215】
* Validates the aperture shape and dimensions, promoting scalar inputs to `(height, width)` pairs where appropriate to simplify subsequent math.【F:multi_pinhole/core.py†L217-L266】
* Records the wavelength band and resets the owning `Camera` link so a camera can claim the eye later.【F:multi_pinhole/core.py†L267-L319】

Operationally, the eye offers:

* `camera2eye`, a vectorized translation that moves world samples (already expressed in the camera frame) into the local optical frame, ready for projection math.【F:multi_pinhole/core.py†L314-L328】
* `calc_rays`, which filters out geometry behind the focal plane, applies any visibility mask coming from aperture occlusion tests, projects surviving points onto the screen, and computes the magnification required to dilate the eye footprint during rasterization.【F:multi_pinhole/core.py†L330-L372】
* A set of read-only properties exposing the resolved geometry (type, size, shape, focal length, position, and principal point) and metadata such as the wavelength range or owning `Camera`, keeping the rest of the pipeline loosely coupled.【F:multi_pinhole/core.py†L374-L409】

## Aperture
`Aperture` describes the physical opening that limits light reaching an eye. It accepts analytic shapes (circle, ellipse, rectangle) or an STL mesh, generates STL geometry when needed, and stores placement and orientation relative to the camera screen.【F:multi_pinhole/core.py†L419-L528】 The STL representation is also used for visibility checks when rasterizing rays, ensuring only unobstructed paths contribute to the final image.【F:multi_pinhole/core.py†L1558-L1583】

## Screen
The `Screen` represents the detector plane and its sampling scheme. Initialization validates the physical shape, pixel grid, and subpixel refinement factor before computing derived quantities such as pixel area, positions, and masks that exclude out-of-aperture regions.【F:multi_pinhole/core.py†L530-L645】  Changing the subpixel resolution rebuilds subpixel coordinates, sparse transform matrices, and mask arrays so downstream rasterizers stay consistent with the requested sampling density.【F:multi_pinhole/core.py†L646-L753】

Supporting utilities include:

* `positions` and `image_mask`, which generate the center coordinates of pixels or subpixels and mark locations outside circular or elliptical screens so they can be zeroed in final images.【F:multi_pinhole/core.py†L754-L799】
* `cosine` and `etendue_per_subpixel`, which compute angular falloff corrections and local etendue weights based on the active eye’s focal length and pupil size.【F:multi_pinhole/core.py†L802-L845】

For rasterization, the screen provides multiple strategies. `ray2image` performs a straightforward subpixel hit test for each ray footprint, `ray2image2` introduces batching and optional parallelism for large ray counts, and `ray2image_grid` accelerates the process by clipping to subpixel tiles before applying exact ellipse/rectangle membership tests. All variants construct sparse CSC matrices whose rows correspond to subpixels and whose columns correspond to rays, scaled by the appropriate etendue weights.【F:multi_pinhole/core.py†L846-L1165】

Finally, helper routines cover coordinate transforms (`xy2uv`, `uv2subpixel_index`), accumulation (`subpixel_to_pixel`), and visualization (`show_image`), ensuring the simulated detector data can be reshaped or displayed without ad-hoc code in higher layers.【F:multi_pinhole/core.py†L1167-L1479】

## Camera
`Camera` ties together one or more `Eye` instances, associated `Aperture` objects, and a `Screen`, while positioning and orienting the assembly in world space. Construction enforces consistent eye types, checks that screen sampling is fine enough for the smallest eye, and stores rotation/translation state.【F:multi_pinhole/core.py†L1330-L1421】 Core responsibilities are:
- Managing world transforms, movement, and rotation so that world coordinates can be mapped into the camera frame via `world2camera`, while exposing convenient accessors for the camera axes, pose, and child objects.【F:multi_pinhole/core.py†L1384-L1527】
- Generating image vectors with `calc_image_vec`, which evaluates aperture visibility through STL intersection tests, requests ray bundles from the target eye, and delegates to the screen’s rasterizer to assemble sparse detector responses.【F:multi_pinhole/core.py†L1558-L1587】
- Providing visualization helpers (`draw_optical_system`, `draw_camera_orientation_plotly`, `draw_camera_orientation`) to inspect optical layouts in Matplotlib or Plotly, facilitating alignment and debugging.【F:multi_pinhole/core.py†L1589-L1786】

Together these classes provide the foundation for higher-level modules (such as `world` and `voxel`) to simulate complex multi-aperture imaging experiments.
