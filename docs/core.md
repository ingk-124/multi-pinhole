# Core Module Reference

This document explains the classes defined in `multi_pinhole.core` and, more
importantly, *how* they compute what they compute: the coordinate-frame
conventions, the pinhole projection formula, the rasterization algorithm that
turns a ray into subpixel weights, and the aperture-occlusion check. For the
full API surface, read the docstrings in `multi_pinhole/core.py` directly —
this document focuses on the underlying process.

## The four coordinate systems

Every calculation in `core.py` moves points between four coordinate systems,
laid out in a comment block at the top of the module.
Understanding the chain is the key to reading the rest of this document:

```
world (x, y, z)  →  camera (X, Y, Z)  →  eye/pinhole (X', Y', Z')  →  screen/image (u, v)
```

1. **World coordinates** `(x, y, z)` — the single global Cartesian frame.
   Everything in a `World` (voxels, cameras, walls) is expressed here. It can
   also be *reinterpreted* in cylindrical or toroidal terms for evaluating
   plasma profiles (see `multi_pinhole.coordinates` and `docs/overview.md`),
   but the underlying representation stays Cartesian `(x, y, z)`.

2. **Camera coordinates** `(X, Y, Z)` — a right-handed Cartesian frame
   attached to one `Camera`. The origin is the center of that camera's
   screen; `Z` is the main optical axis (pointing out of the screen); `X`
   points right and `Y` points down. A world point is converted to camera
   coordinates by translating to the camera position and then applying the
   camera's rotation matrix — this is exactly `Camera.world2camera`:
   `X_cam = R · (x_world − camera_position)`, implemented as
   `(self.rotation_matrix @ (points - self.camera_position[None, :]).T).T`.

3. **Eye (pinhole) coordinates** `(X', Y', Z')` — one per `Eye` mounted on
   the camera. A camera can hold several eyes, each offset within the camera
   plane by `(X_h, Y_h)` and separated from the screen by focal length `f`.
   The eye frame is just the camera frame translated so its origin sits at
   the eye: `X' = X − X_h`, `Y' = Y − Y_h`, `Z' = Z − f` (for a pinhole; a
   concave lens uses `Z' = Z`, see below). This translation is
   `Eye.camera2eye`.

4. **Image coordinates** `(u, v)` — a 2D frame on the screen surface, with
   the origin at the screen's *upper-left* corner (not its center). `u`
   increases downward (matching camera `Y`) and `v` increases to the right
   (matching camera `X`); note the axis order is swapped relative to
   `(X, Y)`. The screen center, which is the camera-frame origin, sits at
   `(u_c, v_c) = screen_size / 2`. `Screen.xy2uv` performs exactly this
   flip-and-shift: `uv = xy[..., ::-1] + screen_size / 2`.

Two eye types change how the pinhole frame sits relative to the screen:

* **Pinhole** (`eye_type="pinhole"`, `focal_length > 0`): the "eye" *is* the
  pinhole itself, offset from the screen by the focal length along `Z`. Its
  position **and** principal point both equal `(X_h, Y_h, f)`.
* **Concave lens** (`eye_type="concave_lens"`, `focal_length < 0`): the
  screen coincides with the lens surface, so the eye position is
  `(X_h, Y_h, 0)`, while the principal point (used for the projection
  formula below) is offset to `(X_h, Y_h, f)`.

This is enforced in `Eye.__init__`, which also normalizes `eye_size` into a
`(height, width)` pair and validates `eye_shape`.

## Rays

`Rays` is an immutable dataclass, defined in `multi_pinhole.rays` and
re-exported through `multi_pinhole.core`/`multi_pinhole`, that carries the
geometric result of projecting scene points through one eye. It records,
for each input point:

* `Z`: the eye-frame axial distance from the eye to the point (signed, so
  callers can distinguish front- from back-facing samples).
* `XY`: the projected impact location on the screen, in camera `(X, Y)`
  coordinates — `NaN` for samples that are behind the eye or otherwise
  excluded, so downstream code can mask with `np.isnan` instead of tracking
  a separate index list.
* `zoom_rate`: the magnification factor `1 + f / Z` (see the projection
  derivation below) needed to dilate the eye's physical footprint before
  rasterizing it onto the screen.
* `front_and_visible`: a boolean mask combining "in front of the eye"
  (`Z > 0`) with any aperture-occlusion result passed in from outside.

All four arrays are aligned and support fancy indexing (`Rays.__getitem__`),
so a caller can slice all four fields with one boolean mask.
`Rays.n` and `Rays.n_visible` expose the total and surviving ray counts,
letting screening code size sparse buffers or skip empty batches without a
separate `np.count_nonzero` call.

A `Rays` instance sits between `Eye.calc_rays` (which produces it) and the
`Screen` rasterizers (which consume it) — it is pure data, with no reference
back to the `Eye` or `Camera` that created it.

## Eye: the pinhole projection

An `Eye` converts a 3D point already expressed in camera coordinates into a
2D landing spot on the screen. `Eye.calc_rays` does this in four steps
(mirroring the docstring's own summary):

1. **Translate into eye coordinates.** `camera2eye` subtracts the eye's
   camera-frame position: `points_in_eye = points_in_camera − eye.position`.
2. **Classify each point.** A point contributes to the image only if it is
   in front of the eye (`Z = points_in_eye[:, 2] > 0`, when `front_only=True`)
   *and* survives any externally-supplied visibility mask (aperture
   occlusion — see below). The combined mask is `front_and_visible`.
3. **Project onto the screen.** For every surviving point, the classic
   pinhole-camera relation is applied in camera `(X, Y)` units:

   ```
   XY = −(X', Y') / Z' · f + principal_point[:2]
   ```

   i.e. divide the lateral eye-frame offset by the (signed) depth, scale by
   the focal length, negate (a pinhole image is inverted), and re-center on
   the eye's principal point on the screen. Concretely, for a pinhole eye
   built with `position=(5, 0)`, `focal_length=20` (so
   `eye.position = eye.principal_point = (5, 0, 20)`, per `Eye.__init__`), a
   source point that lands at eye-frame coordinates `(X', Y', Z') = (2, 0, 10)`
   (an illustrative point, not quoted from the source — the source code has
   no worked numeric example of `calc_rays` itself) maps to
   `XY = −(2, 0)/10 · 20 + (5, 0) = (5 − 4, 0) = (1, 0)`.
4. **Compute the zoom rate.** `zoom_rate = 1 + f / Z'`. This is the
   magnification of the eye's own aperture footprint at that depth: a
   finite-size pinhole/lens does not focus a point source to a perfect
   point except exactly at its focal plane, so at any other depth `Z'` the
   source blurs into a spot on the screen whose size is the eye's physical
   aperture size scaled by `zoom_rate` (see `Screen.ray2image_grid` below).
   Points exactly at `Z' = 0` are excluded by the front-facing test, so this
   never divides by zero for surviving points.

`Eye.camera2eye` is the vectorized building block for step 1; everything
past that is inline in `calc_rays`.

## Aperture: an occluding shape

`Aperture` describes the physical opening that limits light reaching an eye.
It accepts either an analytic shape (circle, ellipse, rectangle) or an
explicit STL mesh; for analytic shapes, `Aperture.set_model` builds an STL
mesh on demand via `stl_utils.generate_aperture_stl` (a Delaunay
triangulation of the shape's interior — see `docs/utilities.md` for how that
mesh is constructed) and translates it to the aperture's position.

That STL mesh is not decorative — it is the actual geometry used for
occlusion testing. When a `Camera` projects points (`calc_image_vec`, below),
it calls `stl_utils.check_visible(mesh_obj=aperture.stl_model, start=eye.position,
grid_points=points_in_camera, ...)` for every aperture on the camera, and a
point only remains visible if it clears *every* aperture's mesh. In other
words, **apertures are treated as blocking surfaces**: the STL mesh is the
opaque material around the opening, and `check_visible` is a ray/mesh
intersection test (Möller–Trumbore, prefiltered by a bounding-cone test —
see `docs/utilities.md`) between the eye and each candidate point. A point
survives only if the segment from the eye to that point does *not* cross
the mesh.

## Screen: pixel/subpixel geometry and rasterization

The `Screen` represents the detector plane. `Screen.__init__` validates the
physical `screen_shape`/`screen_size`, lays out a `pixel_shape = (U_p, V_p)`
grid of pixels across that rectangle, and computes each pixel's center via
`positions()` — a simple `linspace` over each axis, offset by half a pixel
so centers sit at the middle of each cell (not on its edge).
Setting `subpixel_resolution = k` subdivides every pixel into a `k × k`
finer sub-grid (`_set_variables`), and builds a sparse `transform_matrix`
that sums each pixel's `k²` subpixels back together — this is how
`Screen.subpixel_to_pixel` downsamples a high-resolution subpixel image to
the coarser pixel grid via one sparse matrix-vector product instead of a
Python loop.
`image_mask` marks pixels/subpixels outside a circular or elliptical screen
(no masking for a rectangular screen) so they can be zeroed in displayed
images.

### Cosine falloff and etendue weighting

`Screen.cosine(eye)` computes, for every *subpixel*, the cosine of the angle
between the eye's optical axis and the line from the eye to that subpixel:
if `tangent = |subpixel_position − eye_position| / focal_length`, then
`cosine = 1 / sqrt(1 + tangent²)` (the standard `cos(atan(x)) = 1/√(1+x²)`
identity). `etendue_per_subpixel` retains
the corresponding small-aperture `A_subpixel · cos⁴(θ) / (4π)` diagnostic,
but the production rasterizer no longer reuses that detector-only value: a
finite Eye requires source- and Eye-position-dependent local ray geometry.

### `ray2image_grid`: turning a ray bundle into a sparse image

`Screen.ray2image_grid` is the public detector rasterizer.
For a `Rays` bundle, it builds a `(N_subpixel, n_rays)` sparse matrix where
column `r` holds the etendue-weighted subpixel footprint of ray `r`. The
algorithm:

1. **Convert to image coordinates.** `uv = xy2uv(rays.XY)`.
2. **Compute each ray's on-screen spot size.** Each eye has a physical
   `eye_size` (its own aperture extent); at a given ray's magnification, the
   *projected* footprint half-extent is `half = 0.5 · eye_size · zoom_rate`.
   This is the geometric-optics blur disk (or ellipse/rectangle) that a
   finite-size pinhole/lens casts onto the screen for a point source at that
   depth — the further a source point is from the focal plane, the larger
   `zoom_rate` and thus the blur spot.
3. **Cheaply reject rays whose footprint misses the screen** entirely, via
   an axis-aligned bounding-box (AABB) test against the subpixel grid's
   extent (`u_min..u_max`, `v_min..v_max`) — this is documented inline with
   an ASCII diagram of the bounding box around the (possibly non-circular)
   footprint.
4. **For each surviving ray, clip to a small tile of candidate subpixels**
   (`i_min..i_max`, `j_min..j_max` — the subpixel index range the footprint's
   AABB overlaps), then compute the area shared by each cell and the spot.
   Fully covered cells use their whole area, disjoint cells use zero, and
   boundary cells use analytic ellipse/rectangle or rectangle/rectangle
   intersection. Restricting this calculation to the AABB tile (rather than
   the whole screen) is what makes this scale to millions of rays. Because
   coverage is no longer decided from the cell center, a spot smaller than
   one pixel is retained even with `subpixel_resolution=1`.
5. **Integrate local-ray etendue and assemble the sparse matrix.** A detector
   point `q` in the footprint is mapped back to its finite-Eye position by
   `a = (q - q_center) / zoom_rate`. If `rho` is the transverse source offset
   from the Eye centre, the local distance is
   `D² = Z² + |rho - a|²`, and the detector-area density is
   `Z / (4π · zoom_rate² · D³)`. The rasterizer integrates this density over
   each spot/cell overlap with deterministic quadrature, then
   stores the resulting weights directly in CSR/CSC form. A complete spot
   contained in one large detector pixel is integrated over the Eye shape
   itself, so local etendue does not depend on detector subpixel refinement.

Ellipse/cell overlap *area* is analytic, which prevents a small spot from
vanishing merely because no cell center hits it. Density integration is not
generally analytic: rectangles use 2×2 Gauss quadrature, an ellipse completely
inside one cell uses 2 radial × 8 angular quadrature, and a clipped ellipse
boundary cell uses a 4×4 masked midpoint approximation. The last case has no
strict error guarantee. Subpixel refinement can therefore change local PSF
accuracy. A spot clipped at the physical screen boundary contributes less
total signal because only its on-screen part is detected.

The other `Screen` helpers are simpler coordinate/accumulation utilities:
`xy2uv` (camera `(X,Y)` → image `(u,v)`, described above),
`uv2subpixel_index` (image coordinates → integer subpixel indices, dropping
out-of-range hits), `subpixel_to_pixel` (sparse downsampling, described
above), and `show_image` (Matplotlib display of a pixel or subpixel
image).

## Camera: tying eyes, apertures, and the screen together

`Camera` groups one or more `Eye` instances (all sharing the same
`eye_type`), a list of `Aperture` objects, and a single `Screen`, and
positions/orients the whole assembly in world space via a `camera_position`
and `rotation_matrix`. Eye spots may be
smaller than a pixel or subpixel; analytic overlap weighting preserves their
area without imposing a detector-resolution constraint at construction.

For the common one-screen/one-pinhole case, `Camera.single_pinhole(...)`
constructs the geometry in a local reference pose: the screen and eye are
centered on the camera origin, the world-space position is `(0, 0, 0)`, and
the rotation is the identity. Place that rigid assembly afterward:

```python
camera = Camera.single_pinhole(
    focal_length=25,
    eye_size=1,
    screen_size=61 * 0.13,
    pixel_shape=(61, 61),
    subpixel_resolution=5,
    apertures=aperture,
).set_rotation_euler(
    "zxz", (2.9, 98, -19), degrees=True,
).set_camera_position(
    world_position,
).translate_camera(
    (4.15, 0, 0),
)
```

`set_camera_position` and `set_rotation_euler` are absolute setters.
`translate_world` applies a world-coordinate offset, while
`translate_camera` converts a camera-coordinate offset through the current
rotation before moving the camera, so rotation and camera-local translation
are order-dependent. `set_rotation_matrix(matrix)` accepts a world-to-camera
matrix directly. When CAD provides the final screen axes instead of Euler
angles, `set_orientation(look=normal, right=screen_right)` (or its `down=`
form) orthonormalizes those world-coordinate directions and constructs the
same matrix directly. If CAD supplies points rather than direction vectors,
first set the camera position and call
`set_orientation_from_points(look_point=..., right_point=...)` (or its
`down_point=` form); the method subtracts `camera_position` before applying
the same orientation calculation.

Registering a camera in a `World` freezes its optical configuration. The
camera pose and all `Eye`, `Screen`, and `Aperture` geometry become
immutable; exposed NumPy geometry arrays, screen sparse mappings, and STL
data buffers are made read-only, and mutating methods raise `RuntimeError`.
The `eyes` and `apertures` properties expose tuples so their collections
cannot be edited externally. To change a registered setup, construct a new
camera and pass it to `World.change_camera`. A frozen camera may safely be
shared by multiple worlds, and removing it from a world does not thaw it.

### `calc_image_vec`: world points → sparse screen image, step by step

`Camera.calc_image_vec(eye_num, points, ...)` is the top-level entry point
that a `World` calls (once per camera eye) to project a batch of world
points through one eye. It runs three steps:

1. **World → camera.** `points_in_camera = self.world2camera(points)`.
2. **Aperture visibility (optional, on by default).** For every `Aperture`
   on the camera, run `stl_utils.check_visible(mesh_obj=aperture.stl_model,
   start=eye.position, grid_points=points_in_camera, behind_start_included=True)`.
   A point is `visible` only if it clears *all* apertures
   (`np.all(visible_list, axis=0)`) — see the Aperture section above for what
   "clears" means geometrically.
3. **Ray tracing and rasterization.** `eye.calc_rays(points_in_camera,
   visible)` produces a `Rays` bundle (pinhole projection, described above),
   which `screen.ray2image_grid(eye, rays)` turns into the final
   `(N_subpixel, n_points)` sparse matrix.

Note that this matrix only encodes *which subpixels each point's ray
reaches, and with what etendue weight* — it does not know the emission
intensity of each point. Multiplying it by a per-point (or, after
integration, per-voxel) intensity vector is what produces an actual image;
`multi_pinhole.world.World` handles that integration step (see
`docs/world.md`).

### Worked example: tracing one ray end to end

Using a pinhole eye built with `position=(5, 0)`, `focal_length=20` (so
`eye.position = (5, 0, 20)`), mounted on a camera with
`camera_position=(0, 0, -60)` and identity rotation:

1. A world point at `(2, 0, -30)` maps to camera coordinates
   `(2, 0, -30) − (0, 0, -60) = (2, 0, 30)` (identity rotation leaves it
   unchanged).
2. In eye coordinates: subtract the eye position `(5, 0, 20)`, giving
   `(X', Y', Z') = (-3, 0, 10)`. `Z' = 10 > 0`, so this point is in front of
   the eye and survives the front-facing test (a point with `Z' = 0` would
   sit exactly at the pinhole's own depth and be excluded, since the
   projection formula would divide by zero there).
3. The projection formula gives
   `XY = −(-3, 0)/10 · 20 + (5, 0) = (6 + 5, 0) = (11, 0)` in camera
   coordinates, and `zoom_rate = 1 + 20/10 = 3`.
4. `Screen.xy2uv` converts `(11, 0)` to image coordinates by flipping the
   axis order and re-centering: `uv = (0, 11) + screen_size/2`.
5. `ray2image_grid` then spreads this single ray over the subpixels whose
   centers fall within `eye_size/2 · zoom_rate` of that `uv` point (a disk
   for a circular eye, scaled by the magnification from step 3), weighting
   each hit subpixel by the etendue factors described above.

### Visualization helpers

`draw_optical_system`, `draw_camera_orientation_plotly`, and
`draw_camera_orientation` render the eyes, apertures, and screen in
Matplotlib or Plotly 3D scenes for alignment and debugging; they are pure
visualization and do not affect the projection math above.

Together these classes provide the foundation that `multi_pinhole.world`
and `multi_pinhole.voxel` build on to simulate complete multi-aperture
imaging experiments — see `docs/world.md` for how a `World` turns
`calc_image_vec` calls on individual points into a full voxel-to-screen
projection matrix.
