# Coordinates, profiles, and interpolation

> **Level 1 workflow with Level 2 reference.** The first half shows the normal
> path from Cartesian voxel centers to a plasma profile and a detector image.
> The coordinate-convention table and interpolation notes in the second half
> are for checking signs, normalization, and numerical interpretation.

This page is the canonical guide for three related operations:

1. reinterpret Cartesian voxel points in a plasma-oriented coordinate system;
2. evaluate a reusable scalar profile on those coordinates;
3. interpolate voxel-center values at arbitrary points, such as an R–Z plane.

These operations do not change the `Voxel` grid. It remains Cartesian.

## From a Voxel to an emission profile

Profile functions are imported as a module:

```python
from multi_pinhole import profiles
```

They are not added as individual names by `from multi_pinhole import *`.
Reusable profiles take normalized poloidal Cartesian coordinates

$$
x=(R-R_0)/a,\qquad y=Z/a,
$$

where `+x` points radially outward and `+y` points upward.

```python
x, y, phi = voxel.to_coordinates(
    "poloidal_cartesian_inverse",
    normalized=True,
    major_radius=508,
    minor_radius=250,
).T

emission = profiles.axisymmetric_profile(
    x,
    y,
    A=1.0,
    delta=0.1,
    alpha=2.0,
    beta=3.0,
    edge_value=0.0,
)
```

`emission` has shape `(N_voxel,)` and is aligned with
`voxel.gravity_center`. It can be passed directly to:

```python
image = world.project(emission, camera_idx="main")
```

Use zero outside the emitting region. NaN propagates through the sparse
projection matrix.

## Choosing a profile

| Function | Use |
|---|---|
| `axisymmetric_profile` | Shifted but poloidally symmetric profile |
| `kinked_profile` | Radially dependent displacement toward a center angle |
| `flattening_profile` | Kinked profile with a localized flattened region |
| `helical_center_angle` | Propagate a measured center angle between toroidal positions |

All profile inputs broadcast according to NumPy rules. The amplitude `A` and
`edge_value` may carry application-defined physical units; coordinate and
shape parameters are dimensionless.

`center_angle_xy` is measured counter-clockwise from outward poloidal `+x`
toward upward `+y`. It is a poloidal Cartesian angle, independent of the sign
chosen for toroidal `phi`.

For a helical structure:

```python
center_angle_xy = profiles.helical_center_angle(
    phi,
    center_angle_xy_ref=0.2,
    m=1,
    n=-1,
    phi_ref=0.0,
)

emission = profiles.kinked_profile(
    x,
    y,
    A=1.0,
    delta=0.1,
    alpha=2.0,
    beta=3.0,
    xi_0=0.2,
    rho_s=0.5,
    d=2.0,
    center_angle_xy=center_angle_xy,
)
```

The helper evaluates
`center_angle_xy_ref + (n/m) * (phi - phi_ref)` without wrapping the result.
It does not detect the `phi` convention. The caller must use a signed `n`
consistent with `poloidal_cartesian` or `poloidal_cartesian_inverse`.

The complete parameter comparison is executable as:

```bash
python examples/profiles_demo.py
```

## Converting arbitrary points

`Voxel.to_coordinates` converts Cartesian points without changing the
configured legacy `voxel.coordinate_type`:

```python
coordinates = voxel.to_coordinates(
    "cylindrical",
    points=[[1.0, 2.0, 3.0]],
)
```

`points` may be `"centers"`, `"vertices"`, or an array with shape `(..., 3)`.
The inverse API accepts named, broadcastable components:

```python
xyz = voxel.from_coordinates(
    "cylindrical",
    R=np.linspace(1, 2, 5)[:, None],
    phi=np.linspace(0, 2 * np.pi, 100)[None, :],
    Z=0,
)
```

The singleton axes make a `(5, 100)` R–phi mesh. Two independent one-
dimensional arrays with shapes `(5,)` and `(100,)` do not broadcast.

`normalized_coordinates()` is the compatibility API using the coordinate
type stored on the Voxel. Prefer explicit `to_coordinates(...)` in new
analysis code so the convention and scale are visible at the call site.

## Coordinate conventions

| `coordinate_type` | Components | Angle convention | Geometry / normalization parameters |
|---|---|---|---|
| `cartesian` | `x, y, z` | — | `width`, `depth`, `height`; divide by half-ranges |
| `cylindrical` | `R, phi, Z` | `phi=atan2(y,x)`, counter-clockwise from `+x` viewed from `+z` | `radius`, `height`; `Z/(height/2)` |
| `torus` | `r, theta, phi` | `theta=0` outboard, upward positive; `phi` clockwise | `major_radius`, `minor_radius` |
| `torus_inverse` | `r, theta, phi` | `theta=0` inboard, upward positive; `phi` counter-clockwise | `major_radius`, `minor_radius` |
| `poloidal_cartesian` | `x, y, phi` | `x=R-R0`, `y=Z`; `phi` clockwise | `major_radius`, `minor_radius` |
| `poloidal_cartesian_inverse` | `x, y, phi` | same poloidal axes; `phi` counter-clockwise | `major_radius`, `minor_radius` |
| `spherical` | `r, theta, phi` | polar `theta` from `+z`; `phi` counter-clockwise | `radius` |

With `normalized=False`, radial and axial components retain the user's length
unit. Angles always use radians. With `normalized=True`, every required scale
must be supplied explicitly; missing scales are errors.

## Interpolating voxel-center values

`center_interpolator` wraps SciPy's `RegularGridInterpolator` on the three
voxel-center axes:

```python
interpolate = voxel.center_interpolator(
    emission,
    method="linear",
    bounds_error=False,
    fill_value=np.nan,
)

values = interpolate(points=[[500.0, 0.0, 20.0]])
```

The input field may have shape `voxel.shape`, `(N_voxel,)`, or trailing
vector/tensor dimensions. Cartesian query points have shape `(..., 3)`.
Alternatively, supply an explicit coordinate convention and named
components.

### Fixed-phi R–Z cross-section

For a physical R–Z plane, use cylindrical components. Cylindrical `phi` is
always counter-clockwise from world `+x` when viewed from `+z`.

```python
import matplotlib.pyplot as plt
import numpy as np

interpolate = voxel.center_interpolator(
    emission,
    method="linear",
    bounds_error=False,
    fill_value=np.nan,
)

R = np.linspace(250, 750, 201)
Z = np.linspace(-250, 250, 201)
RR, ZZ = np.meshgrid(R, Z, indexing="xy")

section = interpolate(
    coordinate_type="cylindrical",
    R=RR,
    phi=np.deg2rad(45),
    Z=ZZ,
)

fig, ax = plt.subplots()
mesh = ax.pcolormesh(RR, ZZ, section, shading="auto")
ax.set_aspect("equal")
ax.set_xlabel("R [mm]")
ax.set_ylabel("Z [mm]")
fig.colorbar(mesh, ax=ax, label="Emission")
plt.show()
```

This is interpolation of values already sampled at voxel centers. It is not a
volume integral and does not increase the physical resolution of the original
field. If an analytic profile function is still available, evaluating that
function directly on the display grid is more accurate than interpolating its
coarse voxel samples.

`plot_voxel_slice` serves a different purpose: it displays one existing
Cartesian voxel-center plane with physical cell boundaries and does not
interpolate. See [Visualization](visualization.md).
