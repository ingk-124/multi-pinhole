# Project Overview

## Purpose
The multi-pinhole project simulates imaging systems that combine multiple pinhole or concave-lens "eyes" with configurable screens, apertures, and world geometry. The core module formalizes the optical coordinate systems that tie the world, camera, pinhole, and image spaces together, which enables consistent ray tracing across components.【F:multi_pinhole/core.py†L38-L99】 This foundation supports building cameras with multiple optical channels and capturing their interaction with voxelized scenes and STL-defined structures.

## Key Components
- **Core optics** – `multi_pinhole.core` defines reusable classes for rays, eyes, apertures, screens, and cameras, along with utilities for projecting 3D world points onto image planes and visualizing the optical layout.【F:multi_pinhole/core.py†L101-L1804】
- **Voxel modeling** – `multi_pinhole.voxel` provides coordinate transforms (Cartesian, toroidal, cylindrical, spherical) and voxel grid logic for representing plasma or other volumetric targets inside the world.【F:multi_pinhole/voxel.py†L10-L160】
- **World orchestration** – `multi_pinhole.world` binds voxels, cameras, and optional walls into a simulation-ready environment, manages visibility checks, and exposes helpers for summarizing or persisting a scenario.【F:multi_pinhole/world.py†L70-L167】

## Typical Workflow
1. **Describe the scene** by instantiating a `Voxel` (or using helpers such as `shifted_torus` or `helical_island`) and optional STL meshes that bound the environment.【F:multi_pinhole/voxel.py†L10-L160】【F:multi_pinhole/world.py†L70-L118】
2. **Configure optics** by creating one or more `Eye` objects, pairing them with `Aperture` geometries, and attaching them to a `Screen` that defines pixel and subpixel sampling.【F:multi_pinhole/core.py†L177-L1041】
3. **Assemble a camera** by combining the eyes, apertures, and screen, positioning and orienting the rig in world space, and linking it to the world container.【F:multi_pinhole/core.py†L1330-L1560】【F:multi_pinhole/world.py†L97-L118】
4. **Project world points** using `Camera.calc_image_vec`, which converts world coordinates to camera coordinates, filters visibility through apertures, computes rays per eye, and rasterizes their footprint onto the screen’s sparse image representation.【F:multi_pinhole/core.py†L1558-L1587】
5. **Inspect results** by converting subpixel responses to pixel images or plotting camera and optical layouts for debugging and presentation.【F:multi_pinhole/core.py†L1042-L1804】

## Notable Capabilities
- Multiple coordinate systems (Cartesian, cylindrical, toroidal, spherical) make it easier to model devices with complex symmetries while still rendering them through a consistent camera interface.【F:multi_pinhole/core.py†L38-L90】【F:multi_pinhole/voxel.py†L10-L160】
- Aperture support accepts analytic shapes or STL meshes, allowing detailed mechanical masks to gate light paths.【F:multi_pinhole/core.py†L425-L544】
- Screen sampling uses sparse matrices to efficiently accumulate contributions from millions of rays while supporting subpixel refinement and etendue-aware weighting.【F:multi_pinhole/core.py†L546-L1320】
- Camera utilities include 3D Matplotlib and Plotly visualizations that show optical axes, screen geometry, and apertures for alignment checks.【F:multi_pinhole/core.py†L1589-L1786】

## Extensibility
The project layers utility functions (e.g., STL processing, filter characteristics) beneath the main classes so that new optical elements or custom workflows can tap into existing coordinate transforms, visibility checks, and visualization routines without re-implementing the core math.【F:multi_pinhole/core.py†L133-L1586】【F:multi_pinhole/world.py†L70-L167】 Future improvements can add richer material models, additional lens types, or automated calibration while reusing the documented interfaces.
