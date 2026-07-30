# multi-pinhole documentation

The documentation is arranged by how deeply you need to understand the
library. You do not need to read it in file-name order.

## Level 1 — Use the library

Start here if you want to define a camera, calculate a projection, and inspect
the result without studying the implementation.

1. [Overview and first projection](overview.md)
2. [Build a World JSON file](world-config-guide.md)
3. [Coordinates, profiles, and interpolation](coordinates-profiles.md)
4. [Visualize voxel fields and detector images](visualization.md)

The runnable examples are in [`examples/`](../examples). The
[`examples/relax`](../examples/relax/README.md) workflow is the most complete
JSON-based example.

> **Ordinary users can stop after Level 1.** `World`, `Voxel`, `Camera`,
> `World.from_config`, `World.set_projection_matrix`, and `World.project`
> cover the normal simulation workflow.

## Level 2 — Understand the scientific model

Read these sections when choosing numerical resolution, interpreting units,
checking coordinate conventions, or writing a methods section.

- [Pinhole optics and detector model](core.md) — coordinate frames, pinhole
  projection, aperture/eye meaning, finite-eye and detector quadrature.
- [World projection model](world.md) — visibility states, source integration,
  partial voxels, adaptive resolution, projection and backprojection.
- The second half of
  [Coordinates, profiles, and interpolation](coordinates-profiles.md) —
  exact angle conventions, normalization scales, and interpolation limits.

Each page begins with the user-visible contract. Sections marked
**Implementation detail** may be skipped unless you are validating or
extending the algorithms.

## Level 3 — Reference, persistence, and development

- [JSON schema reference](config.md) — exact accepted keys and errors.
- [World archives and serialization](serialization.md) — saving caches,
  compatibility, and the pickle security boundary.
- [Utilities and geometry internals](utilities.md) — STL intersection and
  low-level helpers.
- [Migration to 1.0](migration-v1.md) — old Worlds and import paths.
- [Development and release roadmap (Japanese)](ja/development-roadmap.md).

Roadmap documents describe future work, not current user-facing behavior.

## Where should new information go?

To keep related information together:

- a task-oriented procedure belongs in `overview`, `world-config-guide`, or
  `visualization`;
- a scientific assumption, equation, or approximation belongs in `core` or
  `world`;
- an exhaustive key/signature contract belongs in a reference page;
- private algorithms and performance notes belong at the end of the relevant
  scientific page, not in the getting-started flow.

[日本語の入口](ja/README.md)
