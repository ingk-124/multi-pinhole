# Migrating 0.8/0.9 Worlds to 1.0

## Public imports

Use:

```python
from multi_pinhole import (
    Aperture,
    Camera,
    Eye,
    Rays,
    Screen,
    Voxel,
    World,
)
```

`multi_pinhole.__version__` is public. `multi_pinhole.core` is retained only
to resolve historical class globals in imports and pickle/dill. Its private
rasterizer helpers, `stl_utils`, and old type aliases are no longer exported.
Import implementation helpers from their defining modules only when internal
development genuinely requires them.

## Migrating a trusted saved World

```python
from multi_pinhole import World

world = World.load("old-world.pkl")
world.save("world-v1.mpw")
metadata = World.inspect_archive("world-v1.mpw")
assert metadata["world_schema_version"] == 1
assert metadata["projection_cache_schema_version"] == 3
```

The loader resolves historical `multi_pinhole.core.*` class globals.
Compatible schema-3 visibility and projection caches are preserved. A
missing or incompatible projection cache schema preserves visibility where
possible and clears projection only. Compare a representative
`world.project(emission, key)` result before and after migration.

Only load trusted legacy files: direct dill and the archive payload can
execute arbitrary code.

## Choosing config or archive

Use JSON config for reviewable scene construction and `.mpw` for complete
Python state plus caches. Config supports analytic Apertures, path-backed STL
walls, and the built-in `all`, `box`, and `sphere` inside masks. It rejects
arbitrary callables, standalone inside arrays, STL Apertures, and wall meshes
without source provenance rather than silently losing them.

No optics modules moved for 1.0. Projection formulae, processing order,
dtypes, tolerances, and projection cache schema 3 are unchanged.
