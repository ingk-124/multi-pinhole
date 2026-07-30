# World archive and serialization contract

Use `world.save("scene.mpw")` for a complete checkpoint and
`World.load("scene.mpw")` to restore it. `World.inspect_archive(path)` reads
metadata without unpickling the World.

## Container and manifest

An `.mpw` file is a ZIP archive:

```text
scene.mpw
├── manifest.json
└── world.pkl
```

`manifest.json` is UTF-8 JSON with:

```json
{
  "format": "multi-pinhole-world",
  "world_schema_version": 1,
  "library_version": "1.0.0",
  "projection_cache_schema_version": 3,
  "python_version": "...",
  "numpy_version": "...",
  "scipy_version": "..."
}
```

The three version dimensions are independent:

- `library_version` identifies the package that wrote the archive;
- `world_schema_version` governs the World payload/container contract;
- `projection_cache_schema_version` governs matrix meaning, shape, ordering,
  and storage.

A module move alone does not change projection cache schema. Version 1.0
retains schema 3 because the numeric representation is unchanged.

## Loading and cache migration

World schema 1 is loaded normally. An unsupported World schema fails before
the payload is unpickled. If the projection cache schema is 3, visibility,
per-Eye projection, projection settings, and aggregated `P_matrix` remain
available. If it is incompatible, visibility is retained where possible and
projection data alone is cleared for safe recomputation.

`World.load` also detects old direct-dill files. They have no inspectable
manifest, so `inspect_archive` reports `format="legacy-direct-dill"` and
unknown version fields. After loading a trusted legacy file, call
`world.save("migrated.mpw")`. The compatibility aliases `save_world` and
`load_world` use the new writer and auto-detecting loader.

Saving serializes to a temporary file in the destination directory and then
uses atomic replacement. It does not normalize or invalidate caches on the
live World.

## Security boundary

The JSON manifest is safe to inspect as data. `World.load` executes
pickle/dill reconstruction from `world.pkl` or a legacy file. Pickle/dill can
execute arbitrary code. Never load an archive or legacy World from an
untrusted or unauthenticated source.

The JSON [World config schema](config.md) does not execute callables and is
the appropriate interchange format when caches and arbitrary Python state
are unnecessary.

