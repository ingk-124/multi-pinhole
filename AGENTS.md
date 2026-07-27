# Repository Guidelines

These instructions apply to the entire repository.

See `docs/ja/development-roadmap.md` for the longer-term API, compatibility,
package-structure, and release policy.

## Compatibility

- Support Python 3.10 and newer.
- Prioritize loading existing serialized Worlds and reusing visibility and
  projection caches over preserving every internal import path.
- Keep lightweight import shims, including `multi_pinhole.core`, while old
  pickle or dill artifacts still reference them.
- Do not bump the projection-cache schema for a module move alone. Bump it
  only when matrix meaning, shape, ordering, or storage changes.
- Existing scientific names such as `N_pixel` and `P_matrix` do not need to be
  renamed solely to satisfy snake_case conventions.

## Style and documentation

- Keep lines at 88 characters where practical. Lines up to 120 characters are
  acceptable when wrapping would reduce readability, especially for formulas,
  URLs, test data, and diagnostic messages.
- Document public APIs with their purpose, parameters, return values,
  user-visible exceptions, array shapes, and physical units where applicable.
- Add internal docstrings or comments when they explain contracts, invariants,
  numerical assumptions, or why an algorithm is structured a particular way.
  Do not add comments that merely restate the code.

## Dependencies

- Treat `pyproject.toml` as the source of truth for runtime dependencies.
- Keep `requirements.txt` synchronized with the direct runtime dependencies.
- Do not add or install dependencies without explicit approval.

## Validation

After changing Python code, run:

```bash
python -m compileall -q multi_pinhole tests examples benchmarks
pytest -q
git diff --check
git status --short
```

- Do not use automatic fixes without reviewing their diff.
- Do not commit generated caches, IDE metadata, or `*.egg-info` directories.
