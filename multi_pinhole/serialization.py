"""Versioned World archives and legacy direct-dill migration.

World archives are ZIP files containing a JSON manifest and one dill payload.
The manifest is safe to inspect without executing pickle opcodes.  Loading
either archive or legacy formats still executes pickle/dill data and therefore
must be restricted to trusted files.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dill
import numpy as np
import scipy

from .__about__ import __version__

if TYPE_CHECKING:
    from .world import World

WORLD_ARCHIVE_FORMAT = "multi-pinhole-world"
WORLD_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
WORLD_PAYLOAD_NAME = "world.pkl"

_MANIFEST_FIELDS = {
    "format",
    "world_schema_version",
    "library_version",
    "projection_cache_schema_version",
    "python_version",
    "numpy_version",
    "scipy_version",
}


class WorldSerializationError(ValueError):
    """Raised when a World archive is corrupt, unsupported, or inconsistent."""


def _current_manifest() -> dict[str, Any]:
    from .world import PROJECTION_CACHE_SCHEMA_VERSION

    return {
        "format": WORLD_ARCHIVE_FORMAT,
        "world_schema_version": WORLD_SCHEMA_VERSION,
        "library_version": __version__,
        "projection_cache_schema_version": PROJECTION_CACHE_SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def _validate_manifest(value: Any, source: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorldSerializationError(f"{source}: manifest must be a JSON object")
    manifest = dict(value)
    missing = _MANIFEST_FIELDS - manifest.keys()
    if missing:
        raise WorldSerializationError(
            f"{source}: manifest is missing field(s): {', '.join(sorted(missing))}"
        )
    if manifest["format"] != WORLD_ARCHIVE_FORMAT:
        raise WorldSerializationError(
            f"{source}: unknown World archive format {manifest['format']!r}"
        )
    if isinstance(manifest["world_schema_version"], bool) or not isinstance(
        manifest["world_schema_version"], int
    ):
        raise WorldSerializationError(
            f"{source}: world_schema_version must be an integer"
        )
    if isinstance(manifest["projection_cache_schema_version"], bool) or not isinstance(
        manifest["projection_cache_schema_version"], int
    ):
        raise WorldSerializationError(
            f"{source}: projection_cache_schema_version must be an integer"
        )
    for field in (
        "library_version",
        "python_version",
        "numpy_version",
        "scipy_version",
    ):
        if not isinstance(manifest[field], str):
            raise WorldSerializationError(f"{source}: {field} must be a string")
    return manifest


def inspect_world_archive(source: str | os.PathLike[str]) -> dict[str, Any]:
    """Read World metadata without unpickling the World payload.

    Legacy direct-dill files have no inspectable manifest and are reported with
    ``format="legacy-direct-dill"`` and unknown version fields.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not zipfile.is_zipfile(path):
        return {
            "format": "legacy-direct-dill",
            "world_schema_version": None,
            "library_version": None,
            "projection_cache_schema_version": None,
            "python_version": None,
            "numpy_version": None,
            "scipy_version": None,
        }
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = set(archive.namelist())
            if MANIFEST_NAME not in names:
                raise WorldSerializationError(
                    f"{path}: archive does not contain {MANIFEST_NAME}"
                )
            try:
                manifest = json.loads(archive.read(MANIFEST_NAME))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise WorldSerializationError(
                    f"{path}: invalid {MANIFEST_NAME}: {error}"
                ) from error
    except (OSError, zipfile.BadZipFile) as error:
        raise WorldSerializationError(
            f"cannot inspect World archive {path}: {error}"
        ) from error
    return _validate_manifest(manifest, path)


def save_world_archive(world: "World", destination: str | os.PathLike[str]) -> None:
    """Write a World archive atomically without mutating World state or caches."""
    from .world import World

    if not isinstance(world, World):
        raise TypeError("world must be a World")
    path = Path(destination)
    parent = path.parent
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    manifest = _current_manifest()
    payload = dill.dumps(world)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(
                MANIFEST_NAME,
                json.dumps(
                    manifest,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n",
            )
            archive.writestr(WORLD_PAYLOAD_NAME, payload)
        os.replace(temporary_path, path)
        temporary_path = None
    except (OSError, dill.PicklingError, zipfile.BadZipFile) as error:
        raise WorldSerializationError(
            f"cannot save World archive {path}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _load_archive(path: Path, manifest: Mapping[str, Any]) -> "World":
    if manifest["world_schema_version"] != WORLD_SCHEMA_VERSION:
        raise WorldSerializationError(
            f"{path}: unsupported world_schema_version "
            f"{manifest['world_schema_version']!r}; supported version is "
            f"{WORLD_SCHEMA_VERSION}"
        )
    try:
        with zipfile.ZipFile(path, "r") as archive:
            if WORLD_PAYLOAD_NAME not in archive.namelist():
                raise WorldSerializationError(
                    f"{path}: archive does not contain {WORLD_PAYLOAD_NAME}"
                )
            payload = archive.read(WORLD_PAYLOAD_NAME)
        return dill.loads(payload)
    except WorldSerializationError:
        raise
    except (
        OSError,
        EOFError,
        ImportError,
        AttributeError,
        TypeError,
        zipfile.BadZipFile,
    ) as error:
        raise WorldSerializationError(
            f"cannot load World archive {path}: {error}"
        ) from error


def _load_legacy(path: Path) -> "World":
    try:
        with path.open("rb") as file:
            return dill.load(file)
    except (OSError, EOFError, ImportError, AttributeError, TypeError) as error:
        raise WorldSerializationError(
            f"cannot load legacy direct-dill World {path}: {error}"
        ) from error


def load_world_archive(source: str | os.PathLike[str]) -> "World":
    """Load a current archive or migrate a trusted legacy direct-dill World."""
    from .world import PROJECTION_CACHE_SCHEMA_VERSION, World

    path = Path(source)
    if zipfile.is_zipfile(path):
        manifest = inspect_world_archive(path)
        loaded = _load_archive(path, manifest)
        manifest_cache_version = manifest["projection_cache_schema_version"]
    else:
        loaded = _load_legacy(path)
        manifest_cache_version = getattr(
            loaded, "_projection_cache_schema_version", None
        )
    if not isinstance(loaded, World):
        raise TypeError(f"{path}: serialized object is not a World")

    if manifest_cache_version is None:
        loaded._migrate_unversioned_projection_cache()
    elif manifest_cache_version != PROJECTION_CACHE_SCHEMA_VERSION:
        loaded._invalidate_projection_cache()
        loaded._projection_cache_schema_version = PROJECTION_CACHE_SCHEMA_VERSION
    else:
        loaded._ensure_projection_cache_schema()
    if not hasattr(loaded, "_config_wall_paths"):
        loaded._config_wall_paths = ()
    return loaded
