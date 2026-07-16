"""Lightweight regressions for the 0.7.3 public documentation contract."""

import ast
import inspect
from pathlib import Path
import re

import multi_pinhole
from multi_pinhole import coordinates, profiles, projection
from multi_pinhole.__about__ import __version__
from multi_pinhole.world import PROJECTION_CACHE_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_release_and_top_level_public_import_contract():
    assert __version__ == "0.7.3"
    expected = {
        "Rays", "Eye", "Aperture", "Screen", "Camera", "Voxel", "World",
        "EyeProjectionWorkEstimate", "ProjectionWorkEstimate",
    }
    assert expected <= set(vars(multi_pinhole))


def _module_public_functions(module):
    """Return only non-private functions defined directly in a module AST."""
    tree = ast.parse(inspect.getsource(module))
    return [node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")]


def test_coordinate_and_profile_public_functions_use_numpy_sections():
    for module in (coordinates, profiles):
        names = _module_public_functions(module)
        assert names
        for name in names:
            function = getattr(module, name)
            doc = inspect.getdoc(function) or ""
            assert "Parameters\n----------" in doc, f"missing Parameters: {module.__name__}.{name}"
            assert "Returns\n-------" in doc, f"missing Returns: {module.__name__}.{name}"


def test_projection_public_contract_docstrings():
    functions = (
        projection.projected_axis_spans,
        projection.select_source_resolution,
        projection.select_circumsphere_resolution,
        projection.make_optical_binning,
    )
    for function in functions:
        doc = inspect.getdoc(function) or ""
        assert "Parameters\n----------" in doc
        assert "Returns\n-------" in doc
    for type_ in (
        projection.SourceResolutionEstimate,
        projection.PointSourceResolutionEstimate,
        projection.EyeProjectionWorkEstimate,
        projection.ProjectionWorkEstimate,
        projection.OpticalBinning,
    ):
        assert "Attributes\n----------" in (inspect.getdoc(type_) or "")


def test_critical_public_methods_document_inputs_and_outputs():
    methods = (
        multi_pinhole.Screen.ray2image_grid,
        multi_pinhole.World.project,
        multi_pinhole.World.backproject,
        multi_pinhole.World.set_projection_matrix,
    )
    for method in methods:
        doc = inspect.getdoc(method) or ""
        assert "Parameters\n----------" in doc
        assert "Returns\n-------" in doc or method.__name__ == "set_projection_matrix"


def test_critical_signature_defaults_are_documented_and_unchanged():
    assert inspect.signature(multi_pinhole.World.preflight_projection).parameters["res_mode"].default == "fixed"
    assert inspect.signature(multi_pinhole.World.set_projection_matrix).parameters["parallel"].default == -1
    assert inspect.signature(multi_pinhole.Screen.ray2image_grid).parameters["verbose"].default == 0
    checks = (
        (multi_pinhole.World.preflight_projection, 'default="fixed"'),
        (multi_pinhole.World.set_projection_matrix, "default=-1"),
        (multi_pinhole.Screen.ray2image_grid, "default=0"),
    )
    for function, text in checks:
        assert text in (inspect.getdoc(function) or "")


def test_markdown_links_and_forbidden_source_references():
    files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    link_pattern = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "【F:" not in text
        for target in link_pattern.findall(text):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            assert (path.parent / target).resolve().exists(), f"broken link in {path}: {target}"


def test_readme_projection_workflow_executes():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for symbol in ("set_inside_vertices", "preflight_projection",
                   "set_projection_matrix", "world.project", "world.backproject"):
        assert symbol in text
    assert "sub-classes" not in text
    section = text.split("## Minimal projection workflow", 1)[1]
    code = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
    assert code, "README projection workflow needs a Python code block"
    exec(compile(code.group(1), "README.md", "exec"), {})


def test_release_schema_contract():
    assert __version__ == "0.7.3"
    assert PROJECTION_CACHE_SCHEMA_VERSION == 3
