"""CO3's loader as Paper 2 uses it: co3_common.py defines what paper2_rows.py and the dial trainer take from it.

co3_common.py sits at 2-classifier-guidance/scripts/method_search/co3_common.py, restored on 2026-11-26 from the teaching
scene's co3_common.py.copy. The first tests read the files with ast and run on the laptop; the last imports them for
real and is skipped where stanza, CO3's composers package or torch is missing, so on the held node it runs:

    python3 -m pytest tests/test_paper2_co3_import.py -q
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
METHOD_SEARCH = REPO / "2-classifier-guidance/scripts/method_search"
CO3_COMMON = METHOD_SEARCH / "co3_common.py"
ROWS = METHOD_SEARCH / "paper2_rows.py"
DIAL = REPO / "2-classifier-guidance/scripts/paper2/paper2_identity_dial.py"
HUB_COPY = REPO.parent / "replications/published-methods-on-our-pairs/scripts/reproductions/co3_common.py"


def top_level_names(path: Path) -> set[str]:
    """Names a module defines or imports at its top level."""
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def imported_from(path: Path, module: str) -> set[str]:
    return {a.name for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ImportFrom) and node.module == module for a in node.names}


def attributes_used(path: Path, obj: str) -> set[str]:
    return {node.attr for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == obj}


def test_co3_common_is_present():
    assert CO3_COMMON.exists(), f"{CO3_COMMON} is missing: CO3's row, the dial trainer and task 8 import it"


def test_co3_common_defines_what_paper2_rows_imports():
    wanted = imported_from(ROWS, "co3_common")
    assert wanted == {"load", "load_parser", "set_prompt", "start_noise"}   # paper2_rows.py line 23
    assert wanted <= top_level_names(CO3_COMMON)


def test_everything_the_trainer_takes_from_paper2_rows_exists():
    used = attributes_used(DIAL, "rows")          # co3_sampler: `import paper2_rows as rows`
    assert used == {"load", "load_parser", "set_prompt", "start_noise", "encode"}
    rows_names = top_level_names(ROWS)
    assert used <= rows_names
    # those paper2_rows takes from co3_common must be defined there
    assert (used & imported_from(ROWS, "co3_common")) <= top_level_names(CO3_COMMON)


def test_co3_common_matches_the_hubs_tracked_copy():
    if not HUB_COPY.exists():
        pytest.skip("the hub repository is not on this machine")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(CO3_COMMON) == digest(HUB_COPY)


MISSING = [m for m in ("torch", "stanza", "composers") if importlib.util.find_spec(m) is None]


@pytest.mark.skipif(bool(MISSING), reason=f"needs {', '.join(MISSING)}: runs on the held node with PYTHONPATH from "
                    "paper2_guards.sh (p2_co3_path)")
def test_paper2_rows_and_co3_common_import_for_real():
    sys.path.insert(0, str(METHOD_SEARCH))
    sys.path.insert(0, str(METHOD_SEARCH.parent / "rung2"))
    sys.path.insert(0, str(REPO))
    import co3_common
    import paper2_rows
    assert Path(co3_common.__file__).resolve() == CO3_COMMON.resolve()
    for name in ("load", "load_parser", "set_prompt", "start_noise", "encode"):
        assert callable(getattr(paper2_rows, name))
