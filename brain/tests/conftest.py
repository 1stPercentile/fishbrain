"""A fresh clone runs the quick suite; `--rebuild` runs everything.

The tests named in rebuild_only.txt read the full Fish1 rebuild (the 2.5 GB wiring from `python -m fishbrain.pull`
and the G1, G1b and G1c pipeline outputs under data/). Without it they fail loudly, which is right for the lab and
useless for a first `pytest`. They are skipped by name, never edited, and `--rebuild` runs them as written.
"""
from pathlib import Path

import pytest

LIST = Path(__file__).with_name("rebuild_only.txt")


def pytest_addoption(parser):
    parser.addoption("--rebuild", action="store_true",
                     help="also run the tests that need the full Fish1 rebuild (see tests/rebuild_only.txt)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--rebuild"):
        return
    need = {ln.strip() for ln in LIST.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    skip = pytest.mark.skip(reason="needs the full Fish1 rebuild; run pytest --rebuild after python -m fishbrain.pull")
    for item in items:
        if f"{Path(str(item.fspath)).name}::{item.name}" in need:
            item.add_marker(skip)
