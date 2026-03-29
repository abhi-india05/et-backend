"""Compatibility package for backend-prefixed imports.

This repo is often deployed as a standalone backend project where the
repository root itself contains modules like ``api`` and ``main.py``.
Many modules import via ``backend.*`` paths, so we expose the repo root
on this package path to keep those imports working in both layouts.
"""

from __future__ import annotations

from pathlib import Path


_pkg_dir = Path(__file__).resolve().parent
_repo_root = _pkg_dir.parent

# Allow resolving backend.api, backend.main, etc. from the repo root.
__path__ = [str(_pkg_dir), str(_repo_root)]
