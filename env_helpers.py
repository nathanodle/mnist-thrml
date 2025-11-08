"""Utilities to make runtime configuration portable."""

from __future__ import annotations

import ctypes
import os
import site
from pathlib import Path
from typing import Iterable


def _candidate_lib_dirs() -> list[str]:
    """Return site-packages subdirectories that may contain CUDA .so files."""
    site_paths: list[str] = []
    if hasattr(site, "getsitepackages"):
        site_paths.extend(site.getsitepackages())
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        site_paths.append(user_site)

    lib_dirs: list[str] = []
    for base in site_paths:
        if not base:
            continue
        nvidia_dir = Path(base) / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for pkg_dir in nvidia_dir.iterdir():
            lib_path = pkg_dir / "lib"
            if lib_path.is_dir():
                lib_dirs.append(str(lib_path))
    return lib_dirs


def _ensure_paths_in_env(env_var: str, paths: Iterable[str]) -> bool:
    """Inject paths into an os.environ list-like variable if missing."""
    existing_raw = os.environ.get(env_var, "")
    existing = [p for p in existing_raw.split(os.pathsep) if p]
    changed = False
    for path in paths:
        if path not in existing:
            existing.insert(0, path)
            changed = True
    if changed:
        os.environ[env_var] = os.pathsep.join(existing)
    return changed


def ensure_cuda_shared_libs_visible() -> None:
    """Best-effort attempt to make pip-installed CUDA libs visible without user env setup."""

    try:
        ctypes.CDLL("libcusparse.so")
        return
    except OSError:
        pass

    candidates = []
    for lib_dir in _candidate_lib_dirs():
        path_obj = Path(lib_dir)
        if any(path_obj.glob("libcusparse.so*")):
            candidates.append(lib_dir)

    if not candidates:
        return

    _ensure_paths_in_env("LD_LIBRARY_PATH", candidates)

    try:
        ctypes.CDLL("libcusparse.so")
    except OSError:
        # Nothing else to do; leave the environment as-is so advanced users can debug.
        pass
