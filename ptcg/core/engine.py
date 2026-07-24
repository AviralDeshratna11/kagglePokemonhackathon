"""Robust loader for the cabt C++ engine (``libcg``).

Solves three concrete sources of Week-0 friction:

1. **Import pathing.** Inside the Kaggle submission container the working
   directory is not on ``sys.path``, so ``import cg`` fails even though ``cg/``
   sits next to ``main.py``. :func:`bootstrap_paths` inserts the correct
   directories before any engine import is attempted.

2. **Architecture.** ``kaggle-environments==1.30.1`` only ships ``libcg.so``
   (x86-64) and ``cg.dll``; importing it on an Apple-silicon host raises
   ``OSError``. Newer wheels add ``libcg.dylib`` / ``libcg-arm64.so``.
   :func:`describe_platform` reports exactly which binary will be used and why,
   so the failure mode is a readable message rather than a ctypes traceback.

3. **Missing bindings.** The shipped ``cg/game.py`` binds only five of the
   thirteen symbols the shared library actually exports. In particular
   ``AllCard`` and ``AllAttack`` -- the engine's own authoritative card and
   attack tables -- are unreachable through the public Python API.
   :class:`Engine` binds them, which lets the whole feature pipeline be derived
   from the engine itself instead of from a CSV that may drift out of sync.

Nothing in this module imports torch/numpy; it is safe to load inside the
latency-critical submission path.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "bootstrap_paths",
    "describe_platform",
    "Engine",
    "get_engine",
    "EngineUnavailable",
]


class EngineUnavailable(RuntimeError):
    """Raised when the native engine cannot be loaded on this host."""


# ---------------------------------------------------------------------------
# 1. Path bootstrap
# ---------------------------------------------------------------------------

def bootstrap_paths(extra: Iterable[os.PathLike[str] | str] = ()) -> list[str]:
    """Make ``cg`` importable from every location Kaggle might run us from.

    Returns the list of directories that were prepended to ``sys.path``.
    Safe to call repeatedly.
    """
    here = Path(__file__).resolve()
    candidates: list[Path] = [
        Path.cwd(),
        here.parent,             # ptcg/core
        here.parent.parent,      # ptcg
        here.parent.parent.parent,  # repo root / submission root
        Path("/kaggle_simulations/agent"),
        Path("/kaggle/working"),
    ]
    candidates.extend(Path(p) for p in extra)

    added: list[str] = []
    for p in candidates:
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)
            added.append(s)
    return added


# ---------------------------------------------------------------------------
# 2. Platform description
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlatformInfo:
    system: str
    machine: str
    expected_lib: str
    python: str

    @property
    def needs_amd64_container(self) -> bool:
        """True when the host cannot run the x86-64 ``libcg.so`` natively."""
        return self.machine in ("arm64", "aarch64") and self.system != "Windows"

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "machine": self.machine,
            "expected_lib": self.expected_lib,
            "python": self.python,
            "needs_amd64_container": self.needs_amd64_container,
        }


def describe_platform() -> PlatformInfo:
    system = platform.system()
    machine = platform.machine()
    if system == "Windows":
        lib = "cg.dll"
    elif system == "Darwin":
        lib = "libcg.dylib"
    elif machine in ("arm64", "aarch64"):
        lib = "libcg-arm64.so"
    else:
        lib = "libcg.so"
    return PlatformInfo(system, machine, lib, platform.python_version())


# ---------------------------------------------------------------------------
# 3. Engine handle
# ---------------------------------------------------------------------------

class _StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]


class _SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int),
    ]


#: Symbols the shared library is known to export. Anything the shipped
#: ``cg/game.py`` does not bind is listed as ``bound_by_upstream=False``.
KNOWN_SYMBOLS: dict[str, bool] = {
    "GameInitialize": True,
    "BattleStart": True,
    "BattleFinish": True,
    "GetBattleData": True,
    "Select": True,
    "VisualizeData": True,
    "AllCard": False,
    "AllAttack": False,
    "AgentStart": False,
    "SearchBegin": False,
    "SearchStep": False,
    "SearchEnd": False,
    "SearchRelease": False,
}


@dataclass
class Engine:
    """Thin, dependency-free wrapper over the cabt shared library."""

    lib: ctypes.CDLL
    lib_path: Path
    sha256: str
    platform_info: PlatformInfo
    exported: tuple[str, ...] = field(default_factory=tuple)

    # -- introspection ------------------------------------------------------

    def has(self, symbol: str) -> bool:
        return hasattr(self.lib, symbol)

    def fingerprint(self) -> dict[str, Any]:
        """Everything needed to detect an engine swap between runs.

        The competition organisers can ship a new ``libcg`` at any time; a
        silent binary change would invalidate every offline measurement we
        report. Recording the hash makes that detectable.
        """
        return {
            "lib_path": str(self.lib_path),
            "sha256": self.sha256,
            "platform": self.platform_info.as_dict(),
            "exported": list(self.exported),
            "unbound_upstream": [
                s for s, bound in KNOWN_SYMBOLS.items() if not bound and self.has(s)
            ],
        }

    # -- card / attack tables ----------------------------------------------

    def all_card_data(self) -> list[dict[str, Any]]:
        """Engine-native card table (``AllCard``).

        This is the ground truth for card IDs, types, HP, weakness, evolution
        links and attack IDs. Unlike ``EN_Card_Data.csv`` it can never drift
        out of sync with the simulator we are actually playing in.
        """
        fn = self.lib.AllCard
        fn.restype = ctypes.c_char_p
        fn.argtypes = []
        return json.loads(fn().decode("utf-8"))

    def all_attack(self) -> list[dict[str, Any]]:
        """Engine-native attack table (``AllAttack``): id, name, text, damage, energies."""
        fn = self.lib.AllAttack
        fn.restype = ctypes.c_char_p
        fn.argtypes = []
        return json.loads(fn().decode("utf-8"))


_ENGINE: Engine | None = None

#: Module paths that, when already imported, have *already* run
#: ``GameInitialize`` on the shared library.
_CG_SIM_MODULES = ("cg.sim", "kaggle_environments.envs.cabt.cg.sim")


def _candidate_lib_paths() -> list[Path]:
    info = describe_platform()
    roots: list[Path] = []

    # Prefer a cg/ shipped alongside the submission.
    for p in sys.path:
        try:
            cand = Path(p) / "cg"
        except (TypeError, ValueError):
            continue
        if cand.is_dir():
            roots.append(cand)

    # Fall back to the copy inside kaggle-environments.
    try:
        import kaggle_environments  # noqa: F401

        roots.append(
            Path(kaggle_environments.__file__).parent / "envs" / "cabt" / "cg"
        )
    except Exception:  # pragma: no cover - kaggle_environments may be absent
        pass

    names = [info.expected_lib, "libcg.so", "libcg-arm64.so", "libcg.dylib", "cg.dll"]
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for name in names:
            p = root / name
            if p.exists() and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _bind(lib: ctypes.CDLL) -> None:
    """Attach the ctypes signatures. Idempotent."""
    lib.BattleStart.restype = _StartData
    lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.BattleFinish.argtypes = [ctypes.c_void_p]
    lib.GetBattleData.restype = _SerialData
    lib.GetBattleData.argtypes = [ctypes.c_void_p]
    lib.Select.restype = ctypes.c_int
    lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.VisualizeData.restype = ctypes.c_char_p
    lib.VisualizeData.argtypes = [ctypes.c_void_p]


def _adopt_initialised_lib() -> tuple[ctypes.CDLL, Path] | None:
    """Reuse a ``libcg`` handle that some other module has already set up.

    This matters more than it looks. ``dlopen`` is refcounted per *path*, so
    ``ctypes.cdll.LoadLibrary`` on an already-loaded ``libcg.so`` hands back the
    *same* underlying library -- and calling ``GameInitialize`` a second time on
    it aborts the process with SIGABRT.

    Both ``cg.sim`` (shipped inside our submission bundle) and
    ``kaggle_environments.envs.cabt.cg.sim`` (used by the local harness) call
    ``GameInitialize`` at import time. In any session that touches both -- which
    is every local self-play run -- a naive loader crashes on the second one.
    So: if either module is already imported, adopt its handle instead of
    opening our own.
    """
    for name in _CG_SIM_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        lib = getattr(mod, "lib", None)
        if lib is None:
            continue
        path = Path(getattr(mod, "lib_path", "") or "")
        if not path.exists():
            path = Path(getattr(mod, "__file__", "."))
        return lib, path

    # Not imported yet -- import one, which initialises exactly once.
    for name in _CG_SIM_MODULES:
        try:
            __import__(name)
        except Exception:  # noqa: BLE001
            continue
        mod = sys.modules.get(name)
        lib = getattr(mod, "lib", None) if mod else None
        if lib is not None:
            path = Path(getattr(mod, "lib_path", "") or getattr(mod, "__file__", "."))
            return lib, path
    return None


def get_engine(force_reload: bool = False) -> Engine:
    """Load (once) and return the engine handle.

    ``GameInitialize`` is guaranteed to run exactly once per process.
    """
    global _ENGINE
    if _ENGINE is not None and not force_reload:
        return _ENGINE

    bootstrap_paths()
    info = describe_platform()

    adopted = _adopt_initialised_lib()
    if adopted is not None:
        lib, path = adopted
        _bind(lib)
        digest = ""
        if path.is_file() and path.suffix in (".so", ".dylib", ".dll"):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        exported = tuple(s for s in KNOWN_SYMBOLS if hasattr(lib, s))
        _ENGINE = Engine(lib, path, digest, info, exported)
        return _ENGINE

    candidates = _candidate_lib_paths()
    if not candidates:
        raise EngineUnavailable(
            "No cabt shared library found. Expected a 'cg/' directory containing "
            f"{info.expected_lib!r} on sys.path, or kaggle-environments installed.\n"
            f"Platform: {info.as_dict()}"
        )

    errors: list[str] = []
    for path in candidates:
        try:
            lib = ctypes.cdll.LoadLibrary(str(path))
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue

        lib.GameInitialize()
        _bind(lib)

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        exported = tuple(s for s in KNOWN_SYMBOLS if hasattr(lib, s))
        _ENGINE = Engine(lib, path, digest, info, exported)
        return _ENGINE

    hint = ""
    if info.needs_amd64_container:
        hint = (
            "\nHint: this host is arm64. Either upgrade kaggle-environments "
            "(>=1.31 ships libcg-arm64.so / libcg.dylib) or run inside the "
            "linux/amd64 container: `make docker-build && make docker-shell`."
        )
    raise EngineUnavailable("Could not load any cabt library.\n" + "\n".join(errors) + hint)
