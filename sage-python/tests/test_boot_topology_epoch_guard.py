import atexit
import logging
from pathlib import Path
from typing import Any

import pytest

import sage.boot_topology as boot_topology
from sage.posterior_epoch import A14_BYPASS_ENV, A14_EPOCH_GUARD_ERROR_PREFIX


class _StubBandit:
    def __init__(self, *_args: object) -> None:
        pass


class _StubEngine:
    def __init__(self, load_exc: BaseException | None = None) -> None:
        self.load_exc = load_exc
        self.load_calls: list[str] = []
        self.save_calls: list[str] = []

    def load_state(self, state_dir: str) -> tuple[int, int]:
        self.load_calls.append(state_dir)
        if self.load_exc is not None:
            raise self.load_exc
        return (0, 0)

    def save_state(self, state_dir: str) -> None:
        self.save_calls.append(state_dir)

    def smmu_chunk_count(self) -> int:
        return 1


def _install_boot_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    engine: _StubEngine,
) -> None:
    monkeypatch.setattr(boot_topology, "_HAS_RUST_ROUTER", True)
    monkeypatch.setattr(boot_topology, "_find_cards_toml", lambda: None)
    monkeypatch.setattr(boot_topology, "RustTopologyEngine", lambda: engine, raising=False)
    monkeypatch.setattr(boot_topology, "RustBandit", _StubBandit, raising=False)
    monkeypatch.setattr(boot_topology.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv(A14_BYPASS_ENV, raising=False)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("legacy-state", encoding="utf-8")


def _init_with_engine(engine: _StubEngine) -> dict[str, Any]:
    return boot_topology.init_topology(rust_registry=None, metacognition=None)


def test_boot_re_raises_contamination_signature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / ".sage"
    _touch(state_dir / "bandit_state.db")
    engine = _StubEngine()
    _install_boot_stubs(monkeypatch, tmp_path, engine)

    with pytest.raises(RuntimeError) as exc_info:
        _init_with_engine(engine)

    assert str(exc_info.value).startswith(A14_EPOCH_GUARD_ERROR_PREFIX)
    assert engine.load_calls == []


def test_boot_swallows_other_rust_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = _StubEngine(RuntimeError("ordinary sqlite load failure"))
    _install_boot_stubs(monkeypatch, tmp_path, engine)

    with caplog.at_level(logging.DEBUG, logger="sage.boot"):
        result = _init_with_engine(engine)

    assert result["topology_engine"] is engine
    assert "Boot: No persisted state loaded" in caplog.text


def test_boot_re_raises_pyioerror_contamination_signature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = _StubEngine(IOError("contaminated_pre_a14_state: rust guard"))
    _install_boot_stubs(monkeypatch, tmp_path, engine)

    with pytest.raises(IOError) as exc_info:
        _init_with_engine(engine)

    assert str(exc_info.value).startswith(A14_EPOCH_GUARD_ERROR_PREFIX)


def test_boot_re_raises_oserror_contamination_signature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = _StubEngine(OSError("contaminated_pre_a14_state: rust guard"))
    _install_boot_stubs(monkeypatch, tmp_path, engine)

    with pytest.raises(OSError) as exc_info:
        _init_with_engine(engine)

    assert str(exc_info.value).startswith(A14_EPOCH_GUARD_ERROR_PREFIX)


def test_boot_bypass_logs_warning_and_disables_atexit_save(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    state_dir = tmp_path / ".sage"
    _touch(state_dir / "archive_state.db")
    engine = _StubEngine()
    registered: list[object] = []
    _install_boot_stubs(monkeypatch, tmp_path, engine)
    monkeypatch.setenv(A14_BYPASS_ENV, "1")
    monkeypatch.setattr(atexit, "register", registered.append)

    with caplog.at_level(logging.WARNING, logger="sage.posterior_epoch"):
        with caplog.at_level(logging.WARNING, logger="sage.boot"):
            result = _init_with_engine(engine)

    assert result["topology_engine"] is engine
    assert len(engine.load_calls) == 1
    assert registered == []
    assert "a14_epoch_guard_bypass layer=python" in caplog.text
    assert "a14_epoch_guard_bypass_save_disabled" in caplog.text
