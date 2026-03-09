from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


class TransactionError(RuntimeError):
    """Base transaction error."""


class TransactionConflictError(TransactionError):
    """Raised when optimistic MVCC validation fails."""


@dataclass(slots=True)
class _Version:
    ts: int
    value: Any
    deleted: bool = False


@dataclass(slots=True)
class _Tx:
    tx_id: int
    start_ts: int
    read_set: set[str] = field(default_factory=set)
    write_set: dict[str, Any] = field(default_factory=dict)
    delete_set: set[str] = field(default_factory=set)
    active: bool = True
    op_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class TransactionManager:
    """In-memory MVCC transaction manager with snapshot isolation.

    Design goals:
    - Snapshot reads are lock-free for keys (only a short metadata lock).
    - Commits are serialized through a single commit lock, removing
      circular wait patterns that cause classic deadlocks in key-lock managers.
    - First-committer-wins conflict handling for write-write conflicts.
    """

    def __init__(self) -> None:
        self._meta_lock = threading.RLock()
        self._commit_lock = threading.Lock()
        self._clock = 0
        self._next_tx_id = 1
        self._versions: dict[str, list[_Version]] = {}
        self._active_snapshots: dict[int, int] = {}

    def begin(self) -> _Tx:
        with self._meta_lock:
            tx = _Tx(tx_id=self._next_tx_id, start_ts=self._clock)
            self._next_tx_id += 1
            self._active_snapshots[tx.tx_id] = tx.start_ts
            return tx

    def read(self, tx: _Tx, key: str) -> Any | None:
        with tx.op_lock:
            self._assert_active(tx)

            if key in tx.delete_set:
                return None
            if key in tx.write_set:
                return tx.write_set[key]

            tx.read_set.add(key)

            with self._meta_lock:
                versions = self._versions.get(key, [])
                for version in reversed(versions):
                    if version.ts <= tx.start_ts:
                        return None if version.deleted else version.value
            return None

    def write(self, tx: _Tx, key: str, value: Any) -> None:
        with tx.op_lock:
            self._assert_active(tx)
            tx.delete_set.discard(key)
            tx.write_set[key] = value

    def delete(self, tx: _Tx, key: str) -> None:
        with tx.op_lock:
            self._assert_active(tx)
            tx.write_set.pop(key, None)
            tx.delete_set.add(key)

    def commit(self, tx: _Tx) -> int:
        with tx.op_lock:
            self._assert_active(tx)

            touched = set(tx.write_set) | tx.delete_set
            if not touched:
                tx.active = False
                with self._meta_lock:
                    self._active_snapshots.pop(tx.tx_id, None)
                return tx.start_ts

            with self._commit_lock:
                with self._meta_lock:
                    for key in touched:
                        latest_ts = self._latest_ts(key)
                        if latest_ts > tx.start_ts:
                            tx.active = False
                            self._active_snapshots.pop(tx.tx_id, None)
                            raise TransactionConflictError(
                                f"Write-write conflict on key '{key}'"
                            )

                    self._clock += 1
                    commit_ts = self._clock

                    for key, value in tx.write_set.items():
                        self._versions.setdefault(key, []).append(
                            _Version(ts=commit_ts, value=value, deleted=False)
                        )

                    for key in tx.delete_set:
                        self._versions.setdefault(key, []).append(
                            _Version(ts=commit_ts, value=None, deleted=True)
                        )

                    tx.active = False
                    self._active_snapshots.pop(tx.tx_id, None)
                    self._prune_versions_locked()
                    return commit_ts

    def rollback(self, tx: _Tx) -> None:
        with tx.op_lock:
            self._assert_active(tx)
            tx.write_set.clear()
            tx.delete_set.clear()
            tx.read_set.clear()
            tx.active = False

            with self._meta_lock:
                self._active_snapshots.pop(tx.tx_id, None)

    def _latest_ts(self, key: str) -> int:
        versions = self._versions.get(key, [])
        if not versions:
            return -1
        return versions[-1].ts

    def _prune_versions_locked(self) -> None:
        """Prune obsolete versions while preserving all active snapshots.

        Requires ``self._meta_lock``.
        """
        if not self._versions:
            return

        if self._active_snapshots:
            # The oldest active snapshot may still need the latest version at-or-before it.
            safe_ts = min(self._active_snapshots.values())
        else:
            # No active snapshots: only the latest version can still be observed.
            safe_ts = self._clock

        for key, versions in self._versions.items():
            if len(versions) <= 1:
                continue

            keep_from = len(versions) - 1
            for index in range(len(versions) - 1, -1, -1):
                if versions[index].ts <= safe_ts:
                    keep_from = index
                    break

            if keep_from > 0:
                del versions[:keep_from]

    @staticmethod
    def _assert_active(tx: _Tx) -> None:
        if not tx.active:
            raise TransactionError("Transaction is no longer active")
