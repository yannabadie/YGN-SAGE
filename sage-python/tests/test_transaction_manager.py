from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from sage.memory.transaction_manager import TransactionConflictError, TransactionManager


def test_snapshot_read_is_stable() -> None:
    tm = TransactionManager()

    tx0 = tm.begin()
    tm.write(tx0, "k", "v1")
    tm.commit(tx0)

    tx_reader = tm.begin()
    assert tm.read(tx_reader, "k") == "v1"

    tx_writer = tm.begin()
    tm.write(tx_writer, "k", "v2")
    tm.commit(tx_writer)

    # Reader still sees its original snapshot.
    assert tm.read(tx_reader, "k") == "v1"

    tx_fresh = tm.begin()
    assert tm.read(tx_fresh, "k") == "v2"


def test_first_committer_wins_on_same_key() -> None:
    tm = TransactionManager()

    tx1 = tm.begin()
    tx2 = tm.begin()

    tm.write(tx1, "account", 100)
    tm.write(tx2, "account", 200)

    tm.commit(tx1)
    with pytest.raises(TransactionConflictError):
        tm.commit(tx2)


def test_cross_key_writes_do_not_deadlock() -> None:
    """This pattern deadlocks under naive per-key lock acquisition."""
    tm = TransactionManager()

    seed = tm.begin()
    tm.write(seed, "A", 1)
    tm.write(seed, "B", 1)
    tm.commit(seed)

    def tx_ab() -> int:
        tx = tm.begin()
        a = tm.read(tx, "A") or 0
        b = tm.read(tx, "B") or 0
        tm.write(tx, "A", a + 1)
        tm.write(tx, "B", b + 1)
        return tm.commit(tx)

    def tx_ba() -> int:
        tx = tm.begin()
        b = tm.read(tx, "B") or 0
        a = tm.read(tx, "A") or 0
        tm.write(tx, "B", b + 1)
        tm.write(tx, "A", a + 1)
        return tm.commit(tx)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut1 = pool.submit(tx_ab)
        fut2 = pool.submit(tx_ba)

        results = []
        errors = []
        for fut in (fut1, fut2):
            try:
                results.append(fut.result(timeout=2.0))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    # No deadlock: both futures complete quickly.
    assert len(results) + len(errors) == 2
    # At most one commit may fail due to write-write conflict, not blocking.
    assert all(isinstance(e, TransactionConflictError) for e in errors)


def test_concurrent_commit_on_same_tx_is_rejected() -> None:
    tm = TransactionManager()
    tx = tm.begin()
    tm.write(tx, "k", "v1")

    def commit_once() -> int:
        return tm.commit(tx)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut1 = pool.submit(commit_once)
        fut2 = pool.submit(commit_once)

        results = []
        errors = []
        for fut in (fut1, fut2):
            try:
                results.append(fut.result(timeout=2.0))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    assert len(results) == 1
    assert len(errors) == 1
    assert errors and str(errors[0]) == "Transaction is no longer active"


def test_delete_is_versioned_and_snapshot_stable() -> None:
    tm = TransactionManager()

    seed = tm.begin()
    tm.write(seed, "k", "v1")
    tm.commit(seed)

    tx_reader = tm.begin()
    assert tm.read(tx_reader, "k") == "v1"

    tx_delete = tm.begin()
    tm.delete(tx_delete, "k")
    tm.commit(tx_delete)

    assert tm.read(tx_reader, "k") == "v1"
    assert tm.read(tm.begin(), "k") is None


def test_read_only_commit_keeps_clock_stable() -> None:
    tm = TransactionManager()

    tx = tm.begin()
    commit_ts = tm.commit(tx)

    assert commit_ts == tx.start_ts == 0

    writer = tm.begin()
    tm.write(writer, "k", "v1")
    writer_commit_ts = tm.commit(writer)
    assert writer_commit_ts == 1
