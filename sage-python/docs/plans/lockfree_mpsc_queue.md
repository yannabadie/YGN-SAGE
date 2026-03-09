# Lock-Free Producer-Consumer Queue (CAS Design)

## Context

`EventBus.emit()` can be called from many producer threads while `stream()` consumers run on asyncio event loops. Directly mutating loop-owned queue internals from non-loop threads creates cross-thread race hazards.

The codebase now routes enqueue work through `loop.call_soon_threadsafe(...)` to keep asyncio objects thread-affine.

## CAS Alternative

For higher-throughput, lower-contention fan-out, use a bounded lock-free MPSC ring queue per async consumer:

- Multiple producers: any thread may enqueue.
- Single consumer: the event-loop thread dequeues.
- Coordination via atomic compare-and-swap (CAS), no mutex on hot path.

## Data Structure

```text
capacity = power_of_two
mask = capacity - 1

head: AtomicU64   # consumer-owned index
tail: AtomicU64   # producer-claimed index (CAS)

slot[i]:
  seq: AtomicU64  # sequence barrier
  val: T          # payload
```

Sequence numbers avoid ABA and identify whether a slot is:

- writable for position `p` when `slot.seq == p`
- readable for position `p` when `slot.seq == p + 1`

## Enqueue (producer)

```text
fn enqueue(x):
  loop:
    p = tail.load(acquire)
    s = slot[p & mask]
    seq = s.seq.load(acquire)
    diff = seq - p

    if diff == 0:
      if tail.compare_exchange_weak(p, p + 1, acq_rel, relaxed):
        break  # reserved slot
      continue

    if diff < 0:
      return FULL

    cpu_relax()  # another producer is progressing

  s.val = x
  s.seq.store(p + 1, release)  # publish
  return OK
```

## Dequeue (single consumer)

```text
fn dequeue():
  p = head.load(relaxed)
  s = slot[p & mask]
  seq = s.seq.load(acquire)
  diff = seq - (p + 1)

  if diff < 0:
    return EMPTY
  if diff > 0:
    return RETRY

  x = s.val
  s.seq.store(p + capacity, release)  # make slot writable next cycle
  head.store(p + 1, relaxed)
  return x
```

## Integration Sketch

- Keep one CAS MPSC queue per stream consumer.
- `emit()` performs lock-free enqueue into each consumer queue.
- Use an edge-triggered wakeup (`eventfd`/pipe/post) so consumer loop is notified only on empty->non-empty transitions.
- On consumer shutdown, mark queue as closed and drain safely.

## Notes

- Python stdlib does not expose native CAS primitives for this algorithm.
- Implement with Rust/C/C++ atomics behind a Python extension (PyO3/CFFI), then bind to `EventBus`.
- Keep queue bounded to avoid unbounded producer memory growth under slow consumers.
