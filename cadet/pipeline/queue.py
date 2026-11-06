"""Inter-stage queues with a per-boundary freshness policy.

When two stages run at different rates (or on different nodes), the slower
consumer needs a policy for what to do with the backlog. A real-time autonomy
pipeline usually wants the latest frame, not a stale one; a measurement or
replay path wants every frame in order. Provides :class:`StageQueue`: a bounded
queue with policy ``latest_only`` (keep only the newest message) or ``fifo``
(preserve order, drop oldest when full), both counting drops. The builder places
a queue on each inter-stage boundary; the policy comes from the consuming stage's
``queue_policy`` in the config (``latest_only | fifo``; ``fifo`` honours ``maxsize``).
"""

from collections import deque

LATEST_ONLY = "latest_only"
FIFO = "fifo"
# Default bound for a fifo queue, so a stalled consumer cannot grow memory without
# limit; the oldest message is dropped when the bound is reached.
DEFAULT_FIFO_MAXSIZE = 64


class StageQueue:
    """A bounded message queue with a freshness policy.

    ``latest_only`` holds at most one message (a new put replaces the pending one);
    ``fifo`` preserves order up to ``maxsize`` and drops the oldest when full. In
    both cases :attr:`drops` counts messages discarded by the policy.
    """

    def __init__(self, policy: str = LATEST_ONLY, maxsize: int = DEFAULT_FIFO_MAXSIZE):
        if policy not in (LATEST_ONLY, FIFO):
            raise ValueError("unknown queue policy {!r}".format(policy))
        self.policy = policy
        self._maxsize = 1 if policy == LATEST_ONLY else max(1, maxsize)
        self._q: deque = deque()
        self.drops = 0

    def put(self, msg) -> None:
        """Add ``msg``, applying the policy; may discard an older message."""
        if self.policy == LATEST_ONLY:
            self.drops += len(self._q)  # whatever was pending is now stale
            self._q.clear()
            self._q.append(msg)
            return
        if len(self._q) >= self._maxsize:
            self._q.popleft()  # drop the oldest to make room
            self.drops += 1
        self._q.append(msg)

    def get(self):
        """Return the next message by policy, or ``None`` if empty.

        ``latest_only`` returns the single pending message; ``fifo`` returns the
        oldest.
        """
        if not self._q:
            return None
        return self._q.popleft()

    def __len__(self) -> int:
        return len(self._q)

    def empty(self) -> bool:
        return not self._q
