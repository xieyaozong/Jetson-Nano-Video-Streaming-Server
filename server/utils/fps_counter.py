from __future__ import annotations

import time


class FpsCounter:
    def __init__(self) -> None:
        self._last = time.perf_counter()
        self.value = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        elapsed = max(now - self._last, 1e-6)
        self._last = now
        self.value = 1.0 / elapsed
        return self.value

