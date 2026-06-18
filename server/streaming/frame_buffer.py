from __future__ import annotations

from dataclasses import asdict, dataclass

import threading


@dataclass(frozen=True)
class StreamStats:
    sequence: int
    encoded_fps: float
    source_fps: float
    width: int
    height: int
    clients: int
    inference_enabled: bool
    detections: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FrameBuffer:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._stats = StreamStats(0, 0.0, 0.0, 0, 0, 0, False, 0)
        self._clients = 0

    def publish(self, jpeg: bytes, stats: StreamStats) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._stats = stats
            self._condition.notify_all()

    def wait_for_frame(self, last_sequence: int, timeout: float = 5.0) -> tuple[int, bytes] | None:
        with self._condition:
            ok = self._condition.wait_for(
                lambda: self._jpeg is not None and self._stats.sequence != last_sequence,
                timeout=timeout,
            )
            if not ok or self._jpeg is None:
                return None
            return self._stats.sequence, self._jpeg

    def snapshot(self) -> bytes | None:
        with self._condition:
            return self._jpeg

    def stats(self) -> StreamStats:
        with self._condition:
            return self._stats

    def add_client(self) -> None:
        with self._condition:
            self._clients += 1

    def remove_client(self) -> None:
        with self._condition:
            self._clients = max(0, self._clients - 1)

    def client_count(self) -> int:
        with self._condition:
            return self._clients
