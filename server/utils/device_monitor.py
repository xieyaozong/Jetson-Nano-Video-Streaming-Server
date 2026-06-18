from __future__ import annotations


def system_snapshot() -> dict[str, object]:
    try:
        import psutil

        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
        }
    except Exception:
        return {}

