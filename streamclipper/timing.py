import time
from contextlib import contextmanager


class Timer:
    """Collects (stage, seconds, cached) rows and prints the per-run table."""

    def __init__(self):
        self.rows: list[tuple[str, float, bool]] = []

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        yield
        self.rows.append((name, time.perf_counter() - t0, False))

    def add_cached(self, name: str):
        self.rows.append((name, 0.0, True))

    @property
    def total(self) -> float:
        return sum(s for _, s, _ in self.rows)

    def table(self) -> str:
        lines = [f"{'stage':<12}{'seconds':>9}"]
        for name, secs, cached in self.rows:
            lines.append(f"{name:<12}{'cached' if cached else f'{secs:.2f}':>9}")
        lines.append(f"{'total':<12}{self.total:>9.2f}")
        return "\n".join(lines)
