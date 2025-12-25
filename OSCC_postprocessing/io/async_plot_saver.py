from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Optional


class AsyncPlotSaver:
    """Simple thread-backed figure saver.

    submit(fig, path) queues a savefig call on a background thread and
    returns a Future. call wait() before program exit to ensure all
    saves finish.
    """

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="plot-save")
        self._futures: list[Future] = []

    def submit(self, fig, path: Path | str, *, dpi: Optional[int] = 150, bbox_inches: Optional[str] = "tight") -> Future:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _save():
            try:
                # Figures may be created with an interactive GUI backend (e.g. TkAgg).
                # Calling savefig from a background thread can then touch GUI state and
                # crash with "main thread is not in main loop". Rebind the figure to a
                # pure Agg canvas before saving to keep this thread-safe.
                try:
                    from matplotlib.backends.backend_agg import FigureCanvasAgg

                    FigureCanvasAgg(fig)
                    try:
                        fig.canvas.toolbar = None
                    except Exception:
                        pass
                except Exception:
                    # If matplotlib isn't available or rebinding fails, fall back to
                    # the original savefig behavior.
                    pass
                fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
            finally:
                # Close regardless of success to free memory
                try:
                    import matplotlib.pyplot as plt  # local import to avoid hard dependency
                    plt.close(fig)
                except Exception:
                    pass

        fut = self._executor.submit(_save)
        self._futures.append(fut)
        return fut

    def wait(self, timeout: Optional[float] = None) -> None:
        for fut in self._futures:
            fut.result(timeout=timeout)
        self._futures.clear()

    def shutdown(self, wait: bool = True) -> None:
        try:
            if wait:
                self.wait()
        finally:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
