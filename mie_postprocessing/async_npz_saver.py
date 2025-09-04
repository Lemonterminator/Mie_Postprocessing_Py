from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any
import numpy as np


class AsyncNPZSaver:
    """Background saver for compressed NPZ files.

    save(path, **arrays) queues a numpy savez_compressed call.
    Call wait() before program exit to ensure all saves finish.
    """

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="npz-save")
        self._futures: list[Future] = []

    def save(self, path: Path | str, /, **arrays: Any) -> Future:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _save():
            np.savez_compressed(path, **arrays)

        fut = self._executor.submit(_save)
        self._futures.append(fut)
        return fut

    def wait(self) -> None:
        for fut in self._futures:
            fut.result()
        self._futures.clear()

    def shutdown(self, wait: bool = True) -> None:
        try:
            if wait:
                self.wait()
        finally:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
