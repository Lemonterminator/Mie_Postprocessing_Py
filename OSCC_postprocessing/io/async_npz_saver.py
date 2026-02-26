from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Optional
import numpy as np


class AsyncNPZSaver:
    """Background saver for compressed NPZ files.

    save(path, **arrays) queues a numpy savez_compressed call.
    Call wait() before program exit to ensure all saves finish.
    """

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="npz-save")
        self._futures: list[Future] = []

    @staticmethod
    def _quantize_to_u8_full_range(
        arr: Any,
        *,
        quant_float_upper_bound: float,
        quant_clip_negative: bool,
    ) -> tuple[np.ndarray, float, str]:
        a = np.asarray(arr)
        orig_dtype = str(a.dtype)
        if np.issubdtype(a.dtype, np.floating):
            upper = float(quant_float_upper_bound)
        elif np.issubdtype(a.dtype, np.integer):
            upper = float(np.iinfo(a.dtype).max)
        elif np.issubdtype(a.dtype, np.bool_):
            upper = 1.0
        else:
            # Fallback: cast unknown types to float and use configured bound.
            a = a.astype(np.float32, copy=False)
            upper = float(quant_float_upper_bound)

        if upper <= 0:
            raise ValueError(f"quantization upper bound must be > 0, got {upper}")

        lower = 0.0 if quant_clip_negative else None
        work = a.astype(np.float32, copy=False)
        work = np.clip(work, lower, upper)
        scaled = np.rint((work / upper) * 255.0)
        q = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
        return q, upper, orig_dtype

    def save(
        self,
        path: Path | str,
        /,
        *,
        quantize_u8: bool = False,
        quant_float_upper_bound: float = 1.0,
        quant_clip_negative: bool = True,
        quant_store_metadata: bool = True,
        quant_keys: Optional[list[str]] = None,
        **arrays: Any,
    ) -> Future:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _save():
            payload = dict(arrays)
            if quantize_u8:
                target_keys = list(payload.keys()) if quant_keys is None else [k for k in quant_keys if k in payload]
                for key in target_keys:
                    q, upper, orig_dtype = self._quantize_to_u8_full_range(
                        payload[key],
                        quant_float_upper_bound=quant_float_upper_bound,
                        quant_clip_negative=quant_clip_negative,
                    )
                    payload[key] = q
                    if quant_store_metadata:
                        payload[f"__quant__{key}__enabled"] = np.asarray(True)
                        payload[f"__quant__{key}__mode"] = np.asarray("uint8_full_range")
                        payload[f"__quant__{key}__upper"] = np.asarray(upper, dtype=np.float32)
                        payload[f"__quant__{key}__orig_dtype"] = np.asarray(orig_dtype)

            np.savez_compressed(path, **payload)

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
