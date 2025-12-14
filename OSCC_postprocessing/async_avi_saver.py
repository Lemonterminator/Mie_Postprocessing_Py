from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Optional
import numpy as np
import cv2


def _to_uint8_frames(
    video: np.ndarray,
    *,
    is_color: Optional[bool] = None,
    auto_normalize: bool = True,
) -> tuple[np.ndarray, bool]:
    """
    Normalize video to uint8 and correct shape.

    Returns
    -------
    video_u8 : np.ndarray
        (T, H, W) for gray, or (T, H, W, 3) for color
    is_color : bool
        True if output is BGR/color, False if gray
    """
    if video.ndim == 4 and video.shape[-1] == 1:
        # (T, H, W, 1) -> (T, H, W)
        video = video[..., 0]

    if video.ndim == 3:
        # (T, H, W) -> gray
        inferred_color = False
    elif video.ndim == 4 and video.shape[-1] == 3:
        # (T, H, W, 3) -> color
        inferred_color = True
    else:
        raise ValueError(
            f"Unsupported video shape {video.shape}; expected (T,H,W) or (T,H,W,3) or (T,H,W,1)"
        )

    if is_color is None:
        is_color = inferred_color

    # dtype handling
    if video.dtype == np.uint8:
        video_u8 = video
    else:
        if not auto_normalize:
            # Trust user: assume 0..1
            video_u8 = np.clip(video * 255.0, 0, 255).astype(np.uint8)
        else:
            vmin = float(video.min())
            vmax = float(video.max())
            if vmax <= vmin:
                video_u8 = np.zeros_like(video, dtype=np.uint8)
            else:
                video_u8 = ((video - vmin) * (255.0 / (vmax - vmin))).astype(np.uint8)

    # For color, OpenCV expects BGR. If you store RGB, convert here.
    # We *assume* user already has BGR or just grayscale. To keep it
    # general, we do not reorder channels here.
    return video_u8, is_color


def _write_avi_file(
    path: Path,
    video: np.ndarray,
    *,
    fps: int = 30,
    codec: str = "XVID",
    is_color: Optional[bool] = None,
    auto_normalize: bool = True,
) -> None:
    video_u8, is_color = _to_uint8_frames(
        video, is_color=is_color, auto_normalize=auto_normalize
    )

    if is_color:
        # (T, H, W, 3)
        T, H, W, _ = video_u8.shape
        frame_size = (W, H)
    else:
        # (T, H, W)
        T, H, W = video_u8.shape
        frame_size = (W, H)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size, isColor=is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    if is_color:
        for i in range(T):
            frame = video_u8[i]  # (H, W, 3), uint8
            writer.write(frame)
    else:
        for i in range(T):
            frame = video_u8[i]  # (H, W), uint8
            writer.write(frame)

    writer.release()


class AsyncAVISaver:
    """
    Background saver for .avi video files.

    save(path, video, fps=30, ...) queues an OpenCV write.
    Call wait() before program exit to ensure all writes finish.
    """

    def __init__(self, max_workers: int = 2, default_codec: str = "XVID"):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="avi-save"
        )
        self._futures: list[Future] = []
        self._default_codec = default_codec

    def save(
        self,
        path: Path | str,
        video: np.ndarray,
        /,
        *,
        fps: int = 30,
        codec: Optional[str] = None,
        is_color: Optional[bool] = None,
        auto_normalize: bool = True,
    ) -> Future:
        """
        Parameters
        ----------
        path : Path | str
            Output .avi file path.
        video : np.ndarray
            (T,H,W) gray or (T,H,W,3) color. Can be float or uint8.
        fps : int, default 30
            Video FPS.
        codec : str, optional
            OpenCV fourcc, e.g. 'XVID', 'MJPG', 'mp4v'. Defaults to class default.
        is_color : bool, optional
            Force color/gray. If None, infer from shape.
        auto_normalize : bool, default True
            If video not uint8, min-max to 0..255. If False, assume 0..1.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        codec = codec or self._default_codec

        def _task():
            _write_avi_file(
                path,
                video,
                fps=fps,
                codec=codec,
                is_color=is_color,
                auto_normalize=auto_normalize,
            )

        fut = self._executor.submit(_task)
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
