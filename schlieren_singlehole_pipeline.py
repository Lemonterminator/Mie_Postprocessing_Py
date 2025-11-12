import numpy as np
from mie_postprocessing.video_filters import gaussian_video_cpu, median_filter_video_auto
from mie_postprocessing.async_npz_saver import AsyncNPZSaver
from mie_postprocessing.async_avi_saver import *
from mie_postprocessing.svd_background_removal import godec_like
from mie_postprocessing.video_filters import *
from mie_postprocessing.video_playback import *
# Prefer GPU if CuPy is available; otherwise use a NumPy-compatible shim.

try:
    import cupy as _cupy  # type: ignore

    _cupy.cuda.runtime.getDeviceCount()
    cp = _cupy
    USING_CUPY = True
except Exception as exc:  # pragma: no cover - hardware dependent
    print(f"CuPy unavailable, falling back to NumPy backend: {exc}")
    USING_CUPY = False

    class _NumpyCompat:
        def __getattr__(self, name):
            return getattr(np, name)

        def asarray(self, a, dtype=None):
            return np.asarray(a, dtype=dtype)

        def asnumpy(self, a):
            return np.asarray(a)

        def get(self, a):
            return a

    cp = _NumpyCompat()  # type: ignore

def to_numpy(arr):
    return cp.asnumpy(arr) if USING_CUPY else np.asarray(arr)

# Import rotation utility based on backend availability to avoid hard Cupy dependency
if USING_CUPY:
    from mie_postprocessing.rotate_with_alignment import (
        rotate_video_nozzle_at_0_half_cupy as rotate_video_nozzle_at_0_half_backend,
    )
else:
    from mie_postprocessing.rotate_with_alignment_cpu import (
        rotate_video_nozzle_at_0_half_numpy as rotate_video_nozzle_at_0_half_backend,
    )

def _rotate_align_video_cpu(
    video: np.ndarray,
    nozzle_center: tuple[float, float],
    offset_deg: float,
    *,
    interpolation: str,
    out_shape: tuple[int, int] | None,
    border_mode: str,
    cval: float,
) -> np.ndarray:
    """
    Delegate to the NumPy implementation in mie_postprocessing.rotate_with_alignment_cpu.
    Returns only the rotated video (np.ndarray).
    """
    rotated_np, _, _ = rotate_video_nozzle_at_0_half_backend(
        video,
        nozzle_center,
        offset_deg,
        interpolation=interpolation,
        border_mode=border_mode,
        out_shape=out_shape,
        cval=cval,
    )
    return rotated_np.astype(np.float32, copy=False)

def _min_max_scale(arr: cp.ndarray) -> cp.ndarray:
    mn = arr.min()
    mx = arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return cp.zeros_like(arr)

def _prepare_temporal_smoothing(rotated: cp.ndarray, smooth_frames: int) -> tuple[cp.ndarray, np.ndarray]:
    rotated_cpu = to_numpy(rotated)
    smoothed_np = median_filter_video_auto(np.swapaxes(rotated_cpu, 0, 2), smooth_frames, 1)
    smoothed_np = np.swapaxes(smoothed_np, 0, 2)

    min_val = smoothed_np.min()
    max_val = smoothed_np.max()
    if max_val > min_val:
        smoothed_np = (smoothed_np - min_val) / (max_val - min_val)
    else:
        smoothed_np = np.zeros_like(smoothed_np, dtype=np.float32)

    smoothed_cp = cp.asarray(smoothed_np, dtype=cp.float32)
    return smoothed_cp, smoothed_np

#  Main pipeline function
def schlieren_singlehole_pipeline(video, offset, centre, file_name, 
                                  rotated_folder_SCH, data_SCH, 
                                  save_intermediate_results=True,
                                  FPS=20
                                  ):
    F, H, W = video.shape
    shock_wave_duration = 50
    # Rotate and align the video based on nozzle centre and offset
    INTERPOLATION = "nearest"
    BORDER_MODE = "constant"
    OUT_SHAPE = (H // 2, W)
    if USING_CUPY:
        try:
            rotated_gpu, _, _ = rotate_video_nozzle_at_0_half_backend(
                video,
                centre,
                offset,
                interpolation=INTERPOLATION,
                border_mode=BORDER_MODE,
                out_shape=OUT_SHAPE,
            )
            rotated = cp.asarray(rotated_gpu, dtype=cp.float32)
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"GPU rotation failed ({exc}), falling back to CPU numpy implementation.")
            rotated_np = _rotate_align_video_cpu(
                video,
                centre,
                offset,
                interpolation=INTERPOLATION,
                border_mode=BORDER_MODE,
                out_shape=OUT_SHAPE,
                cval=0.0,
            )
            rotated = cp.asarray(rotated_np, dtype=cp.float32)
    else:
        rotated_np = _rotate_align_video_cpu(
            video,
            centre,
            -45.0,
            interpolation="bicubic",
            border_mode="constant",
            out_shape=OUT_SHAPE,
            cval=0.0,
        )
        rotated = cp.asarray(rotated_np, dtype=cp.float32)


    # Clip the negative number in the rotated video
    rotated = cp.clip(rotated, 0.0, None)

    # Normalize the rotated video to [0, 1]
    rotated = _min_max_scale(rotated)

    avi_saver = AsyncAVISaver(max_workers=4)
    if save_intermediate_results:
        # np.savez(data_SCH / f"{file_name}_rotated.npy", to_numpy(rotated))
        AsyncNPZSaver().save(rotated_folder_SCH / f"{file_name}_rotated.npz", rotated=to_numpy(rotated))
        
        f1 = avi_saver.save(
            rotated_folder_SCH / f"{file_name}_rotated.avi",
            to_numpy(rotated),
            fps=FPS,
            is_color=False,
            auto_normalize=True,
        )



    smooth_frames = 3
    temporal_smoothing, temporal_smoothing_np = _prepare_temporal_smoothing(rotated, smooth_frames)
    
    inverted = 1.0 - temporal_smoothing_np

    foreground_godec = godec_like(inverted, 3)
    godec_pos = np.maximum(foreground_godec, 0.0)
    godec_pos = _min_max_scale(cp.asarray(godec_pos, dtype=cp.float32))

    # --- usage ---
    # video: (T, H, W), grayscale
    sobel_x, sobel_y = sobel_5x5_kernels()

    # gradient in x for every frame

    gx_video = filter_video_fft(godec_pos[:shock_wave_duration], sobel_x, mode='same')

    # gradient in y for every frame
    gy_video = filter_video_fft(godec_pos[:shock_wave_duration], sobel_y, mode='same')

    # gradient magnitude (still (T, H, W))
    grad_mag = np.sqrt(gx_video**2 + gy_video**2)

    gamma = 1.1
    grad_mag_norm_inv = 1 - _min_max_scale(grad_mag) ** gamma
    
    thres = 0.9
    grad_mag_norm_inv[grad_mag_norm_inv < thres] = 0
    grad_mag_norm_inv[grad_mag_norm_inv > thres] = 1

    shock_wave_filtered = np.zeros_like(godec_pos)
    shock_wave_filtered[:shock_wave_duration] = _min_max_scale(
        grad_mag_norm_inv * godec_pos[:shock_wave_duration]
        )
    shock_wave_filtered[shock_wave_duration:] = godec_pos[shock_wave_duration:]

    play_videos_side_by_side((
        cp.swapaxes(godec_pos, 1,2),
        _min_max_scale(cp.swapaxes(shock_wave_filtered, 1, 2))
    ), intv=100)
    # play_video_cv2(godec_pos, intv=6000//FPS)
    '''
    play_videos_side_by_side((
        cp.swapaxes(godec_pos, 1, 2),
        _min_max_scale(cp.swapaxes(np.abs(gx_video), 1, 2)),
        # _min_max_scale(cp.swapaxes(gy_video, 1, 2)),
        # _min_max_scale(cp.swapaxes(grad_mag, 1, 2)),
        _min_max_scale(cp.swapaxes((1-np.sqrt(gx_video**2)), 1, 2)),
        _min_max_scale(cp.swapaxes((1-np.sqrt(gx_video**2))*godec_pos*inverted, 1, 2)),
        cp.swapaxes(inverted, 1, 2)
    ), intv= 600//FPS
    )
    '''
    
    

    avi_saver.wait()
    avi_saver.shutdown()
    # playing to check 

        