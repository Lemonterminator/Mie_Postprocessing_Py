import numpy as np
import cv2
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

CropRect = Tuple[int, int, int, int]  # x, y, w, h
Coordinates = Tuple[float, float]  # x, y

@lru_cache(maxsize=8)
def make_rotation_maps(
    frame_size: Tuple[int, int],
    angle: float,
    crop_rect: Optional[CropRect] = None,
    # rotation_center: Optional[Coordinates] = None
    rotation_center = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute map_x and map_y for cv2.remap."""
    h, w = frame_size
    if crop_rect is None:
        x0, y0, out_w, out_h = 0, 0, w, h
    else:
        x0, y0, out_w, out_h = crop_rect

    j_coords, i_coords = np.meshgrid(np.arange(out_w), np.arange(out_h))
    if rotation_center is None:
        cx, cy = w / 2.0, h / 2.0
    else: 
        cx, cy = rotation_center
    abs_x = j_coords + x0
    abs_y = i_coords + y0

    theta = np.deg2rad(angle)
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    x_rel = abs_x - cx
    y_rel = abs_y - cy
    src_x = cos_a * x_rel + sin_a * y_rel + cx
    src_y = -sin_a * x_rel + cos_a * y_rel + cy

    return src_x.astype(np.float32), src_y.astype(np.float32)

def _remap_frame(frame: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, use_cuda: bool) -> np.ndarray:
    if frame.dtype == np.bool_:
        tmp = frame.astype(np.uint8) * 255
        if use_cuda:
            try:
                gpu_frame = cv2.cuda_GpuMat() # type: ignore
                gpu_frame.upload(tmp)
                gpu_map_x = cv2.cuda_GpuMat(); gpu_map_x.upload(map_x) # type: ignore
                gpu_map_y = cv2.cuda_GpuMat(); gpu_map_y.upload(map_y) # type: ignore
                result = cv2.cuda.remap(gpu_frame, gpu_map_x, gpu_map_y, cv2.INTER_NEAREST)
                remapped = result.download()
            except Exception:
                remapped = cv2.remap(tmp, map_x, map_y, interpolation=cv2.INTER_NEAREST)
        else:
            remapped = cv2.remap(tmp, map_x, map_y, interpolation=cv2.INTER_NEAREST)
        return remapped > 127
    else:
        if use_cuda:
            try:
                gpu_frame = cv2.cuda_GpuMat() # type: ignore
                gpu_frame.upload(frame)
                gpu_map_x = cv2.cuda_GpuMat(); gpu_map_x.upload(map_x) # type: ignore
                gpu_map_y = cv2.cuda_GpuMat(); gpu_map_y.upload(map_y) # type: ignore
                result = cv2.cuda.remap(gpu_frame, gpu_map_x, gpu_map_y, cv2.INTER_CUBIC)
                return result.download()
            except Exception:
                return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_CUBIC)
        else:
            return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_CUBIC)


def _remap_video_cuda(video: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, interpolation=cv2.INTER_CUBIC) -> np.ndarray:
    """Remap a stack of frames on the GPU without threading."""
    gpu_map_x = cv2.cuda_GpuMat(); gpu_map_x.upload(map_x)
    gpu_map_y = cv2.cuda_GpuMat(); gpu_map_y.upload(map_y)
    gpu_frame = cv2.cuda_GpuMat()
    out_frames = []
    for idx in range(video.shape[0]):
        gpu_frame.upload(video[idx])
        result = cv2.cuda.remap(gpu_frame, gpu_map_x, gpu_map_y, interpolation)
        out_frames.append(result.download())
    return np.stack(out_frames, axis=0)

def rotate_and_crop(
    array: np.ndarray,
    angle: float,
    crop_rect: Optional[CropRect] = None,
    # rotation_center: Optional[Coordinates] = None,
    rotation_center = None,
    is_video: bool = False,
    mask: Optional[np.ndarray] = None,
    max_workers: Optional[int] = None
) -> np.ndarray:
    """Rotate an image or video by ``angle`` and optionally crop the result.

    Parameters
    ----------
    array : np.ndarray
        Input image or video. If ``is_video`` is True this should be a
        3-D array ``(frames, height, width)``.
    angle : float
        Rotation angle in degrees.
    crop_rect : tuple, optional
        ``(x, y, w, h)`` rectangle describing the cropped region after
        rotation.
    rotation_center : tuple, optional
        ``(x, y)`` coordinates of the center of rotation.
    is_video : bool
        Whether ``array`` represents a video sequence.
    mask : np.ndarray, optional
        Boolean mask in the cropped coordinate system. Pixels outside the
        mask are ignored during remapping.
    max_workers : int, optional
        Number of worker threads for video processing.
    """
    if is_video:
        h, w = array.shape[1:3]
    else:
        h, w = array.shape[:2]

    
    map_x, map_y = make_rotation_maps((h, w), angle, crop_rect, rotation_center)

    if mask is not None:
        if mask.shape != map_x.shape:
            raise ValueError("Mask size does not match output dimensions")
        mask_bool = mask.astype(bool)
        map_x = map_x.copy()
        map_y = map_y.copy()
        map_x[~mask_bool] = -1
        map_y[~mask_bool] = -1

    use_cuda = False
    try:
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            use_cuda = True
    except Exception:
        use_cuda = False

    if is_video:
        num_frames = array.shape[0]
        if use_cuda:
            return _remap_video_cuda(array, map_x, map_y)
        else:
            rotated = [None] * num_frames

            def task(idx: int):
                return idx, _remap_frame(array[idx], map_x, map_y, use_cuda)

            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = [exe.submit(task, i) for i in range(num_frames)]
                for fut in as_completed(futures):
                    idx, out = fut.result()
                    rotated[idx] = out
            return np.stack(rotated, axis=0)
    else:
        return _remap_frame(array, map_x, map_y, use_cuda)
    
def generate_CropRect(inner_radius, outer_radius, number_of_plumes, centre_x, centre_y):
    section_angle = 360.0/ number_of_plumes
    half_angle_radian = section_angle / 2.0 * np.pi/180.0
    half_width = round(outer_radius*np.sin(half_angle_radian))

    x = round(centre_x + inner_radius)

    y = max(0, round(centre_y - half_width))

    w = round(outer_radius - inner_radius)

    h = 2*half_width

    return (x, y, w, h)

def generate_plume_mask(inner_radius, outer_radius, w, h):
    y1 = -h/outer_radius/2 * inner_radius + h/2
    y2 = h/outer_radius/2 * inner_radius + h/2
    
    # Create blank single-channel mask of same height/width
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define polygon vertices as Nx2 integer array  
    pts = np.array([[0, round(y2)], [0, round(y1)], [w, 0], [w, h]], dtype=np.int32)
    
    # Fill the polygon on the mask
    cv2.fillPoly(mask, [pts], (255,))

    # cv2.imshow("plume_mask", mask) # Debug

    # Apply mask to extract polygon region
    return mask >0 
