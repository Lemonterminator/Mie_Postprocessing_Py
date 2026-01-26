import numpy as np
import cv2
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
import os
from concurrent.futures import as_completed, ThreadPoolExecutor, ProcessPoolExecutor

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

def _remap_frame_cupy(frame, map_x, map_y, interpolation=cv2.INTER_CUBIC):
    """Remap a single CuPy frame using ``cupyx.scipy.ndimage.map_coordinates``."""
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates

    order_map = {cv2.INTER_NEAREST: 0, cv2.INTER_LINEAR: 1, cv2.INTER_CUBIC: 3}
    order = order_map.get(interpolation, 1)

    coords = cp.stack((map_y, map_x))
    return map_coordinates(frame, coords, order=order, mode="constant", cval=0.0)

def _remap_video_cupy(
    video, map_x, map_y, interpolation=cv2.INTER_CUBIC, max_workers: Optional[int] = 1
):
    """Remap a stack of CuPy frames entirely on the GPU.

    Parameters
    ----------
    video : cp.ndarray
        Input stack of frames on the GPU.
    map_x, map_y : array-like
        Precomputed coordinate maps.
    interpolation : int
        OpenCV interpolation flag.
    max_workers : int, optional
        Number of worker threads. Defaults to ``1`` to avoid CPU
        oversubscription when this function is invoked from an outer
        thread pool.
    """
    import cupy as cp

    map_x_cp = cp.asarray(map_x)
    map_y_cp = cp.asarray(map_y)
    num_frames = video.shape[0]
    rotated = [None] * num_frames

    if max_workers in (None, 1):
        for idx in range(num_frames):
            rotated[idx] = _remap_frame_cupy(
                video[idx], map_x_cp, map_y_cp, interpolation
            )
    else:
        def task(idx: int):
            return idx, _remap_frame_cupy(video[idx], map_x_cp, map_y_cp, interpolation)

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(task, i) for i in range(num_frames)]
            for fut in as_completed(futures):
                idx, out = fut.result()
                rotated[idx] = out
    return cp.stack(rotated, axis=0)
    
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

    cp = None
    is_cupy = False
    try:
        import cupy as cp  # type: ignore
        is_cupy = isinstance(array, cp.ndarray)
    except Exception:
        cp = None

    if is_cupy:
        map_x = cp.asarray(map_x)   # type:ignore
        map_y = cp.asarray(map_y)   # type:ignore

    use_cuda = False
    if not is_cupy:
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                use_cuda = True
        except Exception:
            use_cuda = False

    if is_video:
        num_frames = array.shape[0]
        if is_cupy:
            return _remap_video_cupy(array, map_x, map_y, max_workers=max_workers)
        if use_cuda:
            return _remap_video_cuda(array, map_x, map_y)
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
        if is_cupy:
            return _remap_frame_cupy(array, map_x, map_y)
        return _remap_frame(array, map_x, map_y, use_cuda)
   
def generate_CropRect(inner_radius, outer_radius, number_of_plumes, centre_x, centre_y):
    if number_of_plumes==1:
        number_of_plumes = 2 # Reset to 180 degrees for stability
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

def generate_angular_mask(w, h, angle=None):

    x0 = 0
    y0 = h/2

    if angle is None:
        y1 = 0
        y2 = h
    else:
        half_angle_radian = angle / 2.0 * np.pi/180.0
        y1 = -w * np.tan(half_angle_radian) + h/2
        y2 = w * np.tan(half_angle_radian) + h/2

    # Create blank single-channel mask of same height/width
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define polygon vertices as Nx2 integer array  
    pts = np.array([[x0, y0], [w, y1], [w, y2]], dtype=np.int32)
    
    # Fill the polygon on the mask
    cv2.fillPoly(mask, [pts], (255,))

    # cv2.imshow("plume_mask", mask) # Debug

    # Apply mask to extract polygon region
    return mask >0 

# -----------------------------
# Rotation and Filtering functions
# -----------------------------
def rotate_frame(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if frame.dtype == np.bool_:
        # Convert boolean mask to uint8: True becomes 255, False becomes 0.
        frame_uint8 = (frame.astype(np.uint8)) * 255
        # Use INTER_NEAREST to preserve mask values.
        rotated_uint8 = cv2.warpAffine(frame_uint8, M, (w, h), flags=cv2.INTER_NEAREST)
        # Convert back to boolean mask.
        rotated = rotated_uint8 > 127 
    else:
        rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)
    
    return rotated

def rotate_video(video_array, angle=0, max_workers=None):
    num_frames = video_array.shape[0]
    rotated_frames = [None] * num_frames
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(rotate_frame, video_array[i], angle): i 
                           for i in range(num_frames)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                rotated_frames[idx] = future.result()
            except Exception as exc:
                print(f"Frame {idx} generated an exception during rotation: {exc}")
    return np.array(rotated_frames)

# -----------------------------
# CUDA-accelerated Rotation
# -----------------------------
def is_opencv_cuda_available():
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except AttributeError:
        return False


def is_opencv_ocl_available():
    """Check if OpenCV was built with OpenCL support."""
    try:
        return cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()
    except AttributeError:
        return False
    
def rotate_frame_cuda(frame, angle, stream=None):
    """
    在 GPU 上旋转单帧图像／掩码。
    
    Parameters
    ----------
    frame : np.ndarray (H×W or H×W×C) or bool mask
    angle : float
    stream: cv2.cuda.Stream (optional) — 用于异步操作
    
    Returns
    -------
    rotated : 同 frame 类型
    """
    h, w = frame.shape[:2]
    # 计算仿射矩阵（在 CPU 上）
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0).astype(np.float32)
    
    # 上传到 GPU
    if frame.dtype == np.bool_:
        # 布尔先转 uint8（0/255）
        cpu_uint8 = (frame.astype(np.uint8)) * 255
        gpu_mat = cv2.cuda_GpuMat()# type: ignore
        gpu_mat.upload(cpu_uint8, stream)
        # warpAffine（最近邻保留掩码边界）
        gpu_rot = cv2.cuda.warpAffine(
            gpu_mat, M, (w, h),
            flags=cv2.INTER_NEAREST, stream=stream)# type: ignore
        # 下载并阈值回布尔
        out_uint8 = gpu_rot.download(stream)
        rotated = out_uint8 > 127
    else:
        # 对普通灰度或多通道图像
        gpu_mat = cv2.cuda.GpuMat()

        if stream is not None:
            gpu_mat.upload(frame, stream)
            gpu_rot = cv2.cuda.warpAffine(
                gpu_mat, M, (w, h),
                flags=cv2.INTER_CUBIC, stream=stream
            )
            rotated = gpu_rot.download(stream)
        else:
            gpu_mat.upload(frame)
            gpu_rot = cv2.cuda.warpAffine(
                gpu_mat, M, (w, h),
                flags=cv2.INTER_CUBIC
            )
            rotated = gpu_rot.download()
    
    # 等待 GPU 流完成
    if stream is not None:
        stream.waitForCompletion()
    return rotated

def rotate_video_cuda(video_array, angle=0, max_workers=4):
    """
    并行地在 GPU 上旋转整个视频（每帧独立流）。
    
    Parameters
    ----------
    video_array : np.ndarray, shape=(N, H, W) 或 (N, H, W, C) 或 bool
    angle : float — 旋转角度（度）
    max_workers : int — 并行线程数（每线程管理一个 cv2.cuda.Stream）
    
    Returns
    -------
    np.ndarray — 与输入同形状、同 dtype
    """
    num_frames = video_array.shape[0]
    rotated = [None] * num_frames

    # 预创建若干 CUDA 流
    streams = [cv2.cuda.Stream() for _ in range(max_workers)]

    def task(idx, frame):
        # 分配一个流（简单轮询）
        stream = streams[idx % max_workers]
        return idx, rotate_frame_cuda(frame, angle, stream)

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(task, i, video_array[i]) for i in range(num_frames)]
        for fut in as_completed(futures):
            idx, out = fut.result()
            rotated[idx] = out

    return np.stack(rotated, axis=0)

def rotate_frame_ocl(frame, angle):
    """Rotate a single frame using OpenCL via OpenCV's UMat."""
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    if frame.dtype == np.bool_:
        frame_uint8 = (frame.astype(np.uint8)) * 255
        u_src = cv2.UMat(frame_uint8)
        u_rot = cv2.warpAffine(u_src, M, (w, h), flags=cv2.INTER_NEAREST) # type: ignore
        rotated = u_rot.get() > 127
    else:
        u_src = cv2.UMat(frame)
        u_rot = cv2.warpAffine(u_src, M, (w, h), flags=cv2.INTER_CUBIC) # type: ignore
        rotated = u_rot.get()
    return rotated

def rotate_video_ocl(video_array, angle=0, max_workers=4):
    """Rotate video frames using OpenCL (Intel/AMD GPUs)."""
    num_frames = video_array.shape[0]
    rotated = [None] * num_frames

    def task(idx):
        return idx, rotate_frame_ocl(video_array[idx], angle)

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(task, i) for i in range(num_frames)]
        for fut in as_completed(futures):
            idx, out = fut.result()
            rotated[idx] = out
    return np.stack(rotated, axis=0)

def rotate_video_auto(video_array, angle=0, max_workers=4):
    if is_opencv_cuda_available():
        print("Using CUDA for rotation.")
        return rotate_video_cuda(video_array, angle=angle, max_workers=max_workers)
    if is_opencv_ocl_available():
        print("Using OpenCL for rotation.")
        return rotate_video_ocl(video_array, angle=angle, max_workers=max_workers)
    print("CUDA/OpenCL not available, falling back to CPU.")
    return rotate_video(video_array, angle=angle, max_workers=max_workers)

def rotate_all_segments_auto(video, angles, crop, centre, mask=None):
    segments = []

    # Multithreaded rotation and cropping
    with ThreadPoolExecutor(max_workers=min(len(angles), os.cpu_count() or 1)) as exe:
        future_map = {
            exe.submit(
                # rotate_and_crop, video, angle, crop, centre,
                rotate_and_crop, video, angle, crop, centre,
                is_video=True, mask=mask, max_workers=None
            ): idx for idx, angle in enumerate(angles)
        }
        segments_with_idx = []
        for fut in as_completed(future_map):
            idx = future_map[fut]
            result = fut.result()
            segments_with_idx.append((idx, result))
        # Sort by index to preserve order
        segments_with_idx.sort(key=lambda x: x[0])
        segments = [seg for idx, seg in segments_with_idx]
    return segments