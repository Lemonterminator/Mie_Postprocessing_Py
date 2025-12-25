import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import as_completed, ThreadPoolExecutor

# -----------------------------
# Rotation functions
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
    ? GPU ???????/???
    
    Parameters
    ----------
    frame : np.ndarray (H�W or H�W�C) or bool mask
    angle : float
    stream: cv2.cuda.Stream (optional) � ??????
    
    Returns
    -------
    rotated : ? frame ??
    """
    h, w = frame.shape[:2]
    # ??????(? CPU ?)
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0).astype(np.float32)
    
    # ??? GPU
    if frame.dtype == np.bool_:
        # ???? uint8(0/255)
        cpu_uint8 = (frame.astype(np.uint8)) * 255
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(cpu_uint8, stream)
        # warpAffine(?????????)
        gpu_rot = cv2.cuda.warpAffine(
            gpu_mat, M, (w, h),
            flags=cv2.INTER_NEAREST, stream=stream
        )
        # ????????
        out_uint8 = gpu_rot.download(stream)
        rotated = out_uint8 > 127
    else:
        # ???????????
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
    
    # ?? GPU ???
    if stream is not None:
        stream.waitForCompletion()
    return rotated

def rotate_video_cuda(video_array, angle=0, max_workers=4):
    """
    ???? GPU ???????(?????)?
    
    Parameters
    ----------
    video_array : np.ndarray, shape=(N, H, W) ? (N, H, W, C) ? bool
    angle : float � ????(?)
    max_workers : int � ?????(??????? cv2.cuda.Stream)
    
    Returns
    -------
    np.ndarray � ???????? dtype
    """
    num_frames = video_array.shape[0]
    rotated = [None] * num_frames

    # ????? CUDA ?
    streams = [cv2.cuda.Stream() for _ in range(max_workers)]

    def task(idx, frame):
        # ?????(????)
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
        u_rot = cv2.warpAffine(u_src, M, (w, h), flags=cv2.INTER_NEAREST)
        rotated = u_rot.get() > 127
    else:
        u_src = cv2.UMat(frame)
        u_rot = cv2.warpAffine(u_src, M, (w, h), flags=cv2.INTER_CUBIC)
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

def rotate_video_nozzle_centering(video_array, centre_x, centre_y, angle, y_crop=None, interpolation=cv2.INTER_CUBIC):
    """Rotate frames so the nozzle centre ends up at x=0 and mid-height.

    Parameters
    ----------
    video_array : np.ndarray
        Stack of frames shaped (F, H, W) or (F, H, W, C).
    centre_x, centre_y : float
        Nozzle coordinates in the original frame.
    angle : float
        Rotation angle in degrees (positive is counter-clockwise).
    y_crop : tuple[int, int], optional
        Output row range (start, end) to keep after rotation.
    interpolation : int
        Interpolation flag for cv2.remap when data is not boolean.
    """
    video = np.asarray(video_array)
    if video.ndim not in (3, 4):
        raise ValueError("video_array must be 3-D or 4-D (frames, height, width[, channels])")

    num_frames, h, w = video.shape[0], video.shape[1], video.shape[2]

    if y_crop is None:
        y_start, y_end = 0, h
    else:
        if len(y_crop) != 2:
            raise ValueError("y_crop must be a tuple like (start, end)")
        y_start, y_end = int(y_crop[0]), int(y_crop[1])
        if not (0 <= y_start < y_end <= h):
            raise ValueError("y_crop must satisfy 0 <= start < end <= video height")

    h_out = y_end - y_start
    if h_out <= 0:
        raise ValueError("y_crop results in empty output height")

    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(y_start, y_end, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    output_center_y = y_start + (h_out - 1) / 2.0
    theta = np.deg2rad(angle)
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)

    rel_x = grid_x
    rel_y = grid_y - output_center_y

    map_x = (centre_x + cos_a * rel_x + sin_a * rel_y).astype(np.float32)
    map_y = (centre_y - sin_a * rel_x + cos_a * rel_y).astype(np.float32)

    rotated_frames = []
    for idx in range(num_frames):
        frame = video[idx]
        if frame.dtype == np.bool_:
            frame_u8 = frame.astype(np.uint8) * 255
            remapped = cv2.remap(
                frame_u8, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            rotated = remapped > 127
        else:
            rotated = cv2.remap(
                frame, map_x, map_y, interpolation=interpolation,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        rotated_frames.append(rotated)

    return np.stack(rotated_frames, axis=0)
