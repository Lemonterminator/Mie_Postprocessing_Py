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
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(cpu_uint8, stream)
        # warpAffine（最近邻保留掩码边界）
        gpu_rot = cv2.cuda.warpAffine(
            gpu_mat, M, (w, h),
            flags=cv2.INTER_NEAREST, stream=stream
        )
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
