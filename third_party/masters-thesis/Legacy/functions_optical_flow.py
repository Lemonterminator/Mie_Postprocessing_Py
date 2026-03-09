from packages import *
# -----------------------------
# Optical Flow and related functions
# -----------------------------
def compute_flow_pair(prev_frame, next_frame, ): 
    """
    Computes optical flow between two consecutive frames using the Farneback method.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_frame,
        next=next_frame,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=5,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow

def compute_optical_flow(video_array, max_workers=None):
    flows = []
    num_frames = video_array.shape[0]
    frame_pairs = [(video_array[i, :, :], video_array[i+1, :, :]) for i in range(num_frames - 1)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(compute_flow_pair, pair[0], pair[1]): idx 
                           for idx, pair in enumerate(frame_pairs)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                flow = future.result()
                flows.append((idx, flow))
            except Exception as exc:
                print(f'Frame pair {idx} generated an exception: {exc}')
    flows.sort(key=lambda x: x[0])
    sorted_flows = np.array([flow for idx, flow in flows])
    return sorted_flows

def compute_flow_scalar(flow, multiplier=1, y_scale=1):
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]
    flow_mag = np.sqrt(flow_x**2 + y_scale*flow_y**2) * multiplier
    return flow_mag

def compute_flow_scalar_video(flow_array, multiplier=1, max_workers=None, y_scale=1):
    """
    Computes scalar optical flow magnitudes in parallel for each optical flow frame 
    in a video/flow sequence using compute_flow_scalar.
    
    Parameters:
        flow_array (np.ndarray): Array of optical flow frames with shape (N, H, W, 2).
        multiplier (float, optional): A multiplier to scale the computed magnitude.
        max_workers (int, optional): Maximum number of worker processes.
        
    Returns:
        np.ndarray: Array of scalar optical flow magnitudes corresponding to each flow frame.
    """
    # Create a list of future tasks for each flow frame
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(compute_flow_scalar, flow, multiplier, y_scale): idx 
            for idx, flow in enumerate(flow_array)
        }
        results = []
        # Collect results as they are completed
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                scalar_flow = future.result()
                results.append((idx, scalar_flow))
            except Exception as exc:
                print(f'Optical flow at index {idx} generated an exception: {exc}')
    # Sort the results by their original index to preserve the order
    results.sort(key=lambda x: x[0])
    sorted_scalars = np.array([scalar for idx, scalar in results])
    return sorted_scalars



# CUDA based parallization, enviroment needs to be manually installed later on 

def compute_flow_pair_cuda(prev_frame, next_frame):

    """
    Computes optical flow between two consecutive frames using OpenCV's CUDA-accelerated Farneback method.
    
    Parameters:
    - prev_frame: numpy.ndarray (grayscale, uint8)
    - next_frame: numpy.ndarray (grayscale, uint8)
    
    Returns:
    - flow: numpy.ndarray representing the optical flow between the two frames.
    """
    # Upload frames to GPU memory
    prev_gpu = cv2.cuda_GpuMat()
    next_gpu = cv2.cuda_GpuMat()
    prev_gpu.upload(prev_frame)
    next_gpu.upload(next_frame)

        
    # Create a CUDA Farneback optical flow object
    optical_flow = cv2.cuda_FarnebackOpticalFlow.create(
        numLevels=3,
        pyrScale=0.5,
        winSize=3,
        numIters=3,
        polyN=7,
        polySigma=1.2,
        flags=0
    )
    
    # Compute the optical flow on the GPU
    flow_gpu = optical_flow.calc(prev_gpu, next_gpu, None)
    
    # Download the result back to the CPU
    flow = flow_gpu.download()
    return flow

def compute_optical_flow_cuda(video_array):
    """
    Computes optical flow for each consecutive frame pair in the video using CUDA acceleration.
    
    Parameters:
    - video_array: numpy.ndarray of shape (num_frames, height, width) (grayscale, uint8)
    
    Returns:
    - flows: numpy.ndarray containing optical flow for each frame pair.
    """
    num_frames = video_array.shape[0]
    flows = []
    
    # Create the CUDA optical flow object once outside the loop if you wish to reuse it
    optical_flow = cv2.cuda_FarnebackOpticalFlow.create(
        numLevels=3,
        pyrScale=0.5,
        winSize=3,
        numIters=3,
        polyN=7,
        polySigma=1.2,
        flags=0
    )
    
    for i in range(num_frames - 1):
        prev_frame = video_array[i]
        next_frame = video_array[i+1]
        
        # Upload frames to GPU
        prev_gpu = cv2.cuda_GpuMat()
        next_gpu = cv2.cuda_GpuMat()
        prev_gpu.upload(prev_frame)
        next_gpu.upload(next_frame)
        
        # Compute the optical flow on the GPU using the pre-created object
        flow_gpu = optical_flow.calc(prev_gpu, next_gpu, None)
        flow = flow_gpu.download()
        flows.append(flow)
    
    # Convert the list to a numpy array for consistency
    return np.array(flows)



