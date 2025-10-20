from mie_postprocessing import *
from mie_postprocessing.functions_videos import *
from mie_postprocessing.rotate_with_alignment import *
from mie_postprocessing.video_playback import *
from mie_postprocessing.optical_flow import *
from mie_postprocessing.video_filters import *
from mie_postprocessing.functions_bw import *
import os
import json
import sys

folder_path = r"G:\SCH_vids\Set2"
# folder_path = r"D:\OSCC\Meth\T1"
files = os.listdir(folder_path)

files = [os.path.join(folder_path, f) for f in files if os.path.isfile(os.path.join(folder_path, f))]

import numpy as np

def svd_foreground(video_FHW, rank=2, center='median', bkg_frame_limit=-1, return_bg=False):
    """
    video_FHW: ndarray, shape (F, H, W)
    rank: 截断奇异值个数（背景秩）。静止背景可设 1~2，缓慢变化 2~5。
    center: 对每个像素做时域去偏置，'median' 更鲁棒，也可 'mean' 或 None
    return_bg: 是否同时返回背景视频（F,H,W）

    returns:
        fg_FHW: 残差前景，shape (F,H,W)
        (opt) bg_FHW: 低秩背景，shape (F,H,W)
    """
    F, H, W = video_FHW.shape
    X = video_FHW.reshape(F, -1).T        # (HW, F)

    # 时域去偏置：降低光照慢漂移的影响
    if center == 'median':
        if bkg_frame_limit == -1: 
            bias = cp.median(X, axis=1, keepdims=True)
        else:
            bias = cp.median(X[:,:bkg_frame_limit], axis=1, keepdims=True)
        Xc = X - bias
    elif center == 'mean':
        if bkg_frame_limit == -1: 
            bias = cp.mean(X, axis=1, keepdims=True)
        else:
            bias = cp.mean(X[:,:bkg_frame_limit], axis=1, keepdims=True)
        Xc = X - bias
    else:
        bias = 0.0
        Xc = X

    # 截断 SVD：用 cp.linalg.svd 再截断，或换 randomized SVD（见下）
    U, s, Vt = cp.linalg.svd(Xc, full_matrices=False)  # U:(HW,rmax), s:(rmax,), Vt:(F,rmax)
    r = min(rank, s.size)
    Uk = U[:, :r]            # (HW,r)
    sk = s[:r]               # (r,)
    Vtk = Vt[:r, :]          # (r,F)

    Lc = (Uk * sk) @ Vtk     # 低秩背景（zero-centered），(HW,F)
    S = Xc - Lc              # 残差前景（zero-centered）

    # 复原形状
    fg_FHW = S.T.reshape(F, H, W)

    if return_bg:
        bg = (Lc + bias).T.reshape(F, H, W)
        return fg_FHW, bg
    else:
        return fg_FHW

def godec_like(video_FHW, rank=2, lam=2.5, iters=8, center='median', return_bg=False):
    """
    GoDec 风格的近似 RPCA:
        X ≈ L + S,  s.t. rank(L) <= r,  S 稀疏（按阈值截断）
    lam: 稀疏阈值强度（越大前景越“干净”，但易漏检）
    iters: 交替次数，通常 5~10 就够
    """
    F, H, W = video_FHW.shape
    X = video_FHW.reshape(F, -1).T  # (HW,F)

    # 去偏置
    if center == 'median':
        bias = cp.median(X, axis=1, keepdims=True)
        Xc = X - bias
    elif center == 'mean':
        bias = cp.mean(X, axis=1, keepdims=True)
        Xc = X - bias
    else:
        bias = 0.0
        Xc = X

    # 初始化
    L = cp.zeros_like(Xc)
    S = cp.zeros_like(Xc)

    for _ in range(iters):
        # 1) 低秩投影：对 (Xc - S) 做截断SVD
        U, s, Vt = cp.linalg.svd(Xc - S, full_matrices=False)
        r = min(rank, s.size)
        L = (U[:, :r] * s[:r]) @ Vt[:r, :]

        # 2) 稀疏截断：软阈值（更稳）或硬阈值（更稀疏）
        R = Xc - L
        # 全局自适应阈值（MAD）
        med = cp.median(cp.abs(R))
        tau = lam * 1.4826 * med + 1e-8
        # 软阈值
        S = cp.sign(R) * cp.maximum(cp.abs(R) - tau, 0.0)

    fg = (Xc - L).T.reshape(F, H, W)  # 残差视作前景
    if return_bg:
        bg = (L + bias).T.reshape(F, H, W)
        return fg, bg
    else:
        return fg



for file in files:
    if Path(file).suffix == '.json':
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                number_of_plumes = int(data['plumes'])
                offset = float(data['offset'])
                centre = (float(data['centre_x']) + 4, float(data['centre_y'])-3)

for file in files:
    # if Path(file).suffix == '.cine' and Path(file).stem.split("Shadow")[0]=="":
    if Path(file).suffix == '.cine':
        video = load_cine_video(file, frame_limit=399).astype(np.float32)/4096
        F, H, W = video.shape

        # Rotate to horizontal and crop by half
        rotated, _, _ = rotate_video_nozzle_at_0_half_cupy(video, centre, -45, interpolation="bicubic", border_mode="constant", out_shape=(H//2, W))

        intensity = cp.sum(cp.sum(rotated, axis=2), axis=1).get()
        # plt.plot(intensity)
        
        # Inverting 
        rotated = 1 - rotated 

        smooth_frames = 3
        temporal_smoothing = cp.swapaxes(median_filter_video_auto(cp.swapaxes(rotated.get(), 0, 2), smooth_frames, 1), 0, 2)
        # temporal_smoothing = cp.asarray(cp.clip(temporal_smoothing, 0, 1),dtype=cp.float32)

        # Suppose temarrayporal_smoothing is your CuPy 
        min_val = temporal_smoothing.min()
        max_val = temporal_smoothing.max()

        # Avoid division by zero if all values are the same
        if max_val > min_val:
            temporal_smoothing = (temporal_smoothing - min_val) / (max_val - min_val)
        else:
            temporal_smoothing = cp.zeros_like(temporal_smoothing, dtype=cp.float32)

        temporal_smoothing = cp.asarray(temporal_smoothing, dtype=cp.float32)



        # rotated_CPU = rotated.get() # Back to CPU
        F, H, W = rotated.shape

        foreground_svd = svd_foreground(temporal_smoothing, 10, bkg_frame_limit=20)
        foreground_godec = godec_like(temporal_smoothing, 10)

        svd_pos = foreground_svd*(foreground_svd > 0)

        svd_neg = -foreground_svd*(foreground_svd <0)

        
        
        # Min-Max scaling 
        min_val = svd_pos.min()
        max_val = svd_pos.max()
        svd_pos = (svd_pos - min_val)/(max_val - min_val)

        min_val = svd_neg.min()
        max_val = svd_neg.max()
        svd_neg = (svd_neg - min_val)/(max_val - min_val)
        
        gamma = 1.5
        play_videos_side_by_side((
            cp.swapaxes(rotated, 1,2).get(), 
            cp.swapaxes(svd_pos, 1, 2).get()**gamma, 
            cp.swapaxes(svd_neg, 1, 2).get()**gamma
            ), intv=100)
        
        tdi_map = cp.sum(svd_pos**gamma, axis=1).get()
        plt.imshow(tdi_map.T, cmap="jet", origin="lower")

        # Time-Distance Intensity Map

        tdi_map = (tdi_map-tdi_map.min())/(tdi_map.max()-tdi_map.min())
        tdi_map = np.clip(tdi_map, 0, 1)
        tdi_map_u8 = (tdi_map*255).astype(np.uint8)
        thres, bw = cv2.threshold(tdi_map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw1 = keep_largest_component(bw)
        bw2 = bw1.T
        
        penetration = bw2.shape[0]- np.argmax(bw2[::-1, :], axis=0)
        penetration[penetration == bw2.shape[0]]  = 0
        # plt.imshow(bw2, origin="lower")
        plt.plot(penetration, color="red")

        # Examing the penetration results
        rotated_cp = rotated.copy()
        for f in range(F):
            rotated_cp[f, :, penetration[f]:]  *= 1e-1
        play_video_cv2(rotated_cp.get(), intv=100)


        # plt.imshow(cp.sum(foreground_godec, axis=1).get(), cmap="jet", origin="lower")
        

        min_val = foreground_godec.min()
        max_val = foreground_godec.max()
        foreground_godec = (foreground_godec - min_val)/(max_val - min_val)


        play_videos_side_by_side((rotated.get(), np.clip(foreground_svd.get(), 0,1), np.clip(foreground_godec.get(), 0, 1)))


        # Optical flow
        flows_svd = compute_farneback_flows(svd_pos)
        flows_mag = np.sqrt(flows_svd[:, 0, :,:]**2+ flows_svd[:,1,:,:]**2)
        min_val = flows_mag.min()
        max_val = flows_mag.max()
        flows_mag = (flows_mag-min_val)/(max_val-min_val)
        play_video_cv2(flows_mag*10)
        tdi_map_flows = np.sum(flows_mag, axis=1)

        min_val = tdi_map_flows.min()
        max_val = tdi_map_flows.max()
        tdi_map_flows_scaled = (tdi_map_flows- min_val ) / (max_val-min_val)


        xy_median = median_filter_video_auto(foreground_godec, 5, 5)
        xy_blur = gaussian_video_cpu(xy_median, ksize=(7,7))
        play_videos_side_by_side((rotated.get(), temporal_smoothing.get(), foreground_godec.get(), xy_median, xy_blur))
        
        




        



