import os
import sys
import inspect
from typing import Any, Optional, Tuple, List

import numpy as np

from .nvidia_hw_optical_flow import compute_nvidia_hw_flows


def _get_array_module(arr):
    try:
        import cupy as cp  # type: ignore

        if isinstance(arr, cp.ndarray):
            return cp
    except Exception:
        pass
    return np


def _scalar_to_float(value: Any) -> float:
    try:
        return float(value.item())
    except Exception:
        return float(value)


def _filter_kwargs_for_callable(func, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only keyword arguments accepted by ``func``."""
    sig = inspect.signature(func)
    accepted = {
        name
        for name, param in sig.parameters.items()
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


def _maybe_to_numpy(arr):
    """Convert Cupy array to numpy if needed; pass numpy through."""
    try:
        import cupy as cp  # type: ignore
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:
        pass
    return arr


def _normalize_to_255(arr: np.ndarray, bits: Optional[int]):
    """Scale grayscale frame to 0..255 float32. Accepts uint8/uint16/float.

    - If bits is provided (e.g., 12 or 16), uses (2^bits - 1) as max.
    - Otherwise infers from dtype, heuristically handling 12-bit data stored in uint16.
    """
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    input_max = None
    if bits is not None:
        input_max = float((1 << int(bits)) - 1)
    else:
        if np.issubdtype(arr.dtype, np.integer):
            input_max = float(np.iinfo(arr.dtype).max)
            if arr.dtype == np.uint16:
                m = float(arr.max()) if arr.size else 0.0
                if 0.0 < m <= 4095.0:
                    input_max = 4095.0
        elif np.issubdtype(arr.dtype, np.floating):
            m = float(np.nanmax(arr)) if arr.size else 1.0
            if m <= 1.0 + 1e-6:
                input_max = 1.0
            elif m <= 255.0 + 1e-6:
                input_max = 255.0
            else:
                input_max = m
        else:
            input_max = 255.0

    arr = arr.astype(np.float32)
    if input_max and input_max > 0:
        arr = arr * (255.0 / input_max)
    return np.clip(arr, 0.0, 255.0)


def compute_flow_magnitude(flows: "np.ndarray | any") -> "np.ndarray | any":
    """Return optical-flow magnitude in pixels per frame.

    Accepts either ``(F-1, H, W, 2)`` or ``(F-1, 2, H, W)`` flow stacks and
    returns a magnitude stack shaped ``(F-1, H, W)`` in the same array backend
    (NumPy or CuPy).
    """
    xp = _get_array_module(flows)
    arr = flows.astype(xp.float32, copy=False)
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D flow stack, got shape {arr.shape}")
    if arr.shape[-1] == 2:
        return xp.linalg.norm(arr, axis=-1)
    if arr.shape[1] == 2:
        return xp.linalg.norm(arr, axis=1)
    raise ValueError(f"Flow stack must have a vector axis of length 2, got shape {arr.shape}")


def normalize_flow_magnitude(
    magnitudes: "np.ndarray | any",
    *,
    lower_percentile: float = 5.0,
    upper_percentile: float = 99.0,
    eps: float = 1e-6,
) -> "np.ndarray | any":
    """Robustly scale optical-flow magnitudes to ``[0, 1]``.

    The input is assumed to already be in physical pixel-displacement units.
    Scaling is global over the provided stack so different optical-flow
    backends can be consumed on the same downstream score scale.
    """
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Expected 0 <= lower_percentile < upper_percentile <= 100")

    xp = _get_array_module(magnitudes)
    arr = magnitudes.astype(xp.float32, copy=False)
    finite_mask = xp.isfinite(arr)
    positive_mask = finite_mask & (arr > 0)

    if not _scalar_to_float(xp.any(positive_mask)):
        return xp.zeros_like(arr, dtype=xp.float32)

    samples = arr[positive_mask]
    lo = _scalar_to_float(xp.percentile(samples, lower_percentile))
    hi = _scalar_to_float(xp.percentile(samples, upper_percentile))
    max_sample = _scalar_to_float(xp.max(samples))

    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = max_sample
    if hi <= lo + eps:
        lo = 0.0
        hi = max_sample
    if hi <= lo + eps:
        return xp.zeros_like(arr, dtype=xp.float32)

    normalized = xp.clip((arr - lo) / (hi - lo + eps), 0.0, 1.0)
    normalized = xp.where(finite_mask, normalized, 0.0)
    return normalized.astype(xp.float32, copy=False)


def _frame_to_tensor(frame: np.ndarray, bits: Optional[int], device: str):
    import torch
    arr = _normalize_to_255(frame, bits)
    img = torch.from_numpy(arr).permute(2, 0, 1).float()
    return img[None].to(device)


def _resolve_raft_paths() -> Tuple[str, str, str]:
    """Return (raft_root, core_path, default_model_path)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates: List[str] = []
    # env var can point to RAFT root
    if os.getenv('RAFT_DIR'):
        candidates.append(os.getenv('RAFT_DIR'))
    # Try sibling repo paths
    candidates.append(os.path.abspath(os.path.join(here, '..', '..', 'optical_flow', 'RAFT')))
    candidates.append(os.path.abspath(os.path.join(here, '..', 'optical_flow', 'RAFT')))
    candidates.append(os.path.abspath(os.path.join(here, '..', '..', '..', 'optical_flow', 'RAFT')))
    candidates.append(os.path.abspath(os.path.join(here, '..', '..', '..', 'examples', 'RAFT')))

    for root in candidates:
        if root and os.path.isdir(root):
            core = os.path.join(root, 'core')
            if os.path.isdir(core):
                model = os.path.join(root, 'models', 'raft-things.pth')
                return root, core, model
    raise FileNotFoundError(
        "Could not locate RAFT repository. Set RAFT_DIR env var to RAFT root or place it under optical_flow/RAFT or examples/RAFT relative to the repo root.")


def _torch_load_compat(path, map_location):
    import torch
    kwargs = {"map_location": map_location}
    try:
        if 'weights_only' in inspect.signature(torch.load).parameters:
            kwargs['weights_only'] = False
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop('weights_only', None)
        return torch.load(path, **kwargs)


def compute_raft_flows(
    video: "np.ndarray | any",
    model_path: Optional[str] = None,
    iters: int = 20,
    bits: Optional[int] = None,
    device: Optional[str] = None,
    out_hw_last: bool = False,
    return_numpy: bool = True,
    small: bool = False,
    mixed_precision: bool = False,
    alternate_corr: bool = False,
):
    """Compute pair-wise RAFT optical flow for a grayscale video.

    Parameters:
    - video: (F,H,W) grayscale frames; numpy or cupy array. Integer or float types supported.
    - model_path: path to RAFT checkpoint. Defaults to bundled `raft-things.pth` in the RAFT repo.
    - iters: RAFT iterations per pair.
    - bits: bit depth for normalization (None = infer).
    - device: 'cuda' or 'cpu'. Default auto: cuda if available.
    - out_hw_last: if True, returns shape (F-1, H, W, 2); else (F-1, 2, H, W).
    - return_numpy: if True, returns numpy array; else returns torch tensor on `device`.
    - small, mixed_precision, alternate_corr: RAFT model flags.

    Returns:
    - flows: stacked pairwise flow fields between t and t+1.
    """
    import torch

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vid = _maybe_to_numpy(video)
    if not isinstance(vid, np.ndarray):
        raise TypeError("video must be a numpy or cupy array")
    if vid.ndim != 3:
        raise ValueError(f"Expected (F,H,W) array, got shape {vid.shape}")
    F, H, W = vid.shape
    if F < 2:
        raise ValueError("Need at least 2 frames to compute optical flow")

    raft_root, core_path, default_model = _resolve_raft_paths()
    if core_path not in sys.path:
        sys.path.append(core_path)

    # Import RAFT and helpers after path setup
    from raft import RAFT  # type: ignore
    from utils.utils import InputPadder  # type: ignore

    # Build args-like namespace for RAFT ctor
    class _Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __contains__(self, item):
            return hasattr(self, item)

    args = _Args(
        small=bool(small),
        mixed_precision=bool(mixed_precision),
        alternate_corr=bool(alternate_corr),
    )

    ckpt = model_path or default_model
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"RAFT checkpoint not found: {ckpt}")

    # Light validation: avoid HTML/corrupt downloads
    with open(ckpt, 'rb') as f:
        head = f.read(2)
        if head[:1] == b'<' or head == b'':
            raise ValueError("Model file is not a valid PyTorch checkpoint (starts with '<' or empty)")

    model = torch.nn.DataParallel(RAFT(args))
    state = _torch_load_compat(ckpt, device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model = model.module
    model.to(device)
    model.eval()

    flows = []
    with torch.no_grad():
        prev = None
        padder = None
        for t in range(F):
            cur = _frame_to_tensor(vid[t], bits=bits, device=device)
            if prev is None:
                prev = cur
                continue
            if padder is None:
                padder = InputPadder(prev.shape)
            i1, i2 = padder.pad(prev, cur)
            _, flow_up = model(i1, i2, iters=iters, test_mode=True)
            flow_up = padder.unpad(flow_up)
            if out_hw_last:
                flows.append(flow_up[0].permute(1, 2, 0).contiguous())
            else:
                flows.append(flow_up[0].contiguous())
            prev = cur

    if return_numpy:
        stacked = torch.stack(flows, dim=0).detach().cpu().numpy()
        return stacked
    else:
        return torch.stack(flows, dim=0)


def compute_farneback_flows(
    video: "np.ndarray | any",
    bits: Optional[int] = None,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 3,
    iterations: int = 3,
    poly_n: int = 7,
    poly_sigma: float = 1.2,
    flags: int = 0,
    out_hw_last: bool = False,
    return_cupy: bool = False,
):
    """Compute pair-wise Farneback optical flow for a grayscale video.

    Parameters:
    - video: (F,H,W) grayscale frames; numpy or cupy array. Integer or float types supported.
    - bits: bit depth for normalization to 0..255 (None = infer from dtype/value range).
    - Farneback params: pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags.
    - out_hw_last: if True, returns shape (F-1, H, W, 2); else (F-1, 2, H, W).
    - return_cupy: if True and cupy is available, return cupy array.

    Returns:
    - flows: stacked pairwise flow fields between t and t+1.
    """
    import cv2

    vid = _maybe_to_numpy(video)
    if not isinstance(vid, np.ndarray):
        raise TypeError("video must be a numpy or cupy array")
    if vid.ndim != 3:
        raise ValueError(f"Expected (F,H,W) array, got shape {vid.shape}")
    F, H, W = vid.shape
    if F < 2:
        raise ValueError("Need at least 2 frames to compute optical flow")

    # Prepare frames as single-channel float32 in [0,1]
    def prep(frame: np.ndarray) -> np.ndarray:
        # Normalize similar to RAFT helper, but keep single channel
        arr = frame.astype(np.float32)
        if bits is not None:
            denom = float((1 << int(bits)) - 1)
        else:
            if np.issubdtype(frame.dtype, np.integer):
                denom = float(np.iinfo(frame.dtype).max)
                if frame.dtype == np.uint16 and (arr.size and float(arr.max()) <= 4095.0):
                    denom = 4095.0
            else:
                m = float(np.nanmax(arr)) if arr.size else 1.0
                denom = 1.0 if m <= 1.0 + 1e-6 else (255.0 if m <= 255.0 + 1e-6 else m)
        if denom <= 0:
            denom = 1.0
        arr = arr / denom
        return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

    flows = []
    prev = prep(vid[0])
    for t in range(1, F):
        cur = prep(vid[t])
        flow = cv2.calcOpticalFlowFarneback(
            prev, cur, None,
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=flags,
        )  # H x W x 2
        if out_hw_last:
            flows.append(flow.astype(np.float32, copy=False))
        else:
            flows.append(np.transpose(flow, (2, 0, 1)).astype(np.float32, copy=False))
        prev = cur

    stacked = np.stack(flows, axis=0)
    if return_cupy:
        try:
            import cupy as cp  # type: ignore
            return cp.asarray(stacked)
        except Exception:
            pass
    return stacked


def compute_optical_flows(
    video: "np.ndarray | any",
    *,
    backend: str = "auto",
    out_hw_last: bool = True,
    return_cupy: bool | None = None,
    bits: Optional[int] = None,
    **kwargs: Any,
):
    """Unified optical-flow frontend.

    Parameters
    ----------
    video:
        Array of shape ``(F, H, W)``. NumPy or CuPy input is accepted depending on
        the selected backend.
    backend:
        One of ``auto``, ``farneback``, ``raft``, ``nvidia_hw``.
        ``auto`` prefers ``nvidia_hw`` for CuPy float16/float32 inputs, otherwise
        falls back to ``farneback``.
    out_hw_last:
        Return ``(F-1, H, W, 2)`` if True, else ``(F-1, 2, H, W)``.
    return_cupy:
        For CPU-oriented backends, optionally convert the output to CuPy.
        ``nvidia_hw`` always returns CuPy output.
    bits:
        Passed to CPU-oriented normalization helpers.
    kwargs:
        Additional backend-specific keyword arguments.
    """
    backend_key = backend.lower()
    backend_kwargs = dict(kwargs)

    is_cupy = False
    cp = None
    try:
        import cupy as cp  # type: ignore
        is_cupy = isinstance(video, cp.ndarray)
    except Exception:
        cp = None

    if backend_key == "auto":
        if is_cupy and video.dtype in (cp.float16, cp.float32):  # type: ignore[union-attr]
            backend_key = "nvidia_hw"
        else:
            backend_key = "farneback"

    if backend_key == "nvidia_hw":
        if cp is None or not is_cupy:
            raise TypeError("backend='nvidia_hw' requires a CuPy array input")
        flows = compute_nvidia_hw_flows(
            video,
            **_filter_kwargs_for_callable(compute_nvidia_hw_flows, backend_kwargs),
        )
        if out_hw_last:
            return flows
        return cp.transpose(flows, (0, 3, 1, 2))

    if backend_key == "farneback":
        result = compute_farneback_flows(
            video,
            bits=bits,
            out_hw_last=out_hw_last,
            return_cupy=bool(return_cupy),
            **_filter_kwargs_for_callable(compute_farneback_flows, backend_kwargs),
        )
        return result

    if backend_key == "raft":
        result = compute_raft_flows(
            video,
            bits=bits,
            out_hw_last=out_hw_last,
            return_numpy=not bool(return_cupy),
            **_filter_kwargs_for_callable(compute_raft_flows, backend_kwargs),
        )
        if return_cupy:
            try:
                import cupy as cp  # type: ignore
                if isinstance(result, np.ndarray):
                    return cp.asarray(result)
            except Exception:
                pass
        return result

    raise ValueError("backend must be one of: auto, farneback, raft, nvidia_hw")


def compute_optical_flow_magnitude(
    video: "np.ndarray | any",
    *,
    backend: str = "auto",
    normalize: bool = False,
    lower_percentile: float = 5.0,
    upper_percentile: float = 99.0,
    return_cupy: bool | None = None,
    bits: Optional[int] = None,
    **kwargs: Any,
):
    """Compute pairwise optical-flow magnitude.

    The raw magnitude is in pixels per frame for all supported backends.
    Set ``normalize=True`` to robustly map the stack onto ``[0, 1]``.
    """
    magnitudes = compute_flow_magnitude(
        compute_optical_flows(
            video,
            backend=backend,
            out_hw_last=True,
            return_cupy=return_cupy,
            bits=bits,
            **kwargs,
        )
    )
    if not normalize:
        return magnitudes
    return normalize_flow_magnitude(
        magnitudes,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
