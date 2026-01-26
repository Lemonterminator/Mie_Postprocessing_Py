import cupy as cp


_bilateral_kernel = cp.RawKernel(r'''
                                 
extern "C" __global__
void bilateral2d(
    const float* __restrict__ pad,  // (Hp, Wp)
    float* __restrict__ out,        // (H, W)
    const float* __restrict__ spatial, // (wsize, wsize)
    int H, int W, int Wp,
    int k, int wsize,
    float inv_2sr2,
    float eps
){
    int x = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    int y = (int)(blockDim.y * blockIdx.y + threadIdx.y);
    if (x >= W || y >= H) return;

    // center in padded coordinates
    int cx = x + k;
    int cy = y + k;
    float c = pad[cy * Wp + cx];

    float sum_w  = 0.0f;
    float sum_wp = 0.0f;

    // iterate window
    int base_y = cy - k;
    int base_x = cx - k;

    // spatial index base
    int sidx = 0;
    for (int dy = 0; dy < wsize; ++dy){
        int py = base_y + dy;
        int row = py * Wp;
        for (int dx = 0; dx < wsize; ++dx, ++sidx){
            float p = pad[row + (base_x + dx)];
            float d = p - c;
            float range = __expf(-(d*d) * inv_2sr2);
            float w = range * spatial[sidx];
            sum_w  += w;
            sum_wp += w * p;
        }
    }
    out[y * W + x] = sum_wp / (sum_w + eps);
}
                                 
''', 'bilateral2d')



def bilateral_filter_video_cupy_fast(video, wsize, sigma_d, sigma_r, mode="edge",
                                     block=(16, 16), eps=1e-8):
    video_gpu = cp.asarray(video, dtype=cp.float32)
    F, H, W = video_gpu.shape
    k = wsize // 2

    # pad all frames once
    pad = cp.pad(video_gpu, ((0,0),(k,k),(k,k)), mode=mode)
    _, Hp, Wp = pad.shape

    # spatial kernel once
    ax = cp.arange(-k, k+1, dtype=cp.float32)
    xx, yy = cp.meshgrid(ax, ax, indexing="ij")
    inv_2sd2 = cp.float32(1.0) / (2.0 * cp.float32(sigma_d) * cp.float32(sigma_d))
    spatial = cp.exp(-(xx*xx + yy*yy) * inv_2sd2).astype(cp.float32).ravel()  # flatten

    inv_2sr2 = cp.float32(1.0) / (2.0 * cp.float32(sigma_r) * cp.float32(sigma_r))
    eps = cp.float32(eps)

    out = cp.empty((F, H, W), dtype=cp.float32)

    grid = ((W + block[0] - 1) // block[0],
            (H + block[1] - 1) // block[1])

    for f in range(F):
        _bilateral_kernel(
            (grid[0], grid[1]), block,
            (pad[f], out[f], spatial, H, W, Wp, k, wsize, inv_2sr2, eps)
        )
    return out