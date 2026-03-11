#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

template <typename T>
__device__ inline float read_as_float(const T* ptr);

template <>
__device__ inline float read_as_float<float>(const float* ptr) {
    return *ptr;
}

template <>
__device__ inline float read_as_float<__half>(const __half* ptr) {
    return __half2float(*ptr);
}

template <typename T>
__global__ void normalize_to_grayscale_u8_kernel(
    const T* src,
    std::uint8_t* dst,
    std::size_t dst_pitch_bytes,
    int height,
    int width,
    float input_scale) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float value = read_as_float(src + static_cast<std::size_t>(y) * width + x) * input_scale;
    const float clamped = value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value);
    auto* row = reinterpret_cast<std::uint8_t*>(reinterpret_cast<char*>(dst) + static_cast<std::size_t>(y) * dst_pitch_bytes);
    row[x] = static_cast<std::uint8_t>(clamped + 0.5f);
}

__global__ void decode_flow_vectors_to_float_kernel(
    const std::int16_t* src,
    std::size_t src_pitch_bytes,
    float* dst,
    int height,
    int width) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const auto* row = reinterpret_cast<const std::int16_t*>(
        reinterpret_cast<const char*>(src) + static_cast<std::size_t>(y) * src_pitch_bytes);
    const std::int16_t flow_x = row[2 * x];
    const std::int16_t flow_y = row[2 * x + 1];

    const std::size_t dst_index = (static_cast<std::size_t>(y) * width + x) * 2ULL;
    dst[dst_index] = static_cast<float>(flow_x) / 32.0f;
    dst[dst_index + 1] = static_cast<float>(flow_y) / 32.0f;
}

template <typename T>
void launch_normalize_impl(
    const void* src,
    std::uint8_t* dst,
    std::size_t dst_pitch_bytes,
    int height,
    int width,
    float input_scale,
    cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((width + block.x - 1) / block.x),
        static_cast<unsigned int>((height + block.y - 1) / block.y));
    normalize_to_grayscale_u8_kernel<<<grid, block, 0, stream>>>(
        static_cast<const T*>(src),
        dst,
        dst_pitch_bytes,
        height,
        width,
        input_scale);
}

}  // namespace

void launch_normalize_to_grayscale_u8(
    const void* src,
    int dtype_code,
    std::uint8_t* dst,
    std::size_t dst_pitch_bytes,
    int height,
    int width,
    float input_scale,
    cudaStream_t stream) {
    if (dtype_code == 0) {
        launch_normalize_impl<__half>(src, dst, dst_pitch_bytes, height, width, input_scale, stream);
        return;
    }
    launch_normalize_impl<float>(src, dst, dst_pitch_bytes, height, width, input_scale, stream);
}

void launch_decode_flow_vectors_to_float(
    const std::int16_t* src,
    std::size_t src_pitch_bytes,
    float* dst,
    int height,
    int width,
    cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((width + block.x - 1) / block.x),
        static_cast<unsigned int>((height + block.y - 1) / block.y));
    decode_flow_vectors_to_float_kernel<<<grid, block, 0, stream>>>(
        src,
        src_pitch_bytes,
        dst,
        height,
        width);
}
