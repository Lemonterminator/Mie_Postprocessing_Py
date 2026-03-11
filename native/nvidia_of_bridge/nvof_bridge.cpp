#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "NvOF.h"
#include "NvOFCuda.h"
#include "NvOFUtils.h"

namespace {

thread_local std::string g_last_error;

constexpr int kInputDTypeFloat16 = 0;
constexpr int kInputDTypeFloat32 = 1;

void set_last_error(const std::string& message) {
    g_last_error = message;
}

void clear_last_error() {
    g_last_error.clear();
}

void check_cuda_runtime(cudaError_t status, const char* expr) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << expr << " failed: " << cudaGetErrorName(status) << " (" << cudaGetErrorString(status) << ")";
        throw std::runtime_error(oss.str());
    }
}

void check_cuda_driver(CUresult status, const char* expr) {
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* desc = nullptr;
        cuGetErrorName(status, &name);
        cuGetErrorString(status, &desc);
        std::ostringstream oss;
        oss << expr << " failed: " << (name ? name : "CUDA_ERROR") << " (" << (desc ? desc : "unknown") << ")";
        throw std::runtime_error(oss.str());
    }
}

class ScopedCudaContext {
public:
    explicit ScopedCudaContext(int requested_device) {
        check_cuda_driver(cuInit(0), "cuInit");

        if (requested_device >= 0) {
            check_cuda_runtime(cudaSetDevice(requested_device), "cudaSetDevice");
        }
        check_cuda_runtime(cudaFree(nullptr), "cudaFree(0)");

        int current_device = -1;
        check_cuda_runtime(cudaGetDevice(&current_device), "cudaGetDevice");
        device_ = current_device;

        CUcontext current = nullptr;
        check_cuda_driver(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
        if (current != nullptr) {
            context_ = current;
            owns_primary_ctx_ = false;
            return;
        }

        CUdevice cu_device = 0;
        check_cuda_driver(cuDeviceGet(&cu_device, device_), "cuDeviceGet");
        check_cuda_driver(cuDevicePrimaryCtxRetain(&context_, cu_device), "cuDevicePrimaryCtxRetain");
        check_cuda_driver(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
        owns_primary_ctx_ = true;
    }

    ~ScopedCudaContext() {
        if (owns_primary_ctx_) {
            CUdevice cu_device = 0;
            if (cuDeviceGet(&cu_device, device_) == CUDA_SUCCESS) {
                cuDevicePrimaryCtxRelease(cu_device);
            }
        }
    }

    CUcontext get() const { return context_; }
    int device() const { return device_; }

private:
    CUcontext context_ = nullptr;
    int device_ = 0;
    bool owns_primary_ctx_ = false;
};

NvOFBufferCudaDevicePtr* as_device_ptr_buffer(NvOFBuffer* buffer, const char* what) {
    auto* ptr = dynamic_cast<NvOFBufferCudaDevicePtr*>(buffer);
    if (ptr == nullptr) {
        std::ostringstream oss;
        oss << what << " is not backed by a CUDA device pointer buffer";
        throw std::runtime_error(oss.str());
    }
    return ptr;
}

NV_OF_PERF_LEVEL parse_preset(int preset_level) {
    switch (preset_level) {
    case NV_OF_PERF_LEVEL_SLOW:
        return NV_OF_PERF_LEVEL_SLOW;
    case NV_OF_PERF_LEVEL_MEDIUM:
        return NV_OF_PERF_LEVEL_MEDIUM;
    case NV_OF_PERF_LEVEL_FAST:
        return NV_OF_PERF_LEVEL_FAST;
    default:
        throw std::runtime_error("Unsupported preset level");
    }
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
    cudaStream_t stream);

void launch_decode_flow_vectors_to_float(
    const std::int16_t* src,
    std::size_t src_pitch_bytes,
    float* dst,
    int height,
    int width,
    cudaStream_t stream);

extern "C" {

#if defined(_WIN32)
#define NVOF_BRIDGE_EXPORT __declspec(dllexport)
#else
#define NVOF_BRIDGE_EXPORT
#endif

NVOF_BRIDGE_EXPORT const char* nvidia_of_get_last_error() {
    return g_last_error.c_str();
}

NVOF_BRIDGE_EXPORT int nvidia_of_compute_flow(
    const void* input_ptr,
    int dtype_code,
    int frames,
    int height,
    int width,
    float input_scale,
    int device_id,
    int preset_level,
    int requested_grid_size,
    int enable_temporal_hints,
    std::uint64_t stream_handle,
    void* output_ptr) {
    clear_last_error();

    try {
        if (input_ptr == nullptr) {
            throw std::runtime_error("input_ptr is null");
        }
        if (output_ptr == nullptr) {
            throw std::runtime_error("output_ptr is null");
        }
        if (dtype_code != kInputDTypeFloat16 && dtype_code != kInputDTypeFloat32) {
            throw std::runtime_error("dtype_code must be 0(float16) or 1(float32)");
        }
        if (frames < 2) {
            throw std::runtime_error("At least 2 frames are required");
        }
        if (height <= 0 || width <= 0) {
            throw std::runtime_error("height and width must be positive");
        }
        if (!(input_scale > 0.0f)) {
            throw std::runtime_error("input_scale must be > 0");
        }
        if (requested_grid_size <= 0) {
            throw std::runtime_error("requested_grid_size must be positive");
        }

        ScopedCudaContext ctx(device_id);
        auto* stream = reinterpret_cast<CUstream>(stream_handle);
        auto perf = parse_preset(preset_level);

        NvOFObj optical_flow = NvOFCuda::Create(
            ctx.get(),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            NV_OF_BUFFER_FORMAT_GRAYSCALE8,
            NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
            NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
            NV_OF_MODE_OPTICALFLOW,
            perf,
            stream,
            stream);

        uint32_t requested_grid = static_cast<uint32_t>(requested_grid_size);
        uint32_t hw_grid = requested_grid;
        uint32_t scale_factor = 1;
        if (!optical_flow->CheckGridSize(requested_grid)) {
            if (!optical_flow->GetNextMinGridSize(requested_grid, hw_grid)) {
                throw std::runtime_error("Requested grid size is not supported");
            }
            scale_factor = hw_grid / requested_grid;
        }

        optical_flow->Init(hw_grid, NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED, false, false);

        auto input_buffers = optical_flow->CreateBuffers(NV_OF_BUFFER_USAGE_INPUT, 2);
        auto output_buffers = optical_flow->CreateBuffers(NV_OF_BUFFER_USAGE_OUTPUT, 1);

        std::vector<NvOFBufferObj> upsample_buffers;
        std::unique_ptr<NvOFUtils> upsampler;
        NvOFBuffer* flow_output = output_buffers[0].get();

        if (scale_factor > 1) {
            const auto out_width = static_cast<uint32_t>((width + requested_grid - 1) / requested_grid);
            const auto out_height = static_cast<uint32_t>((height + requested_grid - 1) / requested_grid);
            upsample_buffers = optical_flow->CreateBuffers(out_width, out_height, NV_OF_BUFFER_USAGE_OUTPUT, 1);
            upsampler = std::make_unique<NvOFUtilsCuda>(NV_OF_MODE_OPTICALFLOW);
            flow_output = upsample_buffers[0].get();
        }

        auto* in0 = as_device_ptr_buffer(input_buffers[0].get(), "input buffer 0");
        auto* in1 = as_device_ptr_buffer(input_buffers[1].get(), "input buffer 1");
        auto* out = as_device_ptr_buffer(output_buffers[0].get(), "output buffer");

        NvOFBufferCudaDevicePtr* upsampled_out = nullptr;
        if (scale_factor > 1) {
            upsampled_out = as_device_ptr_buffer(flow_output, "upsampled output buffer");
        }

        const auto input_frame_bytes = static_cast<std::size_t>(height) * static_cast<std::size_t>(width) *
            static_cast<std::size_t>(dtype_code == kInputDTypeFloat16 ? sizeof(std::uint16_t) : sizeof(float));
        auto* input_base = static_cast<const std::uint8_t*>(input_ptr);
        auto* output_base = static_cast<float*>(output_ptr);
        const auto output_frame_floats = static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * 2ULL;

        auto launch_into_input = [&](const std::uint8_t* frame_ptr, NvOFBufferCudaDevicePtr* dst) {
            const auto pitch = dst->getStrideInfo().strideInfo[0].strideXInBytes;
            launch_normalize_to_grayscale_u8(
                frame_ptr,
                dtype_code,
                reinterpret_cast<std::uint8_t*>(dst->getCudaDevicePtr()),
                static_cast<std::size_t>(pitch),
                height,
                width,
                input_scale,
                reinterpret_cast<cudaStream_t>(stream));
        };

        launch_into_input(input_base, in0);

        for (int frame_idx = 0; frame_idx < frames - 1; ++frame_idx) {
            const auto* next_frame = input_base + static_cast<std::size_t>(frame_idx + 1) * input_frame_bytes;
            launch_into_input(next_frame, in1);

            optical_flow->Execute(
                input_buffers[0].get(),
                input_buffers[1].get(),
                output_buffers[0].get(),
                nullptr,
                nullptr,
                0,
                nullptr,
                nullptr,
                0,
                nullptr,
                enable_temporal_hints ? NV_OF_FALSE : NV_OF_TRUE);

            NvOFBufferCudaDevicePtr* decode_src = out;
            if (scale_factor > 1) {
                upsampler->Upsample(output_buffers[0].get(), flow_output, scale_factor);
                decode_src = upsampled_out;
            }

            launch_decode_flow_vectors_to_float(
                reinterpret_cast<const std::int16_t*>(decode_src->getCudaDevicePtr()),
                static_cast<std::size_t>(decode_src->getStrideInfo().strideInfo[0].strideXInBytes),
                output_base + static_cast<std::size_t>(frame_idx) * output_frame_floats,
                height,
                width,
                reinterpret_cast<cudaStream_t>(stream));

            std::swap(input_buffers[0], input_buffers[1]);
        }

        check_cuda_runtime(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
        return 0;
    } catch (const std::exception& ex) {
        set_last_error(ex.what());
        return -1;
    }
}

NVOF_BRIDGE_EXPORT int nvidia_of_get_caps(
    int device_id,
    std::uint32_t* grid_sizes_out,
    int max_grid_count,
    int* grid_count_out,
    int* roi_supported_out) {
    clear_last_error();

    try {
        if (grid_count_out == nullptr || roi_supported_out == nullptr) {
            throw std::runtime_error("Output pointers must not be null");
        }
        if (max_grid_count < 0) {
            throw std::runtime_error("max_grid_count must be non-negative");
        }

        ScopedCudaContext ctx(device_id);
        NvOFObj optical_flow = NvOFCuda::Create(
            ctx.get(),
            32,
            32,
            NV_OF_BUFFER_FORMAT_GRAYSCALE8,
            NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
            NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
            NV_OF_MODE_OPTICALFLOW,
            NV_OF_PERF_LEVEL_MEDIUM,
            nullptr,
            nullptr);

        std::vector<std::uint32_t> supported;
        for (std::uint32_t candidate : {1u, 2u, 4u}) {
            if (optical_flow->CheckGridSize(candidate)) {
                supported.push_back(candidate);
            }
        }

        if (grid_sizes_out != nullptr && max_grid_count > 0) {
            const int copy_count = std::min<int>(max_grid_count, static_cast<int>(supported.size()));
            std::memcpy(grid_sizes_out, supported.data(), static_cast<std::size_t>(copy_count) * sizeof(std::uint32_t));
        }

        *grid_count_out = static_cast<int>(supported.size());
        *roi_supported_out = optical_flow->IsROISupported() ? 1 : 0;
        return 0;
    } catch (const std::exception& ex) {
        set_last_error(ex.what());
        return -1;
    }
}

}  // extern "C"
