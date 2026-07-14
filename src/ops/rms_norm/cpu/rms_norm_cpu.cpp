#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight,
               float eps, size_t M, size_t N) {
    for (size_t i = 0; i < M; i++) {
        float sum_sq = 0.0f;
        for (size_t j = 0; j < N; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float val = llaisys::utils::cast<float>(in[i * N + j]);
                sum_sq += val * val;
            } else {
                float val = static_cast<float>(in[i * N + j]);
                sum_sq += val * val;
            }
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(N) + eps);

        for (size_t j = 0; j < N; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float in_val = llaisys::utils::cast<float>(in[i * N + j]);
                float w_val = llaisys::utils::cast<float>(weight[j]);
                out[i * N + j] = llaisys::utils::cast<T>(w_val * in_val / rms);
            } else {
                out[i * N + j] = static_cast<T>(
                    static_cast<float>(weight[j]) * static_cast<float>(in[i * N + j]) / rms);
            }
        }
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t type, size_t M, size_t N) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            eps, M, N);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            eps, M, N);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            eps, M, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
