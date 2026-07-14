#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

inline float bf16ToFloat(llaisys::bf16_t value) {
    uint32_t bits = static_cast<uint32_t>(value._v) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline llaisys::bf16_t floatToBf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    return llaisys::bf16_t{static_cast<uint16_t>((bits + rounding_bias) >> 16)};
}

} // namespace

namespace llaisys::ops::cpu {

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t M, size_t K, size_t N) {
#if defined(_OPENMP)
    const int thread_count = std::min(8, omp_get_max_threads());
#pragma omp parallel for schedule(static) if (M * N >= 512) num_threads(thread_count)
#endif
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
                    sum += bf16ToFloat(in[i * K + k]) * bf16ToFloat(weight[j * K + k]);
                } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(in[i * K + k]) *
                           llaisys::utils::cast<float>(weight[j * K + k]);
                } else {
                    sum += static_cast<float>(in[i * K + k]) *
                           static_cast<float>(weight[j * K + k]);
                }
            }
            if (bias != nullptr) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
                    sum += bf16ToFloat(bias[j]);
                } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(bias[j]);
                } else {
                    sum += static_cast<float>(bias[j]);
                }
            }
            if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
                out[i * N + j] = floatToBf16(sum);
            } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
                out[i * N + j] = llaisys::utils::cast<T>(sum);
            } else {
                out[i * N + j] = static_cast<T>(sum);
            }
        }
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type,
            size_t M, size_t K, size_t N) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            M, K, N);
    case LLAISYS_DTYPE_BF16:
        return linear_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            reinterpret_cast<const llaisys::bf16_t *>(bias),
            M, K, N);
    case LLAISYS_DTYPE_F16:
        return linear_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            reinterpret_cast<const llaisys::fp16_t *>(bias),
            M, K, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
