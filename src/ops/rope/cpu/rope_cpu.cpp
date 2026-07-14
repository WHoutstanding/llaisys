#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

namespace llaisys::ops::cpu {

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           float theta, size_t seqlen, size_t nhead, size_t d) {
    size_t half_d = d / 2;

    for (size_t i = 0; i < seqlen; i++) {
        int64_t p = pos_ids[i];
        for (size_t h = 0; h < nhead; h++) {
            for (size_t j = 0; j < half_d; j++) {
                float freq = static_cast<float>(p) /
                             std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(d));
                float cos_phi = std::cos(freq);
                float sin_phi = std::sin(freq);

                size_t idx_a = i * nhead * d + h * d + j;
                size_t idx_b = i * nhead * d + h * d + j + half_d;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    float a = llaisys::utils::cast<float>(in[idx_a]);
                    float b = llaisys::utils::cast<float>(in[idx_b]);
                    out[idx_a] = llaisys::utils::cast<T>(a * cos_phi - b * sin_phi);
                    out[idx_b] = llaisys::utils::cast<T>(b * cos_phi + a * sin_phi);
                } else {
                    float a = static_cast<float>(in[idx_a]);
                    float b = static_cast<float>(in[idx_b]);
                    out[idx_a] = static_cast<T>(a * cos_phi - b * sin_phi);
                    out[idx_b] = static_cast<T>(b * cos_phi + a * sin_phi);
                }
            }
        }
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, llaisysDataType_t type,
          size_t seqlen, size_t nhead, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
