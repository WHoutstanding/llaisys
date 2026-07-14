#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     float scale, size_t seqlen, size_t nhead, size_t dv,
                     size_t total_len, size_t nkvhead, size_t d) {
    size_t nrep = nhead / nkvhead;  // q heads per kv head (GQA)

    for (size_t i = 0; i < seqlen; i++) {
        for (size_t h = 0; h < nhead; h++) {
            size_t kv_h = h / nrep;

            std::vector<float> scores(total_len);
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t j = 0; j < total_len; j++) {
                float dot = 0.0f;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    for (size_t d_idx = 0; d_idx < d; d_idx++) {
                        float qv = llaisys::utils::cast<float>(q[i * nhead * d + h * d + d_idx]);
                        float kv = llaisys::utils::cast<float>(k[j * nkvhead * d + kv_h * d + d_idx]);
                        dot += qv * kv;
                    }
                } else {
                    for (size_t d_idx = 0; d_idx < d; d_idx++) {
                        dot += static_cast<float>(q[i * nhead * d + h * d + d_idx]) *
                               static_cast<float>(k[j * nkvhead * d + kv_h * d + d_idx]);
                    }
                }
                dot *= scale;

                size_t cur_pos = total_len - seqlen + i;
                if (j > cur_pos) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                } else {
                    scores[j] = dot;
                }
                if (scores[j] > max_score) {
                    max_score = scores[j];
                }
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < total_len; j++) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (size_t j = 0; j < total_len; j++) {
                scores[j] /= sum_exp;
            }

            for (size_t dv_idx = 0; dv_idx < dv; dv_idx++) {
                float sum = 0.0f;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    for (size_t j = 0; j < total_len; j++) {
                        sum += static_cast<float>(scores[j]) *
                               llaisys::utils::cast<float>(v[j * nkvhead * dv + kv_h * dv + dv_idx]);
                    }
                    attn_val[i * nhead * dv + h * dv + dv_idx] = llaisys::utils::cast<T>(sum);
                } else {
                    for (size_t j = 0; j < total_len; j++) {
                        sum += scores[j] *
                               static_cast<float>(v[j * nkvhead * dv + kv_h * dv + dv_idx]);
                    }
                    attn_val[i * nhead * dv + h * dv + dv_idx] = static_cast<T>(sum);
                }
            }
        }
    }
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, float scale, llaisysDataType_t type,
                    size_t seqlen, size_t nhead, size_t dv,
                    size_t total_len, size_t nkvhead, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_<float>(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            scale, seqlen, nhead, dv, total_len, nkvhead, d);
    case LLAISYS_DTYPE_BF16:
        return self_attention_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(attn_val),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            scale, seqlen, nhead, dv, total_len, nkvhead, d);
    case LLAISYS_DTYPE_F16:
        return self_attention_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(attn_val),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            scale, seqlen, nhead, dv, total_len, nkvhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
