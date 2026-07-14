#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t nindex, size_t embedding_dim) {
    for (size_t i = 0; i < nindex; ++i) {
        std::memcpy(
            out + i * embedding_dim,
            weight + static_cast<size_t>(index[i]) * embedding_dim,
            embedding_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t nindex,
    size_t embedding_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight),
            nindex,
            embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            nindex,
            embedding_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            nindex,
            embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
