#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: indices must be int64.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding: weight must be rank 2.");
    CHECK_ARGUMENT(
        out->numel() == index->numel() * weight->shape()[1],
        "Embedding: output shape is incompatible with indices and weight.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            weight->dtype(),
            index->numel(),
            weight->shape()[1]);
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::embedding(
                out->data(),
                index->data(),
                weight->data(),
                weight->dtype(),
                index->numel(),
                weight->shape()[1]);
    #ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            TO_BE_IMPLEMENTED();
            return;
    #endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }    
}
} // namespace llaisys::ops
