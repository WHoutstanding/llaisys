#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"


namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(
        in->ndim() == 2 && out->ndim() == 2 && weight->ndim() == 2,
        "Linear: tensors must be rank 2.");

    const size_t M = out->shape()[0], N = out->shape()[1];
    const size_t K = in->shape()[1];
    CHECK_ARGUMENT(
        in->shape()[0] == M && weight->shape()[0] == N && weight->shape()[1] == K,
        "Linear: incompatible shapes.");
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        CHECK_ARGUMENT(
            bias->ndim() == 1 && bias->shape()[0] == N,
            "Linear: invalid bias shape.");
    }

    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, in->dtype(), M, K, N);
    }

    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::linear(
                out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, in->dtype(), M, K, N);
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
