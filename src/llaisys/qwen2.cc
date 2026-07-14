#include "llaisys/models/qwen2.h"
#include "../models/qwen2/model.hpp"

#include <memory>
#include <stdexcept>

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::Qwen2Model> impl;
};

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {

        if (meta == nullptr) {
            return nullptr;
        }

        auto model = new LlaisysQwen2Model{
            std::make_unique<llaisys::models::Qwen2Model>(
                *meta, device, device_ids, ndevice)
        };

        return model;
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
        delete model;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        if (model == nullptr) {
            return nullptr;
        }

        return model->impl->weights();
    }

    int64_t llaisysQwen2ModelInfer(
        LlaisysQwen2Model *model,
        int64_t *token_ids,
        size_t ntoken) {
        if (model == nullptr || token_ids == nullptr || ntoken == 0) {
            return -1;
        }

        return model->impl->infer(token_ids, ntoken);
    }
}
