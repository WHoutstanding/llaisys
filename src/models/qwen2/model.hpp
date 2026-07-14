#pragma once

#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include "../../llaisys/llaisys_tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace llaisys::models {

class Qwen2Model {
public:
    Qwen2Model(
        const LlaisysQwen2Meta &meta,
        llaisysDeviceType_t device,
        const int *device_ids,
        int ndevice);

    LlaisysQwen2Weights *weights();

    int64_t infer(const int64_t *token_ids, size_t ntoken);

private:
    struct LayerWeights {
        tensor_t attn_norm_w;
        tensor_t attn_q_w;
        tensor_t attn_q_b;
        tensor_t attn_k_w;
        tensor_t attn_k_b;
        tensor_t attn_v_w;
        tensor_t attn_v_b;
        tensor_t attn_o_w;
        tensor_t mlp_norm_w;
        tensor_t mlp_gate_w;
        tensor_t mlp_up_w;
        tensor_t mlp_down_w;
    };

    struct LayerCache {
        tensor_t k;
        tensor_t v;
        size_t length{0};
        size_t capacity{0};
    };

    LlaisysQwen2Meta meta_;
    llaisysDeviceType_t device_;
    int device_id_;
    size_t kv_row_bytes_{0};
    float attention_scale_{0.0f};

    tensor_t in_embed_;
    tensor_t out_embed_;
    tensor_t out_norm_w_;
    std::vector<LayerWeights> layers_;
    std::vector<LayerCache> cache_;

    tensor_t token_ids_;
    tensor_t position_ids_;
    tensor_t hidden_;
    tensor_t residual_;
    tensor_t norm_;
    tensor_t q_linear_;
    tensor_t q_;
    tensor_t k_linear_;
    tensor_t k_;
    tensor_t v_linear_;
    tensor_t v_;
    tensor_t attn_;
    tensor_t attn_flat_;
    tensor_t attn_proj_;
    tensor_t mlp_gate_;
    tensor_t mlp_up_;
    tensor_t mlp_act_;
    tensor_t mlp_out_;
    tensor_t logits_;
    tensor_t max_idx_;
    tensor_t max_val_;

    std::vector<int64_t> cached_tokens_;
    bool has_last_next_token_{false};
    int64_t last_next_token_{-1};

    LlaisysQwen2Weights c_weights_{};

    std::vector<std::unique_ptr<LlaisysTensor>> owned_handles_;
    std::vector<llaisysTensor_t> attn_norm_w_handles_;
    std::vector<llaisysTensor_t> attn_q_w_handles_;
    std::vector<llaisysTensor_t> attn_q_b_handles_;
    std::vector<llaisysTensor_t> attn_k_w_handles_;
    std::vector<llaisysTensor_t> attn_k_b_handles_;
    std::vector<llaisysTensor_t> attn_v_w_handles_;
    std::vector<llaisysTensor_t> attn_v_b_handles_;
    std::vector<llaisysTensor_t> attn_o_w_handles_;
    std::vector<llaisysTensor_t> mlp_norm_w_handles_;
    std::vector<llaisysTensor_t> mlp_gate_w_handles_;
    std::vector<llaisysTensor_t> mlp_up_w_handles_;
    std::vector<llaisysTensor_t> mlp_down_w_handles_;

    tensor_t makeWeight(const std::vector<size_t> &shape);
    tensor_t makeActivation(const std::vector<size_t> &shape, llaisysDataType_t dtype);
    llaisysTensor_t makeHandle(tensor_t tensor);
    void buildWeightHandles();
    void zeroTensor(const tensor_t &tensor);

    void resetCache();
    void ensureCacheCapacity(LayerCache &cache, size_t total_len);
    void reserveCache(size_t total_len);
    void appendCache(LayerCache &cache, size_t position);
    void processToken(int64_t token_id, size_t position, bool compute_logits);
    void processLayer(const LayerWeights &layer, LayerCache &cache, size_t position);
    int64_t computeNextToken();
};

} // namespace llaisys::models
