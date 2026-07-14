#include "model.hpp"

#include "../../utils.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace llaisys::models {

Qwen2Model::Qwen2Model(
    const LlaisysQwen2Meta &meta,
    llaisysDeviceType_t device,
    const int *device_ids,
    int ndevice)
    : meta_(meta),
      device_(device),
      device_id_(device_ids != nullptr && ndevice > 0 ? device_ids[0] : 0) {
    if (device_ != LLAISYS_DEVICE_CPU) {
        throw std::runtime_error("Qwen2Model currently supports CPU inference only");
    }
    if (meta_.nlayer == 0 || meta_.hs == 0 || meta_.nh == 0 || meta_.nkvh == 0 ||
        meta_.dh == 0 || meta_.di == 0 || meta_.maxseq == 0 || meta_.voc == 0) {
        throw std::invalid_argument("invalid Qwen2 metadata");
    }
    if (meta_.nh % meta_.nkvh != 0 || meta_.nh * meta_.dh != meta_.hs) {
        throw std::invalid_argument("invalid Qwen2 attention dimensions");
    }

    const size_t q_size = meta_.nh * meta_.dh;
    const size_t kv_size = meta_.nkvh * meta_.dh;
    kv_row_bytes_ = kv_size * utils::dsize(meta_.dtype);
    attention_scale_ = 1.0f / std::sqrt(static_cast<float>(meta_.dh));

    in_embed_ = makeWeight({meta_.voc, meta_.hs});
    out_embed_ = makeWeight({meta_.voc, meta_.hs});
    out_norm_w_ = makeWeight({meta_.hs});

    layers_.resize(meta_.nlayer);
    cache_.resize(meta_.nlayer);
    for (size_t i = 0; i < meta_.nlayer; ++i) {
        auto &layer = layers_[i];
        layer.attn_norm_w = makeWeight({meta_.hs});
        layer.attn_q_w = makeWeight({q_size, meta_.hs});
        layer.attn_q_b = makeWeight({q_size});
        layer.attn_k_w = makeWeight({kv_size, meta_.hs});
        layer.attn_k_b = makeWeight({kv_size});
        layer.attn_v_w = makeWeight({kv_size, meta_.hs});
        layer.attn_v_b = makeWeight({kv_size});
        layer.attn_o_w = makeWeight({meta_.hs, q_size});
        layer.mlp_norm_w = makeWeight({meta_.hs});
        layer.mlp_gate_w = makeWeight({meta_.di, meta_.hs});
        layer.mlp_up_w = makeWeight({meta_.di, meta_.hs});
        layer.mlp_down_w = makeWeight({meta_.hs, meta_.di});

        zeroTensor(layer.attn_q_b);
        zeroTensor(layer.attn_k_b);
        zeroTensor(layer.attn_v_b);
    }

    token_ids_ = makeActivation({1}, LLAISYS_DTYPE_I64);
    position_ids_ = makeActivation({1}, LLAISYS_DTYPE_I64);
    hidden_ = makeActivation({1, meta_.hs}, meta_.dtype);
    residual_ = makeActivation({1, meta_.hs}, meta_.dtype);
    norm_ = makeActivation({1, meta_.hs}, meta_.dtype);
    q_linear_ = makeActivation({1, q_size}, meta_.dtype);
    q_ = q_linear_->view({1, meta_.nh, meta_.dh});
    k_linear_ = makeActivation({1, kv_size}, meta_.dtype);
    k_ = k_linear_->view({1, meta_.nkvh, meta_.dh});
    v_linear_ = makeActivation({1, kv_size}, meta_.dtype);
    v_ = v_linear_->view({1, meta_.nkvh, meta_.dh});
    attn_ = makeActivation({1, meta_.nh, meta_.dh}, meta_.dtype);
    attn_flat_ = attn_->view({1, q_size});
    attn_proj_ = makeActivation({1, meta_.hs}, meta_.dtype);
    mlp_gate_ = makeActivation({1, meta_.di}, meta_.dtype);
    mlp_up_ = makeActivation({1, meta_.di}, meta_.dtype);
    mlp_act_ = makeActivation({1, meta_.di}, meta_.dtype);
    mlp_out_ = makeActivation({1, meta_.hs}, meta_.dtype);
    logits_ = makeActivation({1, meta_.voc}, meta_.dtype);
    max_idx_ = makeActivation({1}, LLAISYS_DTYPE_I64);
    max_val_ = makeActivation({1}, meta_.dtype);

    buildWeightHandles();
}

LlaisysQwen2Weights *Qwen2Model::weights() {
    return &c_weights_;
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    if (token_ids == nullptr || ntoken == 0) {
        throw std::invalid_argument("Qwen2 infer requires at least one token");
    }

    size_t prefix = 0;
    while (prefix < ntoken && prefix < cached_tokens_.size() &&
           cached_tokens_[prefix] == token_ids[prefix]) {
        ++prefix;
    }
    if (prefix < cached_tokens_.size()) {
        resetCache();
    }

    if (cached_tokens_.size() < ntoken) {
        reserveCache(ntoken);
    }

    for (size_t i = cached_tokens_.size(); i < ntoken; ++i) {
        processToken(token_ids[i], i, i + 1 == ntoken);
        cached_tokens_.push_back(token_ids[i]);
    }

    if (!has_last_next_token_) {
        throw std::runtime_error("Qwen2 infer did not produce logits");
    }
    return last_next_token_;
}

tensor_t Qwen2Model::makeWeight(const std::vector<size_t> &shape) {
    return makeActivation(shape, meta_.dtype);
}

tensor_t Qwen2Model::makeActivation(const std::vector<size_t> &shape, llaisysDataType_t dtype) {
    return Tensor::create(shape, dtype, device_, device_id_);
}

llaisysTensor_t Qwen2Model::makeHandle(tensor_t tensor) {
    owned_handles_.push_back(std::make_unique<LlaisysTensor>(LlaisysTensor{std::move(tensor)}));
    return owned_handles_.back().get();
}

void Qwen2Model::buildWeightHandles() {
    owned_handles_.reserve(3 + meta_.nlayer * 12);

    c_weights_.in_embed = makeHandle(in_embed_);
    c_weights_.out_embed = makeHandle(out_embed_);
    c_weights_.out_norm_w = makeHandle(out_norm_w_);

    attn_norm_w_handles_.resize(meta_.nlayer);
    attn_q_w_handles_.resize(meta_.nlayer);
    attn_q_b_handles_.resize(meta_.nlayer);
    attn_k_w_handles_.resize(meta_.nlayer);
    attn_k_b_handles_.resize(meta_.nlayer);
    attn_v_w_handles_.resize(meta_.nlayer);
    attn_v_b_handles_.resize(meta_.nlayer);
    attn_o_w_handles_.resize(meta_.nlayer);
    mlp_norm_w_handles_.resize(meta_.nlayer);
    mlp_gate_w_handles_.resize(meta_.nlayer);
    mlp_up_w_handles_.resize(meta_.nlayer);
    mlp_down_w_handles_.resize(meta_.nlayer);

    for (size_t i = 0; i < meta_.nlayer; ++i) {
        const auto &layer = layers_[i];
        attn_norm_w_handles_[i] = makeHandle(layer.attn_norm_w);
        attn_q_w_handles_[i] = makeHandle(layer.attn_q_w);
        attn_q_b_handles_[i] = makeHandle(layer.attn_q_b);
        attn_k_w_handles_[i] = makeHandle(layer.attn_k_w);
        attn_k_b_handles_[i] = makeHandle(layer.attn_k_b);
        attn_v_w_handles_[i] = makeHandle(layer.attn_v_w);
        attn_v_b_handles_[i] = makeHandle(layer.attn_v_b);
        attn_o_w_handles_[i] = makeHandle(layer.attn_o_w);
        mlp_norm_w_handles_[i] = makeHandle(layer.mlp_norm_w);
        mlp_gate_w_handles_[i] = makeHandle(layer.mlp_gate_w);
        mlp_up_w_handles_[i] = makeHandle(layer.mlp_up_w);
        mlp_down_w_handles_[i] = makeHandle(layer.mlp_down_w);
    }

    c_weights_.attn_norm_w = attn_norm_w_handles_.data();
    c_weights_.attn_q_w = attn_q_w_handles_.data();
    c_weights_.attn_q_b = attn_q_b_handles_.data();
    c_weights_.attn_k_w = attn_k_w_handles_.data();
    c_weights_.attn_k_b = attn_k_b_handles_.data();
    c_weights_.attn_v_w = attn_v_w_handles_.data();
    c_weights_.attn_v_b = attn_v_b_handles_.data();
    c_weights_.attn_o_w = attn_o_w_handles_.data();
    c_weights_.mlp_norm_w = mlp_norm_w_handles_.data();
    c_weights_.mlp_gate_w = mlp_gate_w_handles_.data();
    c_weights_.mlp_up_w = mlp_up_w_handles_.data();
    c_weights_.mlp_down_w = mlp_down_w_handles_.data();
}

void Qwen2Model::zeroTensor(const tensor_t &tensor) {
    std::memset(tensor->data(), 0, tensor->numel() * tensor->elementSize());
}

void Qwen2Model::resetCache() {
    cached_tokens_.clear();
    for (auto &layer_cache : cache_) {
        layer_cache.length = 0;
    }
    has_last_next_token_ = false;
    last_next_token_ = -1;
}

void Qwen2Model::ensureCacheCapacity(LayerCache &cache, size_t total_len) {
    if (total_len <= cache.capacity) {
        return;
    }

    const size_t new_capacity = std::min(
        meta_.maxseq,
        std::max(total_len, cache.capacity == 0 ? size_t(16) : cache.capacity * 2));
    auto new_k = makeActivation({new_capacity, meta_.nkvh, meta_.dh}, meta_.dtype);
    auto new_v = makeActivation({new_capacity, meta_.nkvh, meta_.dh}, meta_.dtype);
    if (cache.length != 0) {
        std::memcpy(new_k->data(), cache.k->data(), cache.length * kv_row_bytes_);
        std::memcpy(new_v->data(), cache.v->data(), cache.length * kv_row_bytes_);
    }
    cache.k = std::move(new_k);
    cache.v = std::move(new_v);
    cache.capacity = new_capacity;
}

void Qwen2Model::reserveCache(size_t total_len) {
    for (auto &layer_cache : cache_) {
        ensureCacheCapacity(layer_cache, total_len);
    }
}

void Qwen2Model::appendCache(LayerCache &cache, size_t position) {
    if (cache.length != position || cache.capacity <= position) {
        throw std::runtime_error("Qwen2 KV cache capacity mismatch");
    }

    std::memcpy(cache.k->data() + position * kv_row_bytes_, k_->data(), kv_row_bytes_);
    std::memcpy(cache.v->data() + position * kv_row_bytes_, v_->data(), kv_row_bytes_);
    cache.length = position + 1;
}

void Qwen2Model::processToken(int64_t token_id, size_t position, bool compute_logits) {
    if (token_id < 0 || static_cast<size_t>(token_id) >= meta_.voc) {
        throw std::out_of_range("Qwen2 token id out of vocabulary range");
    }
    if (position >= meta_.maxseq) {
        throw std::out_of_range("Qwen2 sequence length exceeds maxseq");
    }

    token_ids_->load(&token_id);
    const int64_t position_id = static_cast<int64_t>(position);
    position_ids_->load(&position_id);
    ops::embedding(hidden_, token_ids_, in_embed_);

    for (size_t layer_idx = 0; layer_idx < meta_.nlayer; ++layer_idx) {
        processLayer(layers_[layer_idx], cache_[layer_idx], position);
    }

    if (compute_logits) {
        last_next_token_ = computeNextToken();
        has_last_next_token_ = true;
    }
}

void Qwen2Model::processLayer(const LayerWeights &layer, LayerCache &cache, size_t position) {
    ops::rms_norm(norm_, hidden_, layer.attn_norm_w, meta_.epsilon);
    ops::linear(q_linear_, norm_, layer.attn_q_w, layer.attn_q_b);
    ops::linear(k_linear_, norm_, layer.attn_k_w, layer.attn_k_b);
    ops::linear(v_linear_, norm_, layer.attn_v_w, layer.attn_v_b);
    ops::rope(q_, q_, position_ids_, meta_.theta);
    ops::rope(k_, k_, position_ids_, meta_.theta);

    appendCache(cache, position);
    const size_t total_len = position + 1;
    ops::self_attention(
        attn_,
        q_,
        cache.k->slice(0, 0, total_len),
        cache.v->slice(0, 0, total_len),
        attention_scale_);
    ops::linear(attn_proj_, attn_flat_, layer.attn_o_w, nullptr);
    ops::add(residual_, hidden_, attn_proj_);

    ops::rms_norm(norm_, residual_, layer.mlp_norm_w, meta_.epsilon);
    ops::linear(mlp_gate_, norm_, layer.mlp_gate_w, nullptr);
    ops::linear(mlp_up_, norm_, layer.mlp_up_w, nullptr);
    ops::swiglu(mlp_act_, mlp_gate_, mlp_up_);
    ops::linear(mlp_out_, mlp_act_, layer.mlp_down_w, nullptr);
    ops::add(hidden_, residual_, mlp_out_);
}

int64_t Qwen2Model::computeNextToken() {
    ops::rms_norm(norm_, hidden_, out_norm_w_, meta_.epsilon);
    ops::linear(logits_, norm_, out_embed_, nullptr);
    ops::argmax(max_idx_, max_val_, logits_);
    return *reinterpret_cast<const int64_t *>(max_idx_->data());
}

} // namespace llaisys::models
