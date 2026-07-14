import json
import mmap
from ctypes import (
    POINTER,
    Structure,
    addressof,
    byref,
    c_char,
    c_float,
    c_int,
    c_int64,
    c_size_t,
    c_void_p,
)
from pathlib import Path
from typing import Sequence

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DataType, DeviceType, llaisysDeviceType_t, llaisysTensor_t

_DTYPE_MAP = {
    "float": DataType.F32,
    "float32": DataType.F32,
    "fp32": DataType.F32,
    "float16": DataType.F16,
    "fp16": DataType.F16,
    "half": DataType.F16,
    "bfloat16": DataType.BF16,
    "bf16": DataType.BF16,
}

_SAFETENSORS_DTYPE_MAP = {
    DataType.F32: "F32",
    DataType.F16: "F16",
    DataType.BF16: "BF16",
}


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]


_LAYER_WEIGHT_FIELDS = {
    "input_layernorm.weight": "attn_norm_w",
    "self_attn.q_proj.weight": "attn_q_w",
    "self_attn.q_proj.bias": "attn_q_b",
    "self_attn.k_proj.weight": "attn_k_w",
    "self_attn.k_proj.bias": "attn_k_b",
    "self_attn.v_proj.weight": "attn_v_w",
    "self_attn.v_proj.bias": "attn_v_b",
    "self_attn.o_proj.weight": "attn_o_w",
    "post_attention_layernorm.weight": "mlp_norm_w",
    "mlp.gate_proj.weight": "mlp_gate_w",
    "mlp.up_proj.weight": "mlp_up_w",
    "mlp.down_proj.weight": "mlp_down_w",
}


_QWEN2_API_LOADED = False


def load_qwen2_api():
    global _QWEN2_API_LOADED
    if _QWEN2_API_LOADED:
        return

    LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    LIB_LLAISYS.llaisysQwen2ModelCreate.restype = c_void_p
    LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
    LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None
    LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [c_void_p]
    LIB_LLAISYS.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)
    LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [
        c_void_p,
        POINTER(c_int64),
        c_size_t,
    ]
    LIB_LLAISYS.llaisysQwen2ModelInfer.restype = c_int64
    _QWEN2_API_LOADED = True


def parse_dtype(config):
    dtype = str(config.get("torch_dtype", "bfloat16"))
    dtype = dtype.lower().replace("torch.", "")
    return _DTYPE_MAP.get(dtype, DataType.BF16)


def build_weight_map(weights, nlayer):
    weight_map = {
        "model.embed_tokens.weight": weights.in_embed,
        "model.norm.weight": weights.out_norm_w,
        "lm_head.weight": weights.out_embed,
    }
    for layer in range(nlayer):
        prefix = f"model.layers.{layer}."
        for suffix, field in _LAYER_WEIGHT_FIELDS.items():
            weight_map[prefix + suffix] = getattr(weights, field)[layer]
    return weight_map


def load_safetensors_file(file, weight_map, tied_word_embeddings, out_embed):
    loaded = set()
    with open(file, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
        data_offset = 8 + header_size

        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) as mm:
            for name, info in header.items():
                if name == "__metadata__":
                    continue

                dst = weight_map.get(name)
                if dst is None:
                    continue

                expected_dtype = _SAFETENSORS_DTYPE_MAP.get(
                    DataType(LIB_LLAISYS.tensorGetDataType(dst))
                )
                actual_dtype = str(info["dtype"]).upper()
                if expected_dtype != actual_dtype:
                    raise TypeError(
                        f"{name} dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
                    )

                start, _ = info["data_offsets"]
                src = c_char.from_buffer(mm, data_offset + start)
                try:
                    data_ptr = c_void_p(addressof(src))
                    LIB_LLAISYS.tensorLoad(dst, data_ptr)
                    loaded.add(name)
                    if name == "model.embed_tokens.weight" and tied_word_embeddings:
                        LIB_LLAISYS.tensorLoad(out_embed, data_ptr)
                        loaded.add("lm_head.weight")
                finally:
                    del src

    return loaded


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self._model = None
        load_qwen2_api()
        model_path = Path(model_path)

        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        eos_token = config.get("eos_token_id", -1)
        if isinstance(eos_token, (list, tuple)):
            eos_token = eos_token[0] if eos_token else -1

        meta = LlaisysQwen2Meta(
            int(parse_dtype(config)),
            config["num_hidden_layers"],
            hidden_size,
            num_heads,
            num_kv_heads,
            hidden_size // num_heads,
            config["intermediate_size"],
            config.get("max_position_embeddings", config.get("max_sequence_length", 0)),
            config["vocab_size"],
            config.get("rms_norm_eps", 1e-6),
            config.get("rope_theta", 10000.0),
            int(eos_token),
        )

        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            llaisysDeviceType_t(device),
            None,
            c_int(0),
        )
        if not self._model:
            raise RuntimeError("failed to create Qwen2 model")

        self._end_token = int(eos_token)
        self._tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
        self._weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        if not self._weights_ptr:
            raise RuntimeError("failed to get Qwen2 model weights")
        self._weights = self._weights_ptr.contents
        weight_map = build_weight_map(self._weights, config["num_hidden_layers"])
        loaded = set()

        safetensor_files = sorted(model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"no safetensors files found in {model_path}")
        for file in safetensor_files:
            loaded.update(
                load_safetensors_file(
                    file,
                    weight_map,
                    self._tie_word_embeddings,
                    self._weights.out_embed,
                )
            )

        required = set(weight_map)
        for layer in range(config["num_hidden_layers"]):
            prefix = f"model.layers.{layer}.self_attn."
            required.discard(prefix + "q_proj.bias")
            required.discard(prefix + "k_proj.bias")
            required.discard(prefix + "v_proj.bias")
        if self._tie_word_embeddings:
            required.discard("lm_head.weight")
        missing = sorted(required - loaded)
        if missing:
            raise RuntimeError(f"missing Qwen2 weights: {missing[:8]}")

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        output = list(inputs)
        for _ in range(max_new_tokens):
            token_ids = (c_int64 * len(output))(*output)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model,
                    token_ids,
                    c_size_t(len(output)),
                )
            )
            if next_token < 0:
                raise RuntimeError("Qwen2 inference failed")
            output.append(next_token)
            if next_token == self._end_token:
                break

        return output
