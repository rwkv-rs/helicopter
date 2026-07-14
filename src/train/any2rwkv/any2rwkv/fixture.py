from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def tiny_qwen35_config(*, layers: int = 60, moe: bool = True) -> dict[str, object]:
    layer_types = ["full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(layers)]
    return {
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
        "architectures": ["Qwen3_5MoeForCausalLM" if moe else "Qwen3_5ForCausalLM"],
        "num_hidden_layers": layers,
        "layer_types": layer_types,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "max_position_embeddings": 4096,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_conv_kernel_dim": 4,
        "num_experts": 4 if moe else 0,
        "num_experts_per_tok": 2 if moe else 0,
        "norm_topk_prob": True,
        "router_aux_loss_coef": 0.001,
        "mtp_num_hidden_layers": 1,
        "vocab_size": 64,
        "rope_theta": 1_000_000.0,
        "partial_rotary_factor": 0.5,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 0.5,
        },
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }


def tiny_state_dict(config: dict[str, object], *, seed: int = 20260714) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    hidden = int(config["hidden_size"])
    vocab = int(config["vocab_size"])
    intermediate = int(config["intermediate_size"])
    key_dim = int(config["linear_num_key_heads"]) * int(config["linear_key_head_dim"])
    value_dim = int(config["linear_num_value_heads"]) * int(config["linear_value_head_dim"])
    value_heads = int(config["linear_num_value_heads"])
    value_head_dim = int(config["linear_value_head_dim"])
    attention_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])

    def weight(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * 0.02

    result = {
        "model.embed_tokens.weight": weight(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": weight(vocab, hidden),
        "mtp.fc.weight": weight(hidden, hidden * 2),
        "mtp.norm.weight": torch.ones(hidden),
        "mtp.pre_fc_norm_hidden.weight": torch.ones(hidden),
        "mtp.pre_fc_norm_embedding.weight": torch.ones(hidden),
    }

    def add_mlp(prefix: str) -> None:
        if int(config.get("num_experts", 0)):
            experts = int(config["num_experts"])
            moe_intermediate = int(config["moe_intermediate_size"])
            shared_intermediate = int(config["shared_expert_intermediate_size"])
            result[f"{prefix}.gate.weight"] = weight(experts, hidden)
            result[f"{prefix}.experts.gate_up_proj"] = weight(experts, moe_intermediate * 2, hidden)
            result[f"{prefix}.experts.down_proj"] = weight(experts, hidden, moe_intermediate)
            result[f"{prefix}.shared_expert.gate_proj.weight"] = weight(shared_intermediate, hidden)
            result[f"{prefix}.shared_expert.up_proj.weight"] = weight(shared_intermediate, hidden)
            result[f"{prefix}.shared_expert.down_proj.weight"] = weight(hidden, shared_intermediate)
            result[f"{prefix}.shared_expert_gate.weight"] = weight(1, hidden)
        else:
            result[f"{prefix}.gate_proj.weight"] = weight(intermediate, hidden)
            result[f"{prefix}.up_proj.weight"] = weight(intermediate, hidden)
            result[f"{prefix}.down_proj.weight"] = weight(hidden, intermediate)

    def add_full_attention(prefix: str) -> None:
        result[f"{prefix}.q_proj.weight"] = weight(attention_heads * head_dim * 2, hidden)
        result[f"{prefix}.k_proj.weight"] = weight(kv_heads * head_dim, hidden)
        result[f"{prefix}.v_proj.weight"] = weight(kv_heads * head_dim, hidden)
        result[f"{prefix}.o_proj.weight"] = weight(hidden, attention_heads * head_dim)
        result[f"{prefix}.q_norm.weight"] = torch.ones(head_dim)
        result[f"{prefix}.k_norm.weight"] = torch.ones(head_dim)

    for index, kind in enumerate(config["layer_types"]):
        prefix = f"model.layers.{index}"
        result[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden)
        result[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden)
        if kind == "linear_attention":
            mixer = f"{prefix}.linear_attn"
            result[f"{mixer}.in_proj_qkv.weight"] = weight(key_dim * 2 + value_dim, hidden)
            result[f"{mixer}.in_proj_z.weight"] = weight(value_dim, hidden)
            result[f"{mixer}.in_proj_b.weight"] = weight(value_heads, hidden)
            result[f"{mixer}.in_proj_a.weight"] = weight(value_heads, hidden)
            result[f"{mixer}.conv1d.weight"] = weight(key_dim * 2 + value_dim, 1, int(config["linear_conv_kernel_dim"]))
            result[f"{mixer}.dt_bias"] = torch.ones(value_heads)
            result[f"{mixer}.A_log"] = torch.zeros(value_heads)
            result[f"{mixer}.norm.weight"] = torch.ones(value_head_dim)
            result[f"{mixer}.out_proj.weight"] = weight(hidden, value_dim)
        else:
            add_full_attention(f"{prefix}.self_attn")
        add_mlp(f"{prefix}.mlp")

    mtp_prefix = "mtp.layers.0"
    result[f"{mtp_prefix}.input_layernorm.weight"] = torch.ones(hidden)
    result[f"{mtp_prefix}.post_attention_layernorm.weight"] = torch.ones(hidden)
    add_full_attention(f"{mtp_prefix}.self_attn")
    add_mlp(f"{mtp_prefix}.mlp")
    return result


def write_fixture(path: Path, *, layers: int = 60, moe: bool = True, seed: int = 20260714) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = tiny_qwen35_config(layers=layers, moe=moe)
    tensors = tiny_state_dict(config, seed=seed)
    (path / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    vocabulary = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    vocabulary.update({f"token_{index}": index for index in range(3, int(config["vocab_size"]))})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "model_max_length": 4096,
                "tokenizer_class": "PreTrainedTokenizerFast",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "eos_token": "<eos>",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "special_tokens_map.json").write_text(
        json.dumps({"pad_token": "<pad>", "unk_token": "<unk>", "eos_token": "<eos>"}) + "\n",
        encoding="utf-8",
    )
    (path / "generation_config.json").write_text(json.dumps({"do_sample": False, "eos_token_id": 2}) + "\n", encoding="utf-8")
    save_file(tensors, path / "model.safetensors")
    return path
