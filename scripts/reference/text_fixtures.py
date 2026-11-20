"""PyTorch reference for motion-text: a tiny bidirectional Llama with two LoRA
adapters (one merged into the base, one applied on top, as LLM2Vec loads
McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp and -mntp-supervised) and a
mean-pooled output.

Writes crates/motion-text/tests/fixtures/tiny_llama/:
  config.json, model.safetensors            the random base (HF key names, `model.` prefix)
  mntp/adapter_config.json, adapter_model.safetensors        adapter 1 (merged)
  supervised/adapter_config.json, adapter_model.safetensors  adapter 2 (applied)
  expected.safetensors   input_ids [S], pooled_mask [S], expected_pooled [H],
                         expected_last_hidden [S, H] (after the final norm)

Run from the repository root: python3 -I scripts/reference/text_fixtures.py
Optionally MOTION_LLAMA3_TOKENIZER=<path to tokenizer.json> adds tokenizer
expectations (prompt strings → ids and pooled spans) to tokens.json.
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

torch.manual_seed(0)
out = os.path.join("crates", "motion-text", "tests", "fixtures", "tiny_llama")
os.makedirs(os.path.join(out, "mntp"), exist_ok=True)
os.makedirs(os.path.join(out, "supervised"), exist_ok=True)

cfg = LlamaConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=100,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    max_position_embeddings=64,
    attention_bias=False,
    tie_word_embeddings=False,
)
cfg._attn_implementation = "eager"
model = LlamaModel(cfg).eval()
# Perturb the init so nothing is degenerate.
with torch.no_grad():
    for p in model.parameters():
        p.add_(0.05 * torch.randn_like(p))

base_state = {f"model.{k}": v.detach().clone().contiguous() for k, v in model.state_dict().items()}
save_file(base_state, os.path.join(out, "model.safetensors"))
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(
        {
            "_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
            "architectures": ["LlamaModel"],
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "rope_scaling": None,
            "max_position_embeddings": 64,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "torch_dtype": "float32",
        },
        f,
        indent=1,
    )

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def random_adapter(model, name, r, alpha, use_rslora):
    lcfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules=TARGETS, bias="none", use_rslora=use_rslora)
    pm = get_peft_model(model, lcfg)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_" in n:
                p.copy_(0.3 * torch.randn_like(p))
    sd = {k: v.detach().clone().contiguous() for k, v in get_peft_model_state_dict(pm).items()}
    save_file(sd, os.path.join(out, name, "adapter_model.safetensors"))
    with open(os.path.join(out, name, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "peft_type": "LORA",
                "r": r,
                "lora_alpha": alpha,
                "lora_dropout": 0.05,
                "use_rslora": use_rslora,
                "target_modules": TARGETS,
                "bias": "none",
                "base_model_name_or_path": "tiny",
            },
            f,
            indent=1,
        )
    return pm


pm1 = random_adapter(model, "mntp", r=4, alpha=8, use_rslora=False)
model = pm1.merge_and_unload()
pm2 = random_adapter(model, "supervised", r=3, alpha=6, use_rslora=True)
pm2.eval()

S = 9
input_ids = torch.tensor([[0, 5, 17, 42, 3, 99, 8, 61, 1]])
# Bidirectional: a 4-D additive zero mask lets every token attend to every token.
full_mask = torch.zeros(1, 1, S, S)
with torch.no_grad():
    hidden = pm2(input_ids=input_ids, attention_mask=full_mask).last_hidden_state[0]
    # Sanity: the first token must see later tokens (a causal run would not).
    first_alone = pm2(input_ids=input_ids[:, :1], attention_mask=torch.zeros(1, 1, 1, 1)).last_hidden_state[0, 0]
    assert not torch.allclose(first_alone, hidden[0], atol=1e-5), "attention is still causal"
pooled_mask = torch.zeros(S)
pooled_mask[-5:] = 1.0  # the last five tokens, as LLM2Vec's embed_mask does
pooled = hidden[pooled_mask.bool()].mean(dim=0)
save_file(
    {
        "input_ids": input_ids[0].to(torch.int64),
        "pooled_mask": pooled_mask,
        "expected_last_hidden": hidden.contiguous(),
        "expected_pooled": pooled.contiguous(),
    },
    os.path.join(out, "expected.safetensors"),
)
print("wrote", out, "pooled[:4] =", pooled[:4].tolist())

tok_path = os.environ.get("MOTION_LLAMA3_TOKENIZER")
if tok_path:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    cases = []
    for text in ["A person walks forward.", "Someone jumps twice, then waves.", "Jump!"]:
        # LLM2Vec with batch_size 1, no instruction, Llama-3-Instruct template.
        inner = "!@#$%^&*()" + text
        prepared = "<|start_header_id|>user<|end_header_id|>\n\n" + inner.strip() + "<|eot_id|>"
        parts = prepared.split("!@#$%^&*()")
        original = "".join(parts)
        ids = tok.encode(original, add_special_tokens=True).ids
        embed_ids = tok.encode(parts[1], add_special_tokens=False).ids
        cases.append({"text": text, "ids": ids, "pooled_count": len(embed_ids)})
    with open(os.path.join(out, "tokens.json"), "w") as f:
        json.dump(cases, f, indent=1)
    print("wrote tokens.json", [(c["text"], len(c["ids"]), c["pooled_count"]) for c in cases])
