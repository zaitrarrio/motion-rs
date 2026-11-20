# Text encoder (`motion-text`)

Kimodo conditions on one 4096-d embedding per prompt from
[LLM2Vec](https://github.com/McGill-NLP/llm2vec) over Meta-Llama-3-8B-Instruct:
bidirectional attention, two LoRA adapters (`-mntp` merged, then
`-mntp-supervised` applied), mean pooling over the prompt tokens.

| piece | file |
|---|---|
| Llama-3 forward, streamed one layer at a time from the mmapped shards | `llama.rs` |
| LoRA adapter loading and merging | `lora.rs` |
| Kimodo's prompt sanitising | `sanitize.rs` |
| the Llama-3 chat framing (`LLAMA3_USER_HEADER`, separator, `LLAMA3_EOT`), byte-level BPE tokenizer, pooling | `lib.rs` |
| precomputed embeddings file (`embeddings` tensor, prompts in the metadata) | `embeddings.rs` |

## Status

The base model is gated on Hugging Face and its weights were not available
where this was written, so **the real encoder has not run here**. What is
verified:

- a tiny random Llama with both adapters against `transformers` + `peft`
  (`scripts/reference/text_fixtures.py`): last hidden state 2.6e-6, pooled
  embedding 5.6e-7 relative;
- the tokenizer against the Python tokenizer on the real `tokenizer.json`
  (exact).

## Use

```bash
motion download llama-3-8b-instruct
```

```bash
motion download llm2vec-mntp
```

```bash
motion download llm2vec-supervised
```

```bash
motion encode --llama-dir weights/Meta-Llama-3-8B-Instruct --adapters weights/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp weights/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised --out prompts.safetensors "A person walks forward" "A person jumps"
```

The first download needs `HF_TOKEN` after accepting Meta's licence. The
embeddings file then serves `motion generate --embeddings` and the studio's
prompt gallery (`motion serve --embeddings`), so the 8B model is needed only
once per prompt. Pass `--llama-dir` and `--adapters` to `motion serve` to
encode any prompt in the studio.

On `--backend wgpu` the encoder needs storage bindings larger than the
WebGPU default of 128 MiB (the adapter's own limit is requested); a buffer
above the adapter's limit is refused with its size.
