//! PyTorch parity of the bidirectional, LoRA-merged Llama on a tiny random model
//! (`scripts/reference/text_fixtures.py`: transformers + peft, one adapter merged,
//! one applied), and the tokenizer against the Python tokenizer (`tokens.json`,
//! needs the real `tokenizer.json`: `MOTION_LLAMA3_TOKENIZER`).

use motion_core::weights::{SafetensorsDir, WeightSource};
use motion_cpu::Cpu;
use motion_text::llama::{Llama, LlamaConfig};
use motion_text::lora::Adapter;
use motion_text::{mean_last, TextEncoder, Tokens};
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_llama")
}

fn tiny() -> Llama {
    let d = dir();
    let cfg = LlamaConfig::load(&d.join("config.json")).unwrap();
    let ads = vec![Adapter::load(&d.join("mntp")).unwrap(), Adapter::load(&d.join("supervised")).unwrap()];
    Llama::open(&[d.join("model.safetensors")], cfg, ads).unwrap()
}

#[test]
fn tiny_llama_matches_pytorch() {
    let m = tiny();
    let exp = SafetensorsDir::open(&dir().join("expected.safetensors")).unwrap();
    let ids: Vec<u32> = exp.get("input_ids", &[9]).unwrap().iter().map(|&v| v as u32).collect();
    let hidden = m.forward(&Cpu, &ids).unwrap();
    let want = exp.get("expected_last_hidden", &[9, 32]).unwrap();
    let e = motion_core::tensor::rel_err(&hidden, &want);
    println!("last hidden rel {e:.2e}");
    assert!(e < 1e-4);
    let pooled = mean_last(&hidden, 32, 5);
    let e = motion_core::tensor::rel_err(&pooled, &exp.get("expected_pooled", &[32]).unwrap());
    println!("pooled rel {e:.2e}");
    assert!(e < 1e-4);
}

#[test]
#[ignore = "needs the LLM2Vec / Llama-3 tokenizer.json (MOTION_LLAMA3_TOKENIZER)"]
fn tokenizer_matches_python() {
    let Ok(tok) = std::env::var("MOTION_LLAMA3_TOKENIZER") else { return };
    let tokenizer = tokenizers::Tokenizer::from_file(&tok).unwrap();
    let enc = TextEncoder::from_parts(tiny(), tokenizer);
    let cases: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir().join("tokens.json")).unwrap()).unwrap();
    for c in cases.as_array().unwrap() {
        let want = Tokens {
            ids: c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect(),
            pooled: c["pooled_count"].as_u64().unwrap() as usize,
        };
        assert_eq!(enc.tokenize(c["text"].as_str().unwrap()).unwrap(), want, "{}", c["text"]);
    }
}
