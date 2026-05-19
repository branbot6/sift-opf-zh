# sift-opf-zh

> LoRA fine-tune of **openai/privacy-filter** on Chinese PII detection — for the [Sift](https://github.com/) on-device privacy filter project.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Base](https://img.shields.io/badge/base-openai%2Fprivacy--filter-blueviolet)](https://huggingface.co/openai/privacy-filter)
[![Macro F1](https://img.shields.io/badge/Macro_F1-0.80-brightgreen)](#evaluation)

## TL;DR

A 2.3 MB LoRA adapter that lifts `openai/privacy-filter` from **F1 0.67 → 0.80** (macro) on Chinese PII detection across 8 entity classes — `private_email / phone / person / address / url / date / secret / account_number`.

Trained on **12,126 hand-curated + synthetic samples**, evaluated on an independent **1,816-sample gold benchmark** with zero training-test overlap and strict annotation rules (e.g. addresses must contain digits).

## Key results

| Class | F1 | Recommended use |
|---|---:|---|
| private_email | 0.97 | ✅ Production-ready |
| private_phone | 0.97 | ✅ Production-ready |
| private_person | 0.86 | ✅ Production-ready |
| private_date | 0.78 | ⚠️ Add regex post-filter |
| secret | 0.78 | ⚠️ Add regex post-filter (sk-*, AKIA*, ghp_*) |
| account_number | 0.74 | ⚠️ Add Luhn / ID-card checksum |
| private_url | 0.73 | ⚠️ Add public-domain whitelist |
| private_address | 0.58 | ⚠️ Add "must contain digit" rule |
| **Macro F1** | **0.80** | |
| **Negative FPR** | **8.3%** | (on 744 PII-free samples) |

See [eval/eval_results.json](eval/eval_results.json) for the full breakdown.

## Quick start

```bash
pip install -r requirements.txt
python examples/inference.py "我的手机号 13812345678,邮箱 a@b.com"
```

Programmatic use:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

base = "openai/privacy-filter"
tokenizer = AutoTokenizer.from_pretrained("adapter", trust_remote_code=True)
model = AutoModelForTokenClassification.from_pretrained(base, trust_remote_code=True)
model = PeftModel.from_pretrained(model, "adapter")
model.eval()

# ... see examples/inference.py for full decode loop ...
```

## Reproduce training

```bash
python scripts/train.py \
    --train data/train.jsonl \
    --eval data/gold_test.jsonl \
    --output checkpoints/sift-opf-zh \
    --epochs 3 --batch 4
```

≈ 3h on Apple M2 Max (MPS), peak VRAM ~12 GB.

## Reproduce evaluation

```bash
python scripts/eval.py \
    --checkpoint adapter \
    --eval data/gold_test.jsonl \
    --output eval_results.json
```

≈ 7 min on M2 Max.

## Repository layout

```
sift-opf-zh/
├── adapter/              # LoRA weights (2.3 MB) + tokenizer
├── data/
│   ├── train.jsonl       # 12,126 training samples
│   └── gold_test.jsonl   # 1,816 evaluation samples
├── eval/
│   └── eval_results.json # Full metric breakdown
├── examples/
│   └── inference.py      # 5-line example
├── scripts/
│   ├── train.py          # Training script
│   ├── eval.py           # Evaluation script
│   └── synth_helper.py   # Data synthesis helper
├── CHANGELOG.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Model details

- **Base**: [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) (1.4 B params, MoE, BIOES, 33 labels)
- **Adapter**: LoRA (r=16, α=32, target = q/k/v/o_proj)
- **Trainable params**: ~600K (0.04% of base)
- **Adapter size**: 2.3 MB (.safetensors)
- **Languages**: Simplified Chinese (Mainland), with some English / pinyin
- **Schema**: 8 entity classes from OPF's 33-label set, BIOES decoding

### Training data

- ~7,200 cleaned [wan9yu/pii-bench-zh](https://huggingface.co/datasets/wan9yu/pii-bench-zh) samples
- ~1,900 hand-written synthetic samples covering edge cases
- ~1,970 independent real-style samples (merged from earlier eval set)
- ~280 v2.1 corrections (long-text, hard-negatives, category balance)
- **All addresses re-cleaned per "must contain digit" rule**

### Evaluation data

`data/gold_test.jsonl` is an **independent** 1,816-sample benchmark:
- 14 distinct scenarios (chat, email, customer-service logs, IDs, long documents)
- 41% pure-negative samples (PII-free) for FPR measurement
- 22% medium / long text (50-800 chars)
- Zero overlap with training data (deduplicated)
- Address annotations follow strict "must contain digit" rule

## Limitations & intended use

### ✅ Intended use
- On-device PII detection in Simplified Chinese consumer apps
- Pre-redaction stage in document processing
- Conversational privacy filter (chatbots, customer service)

### ⚠️ NOT recommended for
- **Standalone production use without rule-based post-filter** — secret/account/url/address all need regex/checksum/whitelist sanity checks. The model + rules is a sound system; the model alone is not.
- Languages other than Simplified Chinese
- Public-figure name detection (model is intentionally conservative — "邓小平" / "马斯克" won't be tagged as private_person)
- Address detection without digits (model follows strict rule — "北京海淀" won't be tagged; this is a feature, not a bug, for the Sift use case)

### Known failure modes
- Long-text F1 drops ~10pp vs short-text (training data was short-text heavy)
- Boundary errors on partial addresses (e.g. "建国路 88" without 号)
- Occasional misses on novel API-key formats not in training (sk-*, AKIA*, ghp_*, xoxb-* are covered)

## How v2 was developed

See [CHANGELOG.md](CHANGELOG.md) for the version history (v1 → v2.1 → v2.2 experiments).

Key lesson: **synthetic templates beat raw count** — v2.2 added 1,800 templated samples and *regressed* on 4 classes, while v2's organic mix at lower volume hit higher macro. Synthetic data should use LLM rewriting, not Python f-string templates.

## Citation

```bibtex
@software{sift_opf_zh_2026,
  title  = {sift-opf-zh: A LoRA fine-tune of openai/privacy-filter for Chinese PII},
  year   = {2026},
  url    = {https://github.com/YOUR_HANDLE/sift-opf-zh},
  note   = {v2.0}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

The base model `openai/privacy-filter` is licensed separately by OpenAI; consult its license before use.

## Acknowledgements

- [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) — base model
- [wan9yu/pii-bench-zh](https://huggingface.co/datasets/wan9yu/pii-bench-zh) — seed data
- [PEFT](https://github.com/huggingface/peft) — LoRA training stack
