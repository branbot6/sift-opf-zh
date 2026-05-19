# LoRA Adapter — sift-opf-zh v2

LoRA fine-tune of [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) for Chinese PII detection.

## Files

| File | Purpose |
|---|---|
| `adapter_config.json` | LoRA configuration (r=16, α=32, q/k/v/o_proj) |
| `adapter_model.safetensors` | LoRA weights (2.3 MB) |
| `tokenizer.json` | BPE tokenizer (must match base) |
| `tokenizer_config.json` | Tokenizer config |

## Load

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("adapter", trust_remote_code=True)
base = AutoModelForTokenClassification.from_pretrained(
    "openai/privacy-filter", trust_remote_code=True
)
model = PeftModel.from_pretrained(base, "adapter")
model.eval()
```

For a complete inference example with BIOES decoding and BPE-whitespace handling, see [`../examples/inference.py`](../examples/inference.py).

## Labels

The model uses OPF's BIOES schema with 33 labels. Active classes for Chinese PII:

- `private_email`
- `private_phone`
- `private_person`
- `private_address` (must contain digit per training rule)
- `private_url`
- `private_date`
- `secret`
- `account_number`
