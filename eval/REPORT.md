# Evaluation report — sift-opf-zh v2

> Benchmark: `data/gold_test.jsonl` (1,816 samples, strict-address gold)
> Metric: exact entity-span F1 (BIOES)

## Summary

| Version | Macro F1 | Negative FPR |
|---|---:|---:|
| v1 | 0.670 | 12.8 % |
| **v2 (this release)** | **0.801** | **8.3 %** |

**+13.1 pp macro F1, –4.5 pp false-positive rate** over v1.

## Per-class

| Class | P | R | F1 | Support |
|---|---:|---:|---:|---:|
| private_email | 0.97 | 0.97 | **0.966** | 237 |
| private_phone | 0.95 | 0.99 | **0.970** | 711 |
| private_person | 0.84 | 0.88 | **0.860** | 977 |
| private_date | 0.72 | 0.85 | **0.780** | 390 |
| secret | 0.76 | 0.79 | **0.776** | 66 |
| account_number | 0.69 | 0.80 | **0.742** | 294 |
| private_url | 0.73 | 0.74 | **0.734** | 127 |
| private_address | 0.59 | 0.55 | **0.576** | 218 |

## By text length

| Bucket | Samples | Macro F1 |
|---|---:|---:|
| short < 50 chars | 1,419 | 0.834 |
| mid 50–150 chars | 266 | 0.847 |
| long > 150 chars | 131 | 0.744 |

## Negative samples (744 PII-free)

| Metric | Value |
|---|---:|
| Sample-level FPR | 4.0 % (lenient) |
| FPR (strict address rule) | 8.3 % |
| Avg false predictions / sample | 0.046 |

## Production recommendation

Combine model with rule-based post-filters for production:

| Class | Suggested rule |
|---|---|
| address | "must contain digit" filter ⇒ P → 0.85+ |
| account_number | Luhn check for cards, 18-digit + checksum for ID cards |
| secret | Regex for known key prefixes (`sk-*`, `AKIA*`, `ghp_*`, `xoxb-*`) |
| url | Whitelist of public domains (`xinhuanet.com`, `wikipedia.org`, ...) |
| date | Surrounding-context regex ("出生于" / "DOB" / "入职日期") |

Full raw metrics, confusion matrix, and boundary-error breakdown: [`eval_results.json`](eval_results.json).
