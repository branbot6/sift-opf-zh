# Data

## Files

| File | Samples | Purpose |
|---|---:|---|
| `train.jsonl` | 12,126 | Training set |
| `gold_test.jsonl` | 1,816 | Independent evaluation benchmark |

## Schema

Each line is a JSON object:

```json
{
  "text": "我的手机号是 13812345678",
  "entities": [
    {"start": 7, "end": 18, "label": "private_phone"}
  ]
}
```

- `text` (str): the raw Chinese / mixed text
- `entities` (list): character-offset spans (Python-style half-open `[start, end)`)
- `label` (str): one of the 8 supported classes

## Composition

### `train.jsonl` (12,126 samples)

| Source | Count | Notes |
|---|---:|---|
| `wan9yu/pii-bench-zh` (cleaned) | ~7,200 | Templates removed, address spans fixed |
| Hand-written synthetic | ~1,900 | Wide scenario coverage |
| Real-style corpus | ~1,970 | Independent of gold_test |
| v2.1 fixes | ~1,000 | Long text + hard-negatives + category balance |

- 16.4 % negative samples (PII-free)
- All addresses contain ≥ 1 Arabic digit (strict rule)
- Zero overlap with `gold_test.jsonl` (text-hash deduplicated)

### `gold_test.jsonl` (1,816 samples)

Independent benchmark across 14 scenarios:

- Chat / SMS
- Email / customer service logs
- Identity documents
- Long documents (with sparse PII — the "needle in haystack" case)
- Pure negatives (44 % of samples) for FPR measurement

Length distribution:

| Bucket | Samples |
|---|---:|
| < 50 chars | 1,419 |
| 50–150 chars | 266 |
| > 150 chars | 131 |

## Annotation rules

- **address**: must contain at least one Arabic digit (`0-9` or full-width `０-９`).
  - ✅ `海淀学院路 17 号`, `万科城 3 号楼 1502`
  - ❌ `北京`, `朝阳区`, `中关村` (these are tagged `O`)
  - Rationale: aligns model with Sift's "personal location vs. public landmark" semantics.

- **person**: real personal names only. Public figures (`邓小平`, `马斯克`) are intentionally tagged `O`.

- **url**: private/identifying links (网盘, social profiles, short links).
  Public domains (`xinhuanet.com`, `wikipedia.org`) are tagged `O`.

- **date**: dates bound to a person (birth, employment, contract).
  News dates ("2024 Q3", "2008 汶川地震") are tagged `O`.

- **secret**: API keys, tokens, passwords, DB connection strings.

- **account_number**: bank cards, ID cards, order numbers, student/employee IDs, social security.

## Synthetic data disclaimer

All "PII-looking" content (phone numbers, ID cards, bank cards) is **synthetic and randomized** — no real personal information is included in this dataset. Numbers may coincidentally match real records but are not derived from any real source.
