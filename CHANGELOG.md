# Changelog

## v2.0 — 2026-05-18 (current release)

**Macro F1: 0.80 | Negative FPR: 8.3%**

- 12,126 training samples (wan9yu cleaned + hand-written + real-style + v2.1 fixes)
- Strict address rule: must contain Arabic digit
- BIOES decoding with BPE-whitespace fix (the bug that hid v1's URL F1 from 0 → 0.68)
- LoRA r=16, 3 epochs, ~3 h on M2 Max

### Improvements over v1
- macro F1: 0.67 → 0.80 (+13.4 pp)
- All 8 classes improved; URL +33.6 pp, secret +19 pp, account +16 pp
- Negative FPR cut from 12.8 % to 8.3 %

## v2.2 — 2026-05-19 (experiment, rolled back)

**Macro F1: 0.79 (-1.4 pp)** — not released.

Targeted boost on address (+8.8 pp ✅) but secret/account/url regressed by 4–10 pp due to:
1. Synthetic templates too narrow → model overfit to template syntax
2. Hard-negatives too aggressive (~1 : 4 ratio) → model became over-conservative
3. No dev set / early-stop → over-fit on synthetic train

Recovered insight, rolled back to v2.

## v2.1 — 2026-05-18 (intermediate)

- Hard-negatives 217 (public figures, countries, hotlines)
- Long-text +86 (150–400 chars)
- Category rebalancing (secret / url / date)
- Removed 346 template residues from wan9yu (`签证申请人:`, `挡路了 车牌` ...)

## v1.0 — 2026-05-17

**Macro F1: 0.67**

- Initial release: 8,000 wan9yu/pii-bench-zh + 1,966 hand-written
- LoRA r=16, 3 epochs
- Critical BPE-whitespace bug discovered and fixed (URL F1: 0 → 0.68)
