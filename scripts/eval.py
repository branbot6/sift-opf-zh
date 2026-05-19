"""Evaluate fine-tuned OPF on Chinese held-out data.

Run:
    python -m training.eval_zh \
        --checkpoint training/checkpoints/opf-zh-v0 \
        --eval data/dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).parent / "reports"


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def entities_to_bio(text: str, entities: list[dict]) -> list[str]:
    """Char-level BIO tags."""
    tags = ["O"] * len(text)
    for e in entities:
        s, end, label = e["start"], e["end"], e["label"]
        if s >= len(text):
            continue
        tags[s] = f"B-{label}"
        for i in range(s + 1, min(end, len(text))):
            tags[i] = f"I-{label}"
    return tags


def predict_entities(model, tokenizer, text: str, device: str) -> list[dict]:
    """Token-level NER prediction → entity spans."""
    import torch
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits[0]
    label_ids = logits.argmax(dim=-1).tolist()
    id2label = model.config.id2label

    # BIOES decoding: B begins, I continues, E closes, S is single-token entity, O is outside.
    entities: list[dict] = []
    cur = None
    for tok_id, (s, e) in zip(label_ids, offsets):
        if s == e == 0:
            if cur:
                entities.append(cur)
                cur = None
            continue
        label = id2label.get(tok_id, "O")
        if label == "O":
            if cur:
                entities.append(cur)
                cur = None
            continue
        prefix, _, cat = label.partition("-")
        if prefix == "S":
            # Single-token entity: 当前若有进行中,先结束,然后单独开闭一个 entity
            if cur:
                entities.append(cur)
                cur = None
            entities.append({"start": s, "end": e, "label": cat})
        elif prefix == "B":
            if cur:
                entities.append(cur)
            cur = {"start": s, "end": e, "label": cat}
        elif prefix == "I":
            if cur and cur["label"] == cat:
                cur["end"] = e
            else:
                # I 没有对应的 B,作为新 entity 开始(decode rescue)
                if cur:
                    entities.append(cur)
                cur = {"start": s, "end": e, "label": cat}
        elif prefix == "E":
            if cur and cur["label"] == cat:
                cur["end"] = e
                entities.append(cur)
                cur = None
            else:
                # E 没对应 B/I,当作 single token
                if cur:
                    entities.append(cur)
                entities.append({"start": s, "end": e, "label": cat})
                cur = None
    if cur:
        entities.append(cur)

    # 修正 token 边界 bug:tokenizer 把前导/尾部空格包进 token,
    # 导致 entity span 比 gold 多 1 个空格字符 → 修剪掉两端空白
    cleaned = []
    for ent in entities:
        s, e = ent["start"], ent["end"]
        # 不能超过 text 长度
        s, e = max(0, s), min(len(text), e)
        # 去掉前导空白
        while s < e and text[s].isspace():
            s += 1
        # 去掉尾部空白
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            cleaned.append({"start": s, "end": e, "label": ent["label"]})
    return cleaned


def _fbeta(p: float, r: float, beta: float) -> float:
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def compute_f1(eval_samples: list[dict], predictions: list[list[dict]]) -> dict:
    """Per-category entity-level P / R / F1 / F2 (exact span match).

    F2 weighs Recall 2× more than Precision — 适合 PII 场景(漏检代价 > 误检)。
    """
    from collections import defaultdict
    by_cat: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for sample, preds in zip(eval_samples, predictions):
        gold = {(e["start"], e["end"], e["label"]) for e in sample["entities"]}
        pred = {(e["start"], e["end"], e["label"]) for e in preds}
        cats = {l for _, _, l in gold | pred}
        for cat in cats:
            gold_c = {x for x in gold if x[2] == cat}
            pred_c = {x for x in pred if x[2] == cat}
            by_cat[cat]["tp"] += len(gold_c & pred_c)
            by_cat[cat]["fp"] += len(pred_c - gold_c)
            by_cat[cat]["fn"] += len(gold_c - pred_c)

    report = {}
    for cat, c in by_cat.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
        report[cat] = {
            "precision": p, "recall": r,
            "f1": _fbeta(p, r, 1.0),
            "f2": _fbeta(p, r, 2.0),
            "support": c["tp"] + c["fn"],
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        }

    # Macro avg
    cats = list(report.keys())
    if cats:
        report["__macro__"] = {
            "precision": sum(report[c]["precision"] for c in cats) / len(cats),
            "recall": sum(report[c]["recall"] for c in cats) / len(cats),
            "f1": sum(report[c]["f1"] for c in cats) / len(cats),
            "f2": sum(report[c]["f2"] for c in cats) / len(cats),
        }
    return report


def compute_confusion_matrix(eval_samples: list[dict], predictions: list[list[dict]]) -> dict:
    """实体级混淆矩阵: 把 gold span 投到 pred span 的标签上.

    返回:
      {
        ('private_person', 'O'):           45,   # 该是 person 但漏了
        ('O', 'private_person'):           12,   # 不是 person 但误检
        ('private_address', 'private_person'): 3, # 把地址当成人名
        ...
      }
    """
    from collections import Counter
    confusions = Counter()

    for sample, preds in zip(eval_samples, predictions):
        gold = {(e["start"], e["end"]): e["label"] for e in sample["entities"]}
        pred = {(e["start"], e["end"]): e["label"] for e in preds}
        # 漏检 + 类别错
        for span, glabel in gold.items():
            plabel = pred.get(span, "O")  # 完全没预测 = "O"
            if plabel != glabel:
                confusions[(glabel, plabel)] += 1
        # 误检
        for span, plabel in pred.items():
            if span not in gold:
                confusions[("O", plabel)] += 1
    return dict(confusions)


def compute_boundary_errors(eval_samples: list[dict], predictions: list[list[dict]]) -> dict:
    """诊断 「label 对但 span 边界错」 的情况.

    返回:
      {
        'private_address': {
          'total_gold': 376,
          'exact_match': 320,         # 完全对
          'partial_overlap': 35,      # label 一样但 span 重叠不完全(过短 / 过长)
          'gold_avg_len': 21.3,
          'pred_avg_len_when_partial': 18.7,  # < gold_avg_len 说明「太短」
        },
        ...
      }
    """
    from collections import defaultdict
    stats = defaultdict(lambda: {
        "total_gold": 0, "exact_match": 0, "partial_overlap": 0,
        "gold_len_sum": 0, "partial_pred_len_sum": 0, "n_partial": 0,
    })

    for sample, preds in zip(eval_samples, predictions):
        for g in sample["entities"]:
            label = g["label"]
            gs, ge = g["start"], g["end"]
            stats[label]["total_gold"] += 1
            stats[label]["gold_len_sum"] += ge - gs
            # 完全匹配?
            if any(p["start"] == gs and p["end"] == ge and p["label"] == label for p in preds):
                stats[label]["exact_match"] += 1
                continue
            # 部分重叠 + 同 label?
            for p in preds:
                if p["label"] != label:
                    continue
                # 区间重叠
                overlap = max(0, min(ge, p["end"]) - max(gs, p["start"]))
                if overlap > 0:
                    stats[label]["partial_overlap"] += 1
                    stats[label]["partial_pred_len_sum"] += p["end"] - p["start"]
                    stats[label]["n_partial"] += 1
                    break  # 只算一次

    # 整理
    out = {}
    for label, s in stats.items():
        out[label] = {
            "total_gold": s["total_gold"],
            "exact_match": s["exact_match"],
            "partial_overlap": s["partial_overlap"],
            "missed_entirely": s["total_gold"] - s["exact_match"] - s["partial_overlap"],
            "gold_avg_len": s["gold_len_sum"] / s["total_gold"] if s["total_gold"] else 0,
            "pred_avg_len_when_partial": (
                s["partial_pred_len_sum"] / s["n_partial"] if s["n_partial"] else 0
            ),
        }
    return out


def compute_per_length(eval_samples: list[dict], predictions: list[list[dict]]) -> dict:
    """按文本长度分桶: 短 (<50) / 中 (50-150) / 长 (>150). 看长度是否影响 F1."""
    buckets = {
        "short_<50": [],
        "mid_50_150": [],
        "long_>150": [],
    }
    for s, p in zip(eval_samples, predictions):
        L = len(s["text"])
        if L < 50:
            buckets["short_<50"].append((s, p))
        elif L < 150:
            buckets["mid_50_150"].append((s, p))
        else:
            buckets["long_>150"].append((s, p))

    out = {}
    for name, items in buckets.items():
        if not items:
            out[name] = {"n_samples": 0}
            continue
        samples = [it[0] for it in items]
        preds = [it[1] for it in items]
        rep = compute_f1(samples, preds)
        macro = rep.get("__macro__", {})
        out[name] = {
            "n_samples": len(items),
            "macro_precision": macro.get("precision", 0),
            "macro_recall": macro.get("recall", 0),
            "macro_f1": macro.get("f1", 0),
            "macro_f2": macro.get("f2", 0),
        }
    return out


def compute_neg_fpr(eval_samples: list[dict], predictions: list[list[dict]]) -> dict:
    """Hard-negative FPR: 在 0-entity 样本上模型预测了多少 PII (越低越好).

    这些样本本来 0 entity (公众人物 / 国家名 / 普通对话), 任何预测都是误报.
    """
    neg_samples = [(s, p) for s, p in zip(eval_samples, predictions) if not s["entities"]]
    if not neg_samples:
        return {"n_negative_samples": 0}

    n_neg = len(neg_samples)
    false_predictions = sum(len(p) for _, p in neg_samples)
    n_samples_with_fp = sum(1 for _, p in neg_samples if p)

    return {
        "n_negative_samples": n_neg,
        "false_predictions_total": false_predictions,
        "false_pred_per_sample": false_predictions / n_neg,
        "samples_with_any_fp": n_samples_with_fp,
        "fp_sample_rate": n_samples_with_fp / n_neg,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to fine-tuned model directory")
    p.add_argument("--eval", required=True, help="eval JSONL")
    p.add_argument("--output", default=str(REPORT_DIR / "eval.json"))
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as e:
        print(f"Missing dependency: {e}\nInstall: pip install sift-privacy[train]", file=sys.stderr)
        return 1

    print(f"loading {args.checkpoint}")
    # 如果 checkpoint 是 LoRA adapter,则先加载 base model 再 attach adapter
    from pathlib import Path
    is_adapter = (Path(args.checkpoint) / "adapter_config.json").exists()

    if is_adapter:
        import json as _json
        cfg = _json.loads((Path(args.checkpoint) / "adapter_config.json").read_text())
        base = cfg["base_model_name_or_path"]
        print(f"  detected LoRA adapter, base={base}")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
        base_model = AutoModelForTokenClassification.from_pretrained(
            base, trust_remote_code=True, dtype=torch.float32,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
        model = AutoModelForTokenClassification.from_pretrained(
            args.checkpoint, trust_remote_code=True, dtype=torch.float32,
        )

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"  device: {device}")
    model = model.to(device)
    model.eval()

    eval_samples = load_jsonl(args.eval)
    print(f"evaluating on {len(eval_samples)} samples")
    predictions = []
    from tqdm import tqdm
    for s in tqdm(eval_samples, desc="predict"):
        predictions.append(predict_entities(model, tokenizer, s["text"], device))

    print("\nComputing metrics...")
    per_class = compute_f1(eval_samples, predictions)
    confusion = compute_confusion_matrix(eval_samples, predictions)
    boundary = compute_boundary_errors(eval_samples, predictions)
    per_length = compute_per_length(eval_samples, predictions)
    neg_fpr = compute_neg_fpr(eval_samples, predictions)

    report = {
        "checkpoint": args.checkpoint,
        "eval_set": args.eval,
        "n_samples": len(eval_samples),
        "per_class": per_class,
        "confusion_matrix_top20": sorted(
            confusion.items(), key=lambda x: -x[1]
        )[:20],
        "boundary_errors": boundary,
        "per_length_buckets": per_length,
        "negative_sample_fpr": neg_fpr,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # 把 tuple key 转成 "X→Y" 字符串方便 json
        report_json = dict(report)
        report_json["confusion_matrix_top20"] = [
            {"gold": k[0], "pred": k[1], "count": v}
            for k, v in report["confusion_matrix_top20"]
        ]
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    # 控制台打印
    print(f"\n{'='*70}")
    print(f" 报告(JSON 保存到 {out_path})")
    print('='*70)
    print(f"\n【1】Per-class P / R / F1 / F2 (F2 强调 Recall,适合 PII)")
    print(f"{'category':22s} {'P':>6s} {'R':>6s} {'F1':>6s} {'F2':>6s} {'sup':>6s}")
    for cat, m in sorted(per_class.items()):
        if cat == "__macro__":
            continue
        print(f"  {cat:20s} {m['precision']:6.3f} {m['recall']:6.3f} "
              f"{m['f1']:6.3f} {m['f2']:6.3f} {m['support']:6d}")
    if "__macro__" in per_class:
        m = per_class["__macro__"]
        print(f"  {'MACRO':20s} {m['precision']:6.3f} {m['recall']:6.3f} "
              f"{m['f1']:6.3f} {m['f2']:6.3f}")

    print(f"\n【2】Confusion matrix top 10 (gold → pred)")
    for (g, p), n in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
        print(f"  {g:22s} → {p:22s}  {n}")

    print(f"\n【3】Boundary errors (label 对但 span 边界错)")
    print(f"{'category':22s} {'gold':>6s} {'exact':>6s} {'partial':>8s} "
          f"{'missed':>7s} {'gold_avg_len':>13s}")
    for cat, s in sorted(boundary.items()):
        print(f"  {cat:20s} {s['total_gold']:6d} {s['exact_match']:6d} "
              f"{s['partial_overlap']:8d} {s['missed_entirely']:7d} "
              f"{s['gold_avg_len']:13.1f}")

    print(f"\n【4】Per-length 分桶 F1")
    print(f"{'bucket':22s} {'n':>6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'F2':>6s}")
    for name, b in per_length.items():
        if b['n_samples'] == 0:
            print(f"  {name:20s} {b['n_samples']:6d}  (no samples)")
            continue
        print(f"  {name:20s} {b['n_samples']:6d} "
              f"{b['macro_precision']:6.3f} {b['macro_recall']:6.3f} "
              f"{b['macro_f1']:6.3f} {b['macro_f2']:6.3f}")

    print(f"\n【5】Negative-sample FPR (0-entity 样本上的误报)")
    if neg_fpr.get('n_negative_samples', 0) > 0:
        print(f"  负样本总数:        {neg_fpr['n_negative_samples']}")
        print(f"  误报总次数:        {neg_fpr['false_predictions_total']}")
        print(f"  每条平均误报:      {neg_fpr['false_pred_per_sample']:.3f}")
        print(f"  有误报的样本占比:  {neg_fpr['fp_sample_rate']*100:.1f}%")
    else:
        print(f"  ⚠️ 没有负样本可测")

    return 0


if __name__ == "__main__":
    sys.exit(main())
