"""Minimal inference example for sift-opf-zh.

Usage:
    python examples/inference.py "我的手机号 13812345678,邮箱 a@b.com"
"""
import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
ADAPTER = REPO / "adapter"
BASE = "openai/privacy-filter"


def load_model(device: str = "auto"):
    if device == "auto":
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER), trust_remote_code=True)
    base_model = AutoModelForTokenClassification.from_pretrained(BASE, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, str(ADAPTER))
    model.to(device).eval()
    return model, tokenizer, device


def predict(text: str, model, tokenizer, device: str):
    """Predict PII entity spans on a single text. Returns list of {start, end, label, text}."""
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt",
                    truncation=True, max_length=512)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]
    pred_ids = logits.argmax(-1).tolist()
    id2label = model.config.id2label

    entities, current = [], None
    for tid, (s, e) in zip(pred_ids, offsets):
        if s == e:  # special token
            continue
        tag = id2label[tid]
        prefix = tag[0]
        label = tag[2:] if len(tag) > 2 else None
        if prefix in ("B", "S"):
            if current:
                entities.append(current)
            current = {"start": s, "end": e, "label": label}
        elif prefix in ("I", "E") and current and current["label"] == label:
            current["end"] = e
            if prefix == "E":
                entities.append(current); current = None
        else:
            if current:
                entities.append(current); current = None
    if current:
        entities.append(current)

    cleaned = []
    for ent in entities:
        s, e = max(0, ent["start"]), min(len(text), ent["end"])
        while s < e and text[s].isspace(): s += 1
        while e > s and text[e - 1].isspace(): e -= 1
        if s < e:
            cleaned.append({"start": s, "end": e, "label": ent["label"], "text": text[s:e]})
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+", help="Chinese text to detect PII in.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    text = " ".join(args.text)

    print(f"Loading model ({args.device})...")
    model, tokenizer, device = load_model(args.device)
    print(f"Device: {device}\n")

    entities = predict(text, model, tokenizer, device)
    print(f"Input:    {text}\n")
    if not entities:
        print("No PII detected.")
        return
    print(f"{'Label':<20} {'Span':<6} Text")
    print("-" * 60)
    for ent in entities:
        span = f"{ent['start']}-{ent['end']}"
        print(f"{ent['label']:<20} {span:<6} {ent['text']}")


if __name__ == "__main__":
    main()
