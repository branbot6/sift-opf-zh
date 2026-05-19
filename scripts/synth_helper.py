"""Helper for hand-written synthetic samples.

每条样本只写 (text, [(entity_text, label), ...]),Python 自动算 char offset
并验证 text[start:end] == entity_text。出错就抛,保证落盘的全是合法样本。

用法:
    from synth_helper import make, write_jsonl
    samples = [
        make("我叫张三，电话13812345678", [("张三", "private_person"),
                                            ("13812345678", "private_phone")]),
    ]
    write_jsonl(samples, "data/synth/url.jsonl")
"""
from __future__ import annotations
import json
from pathlib import Path


def make(text: str, entities_spec: list[tuple[str, str]]) -> dict:
    """Build a sample dict. entities_spec = [(entity_text, label), ...].

    Searches each entity left-to-right, advancing cursor after each match
    (so duplicates in the same text get distinct spans).
    """
    entities = []
    cursor = 0
    for ent_text, label in entities_spec:
        idx = text.find(ent_text, cursor)
        if idx < 0:
            # Fallback: maybe entity appears earlier; try from 0
            idx = text.find(ent_text)
        if idx < 0:
            raise ValueError(f"entity {ent_text!r} not found in text {text!r}")
        end = idx + len(ent_text)
        entities.append({"start": idx, "end": end, "label": label})
        cursor = end
    # span sanity check
    for e in entities:
        assert text[e["start"]:e["end"]] in [s for s, _ in entities_spec], \
            f"span mismatch: {text[e['start']:e['end']]!r}"
    return {"text": text, "entities": entities}


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write_jsonl(samples: list[dict], path: str | Path) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} samples → {p}")
