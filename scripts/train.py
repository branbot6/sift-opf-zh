"""LoRA fine-tune OPF on Chinese data.

Run:
    pip install transformers torch peft datasets accelerate
    python -m training.train_zh \
        --train data/train.jsonl --eval data/dev.jsonl \
        --output checkpoints/opf-zh-v0 \
        --epochs 3 --lr 5e-5 --batch 16

Output: training/checkpoints/opf-zh-v0/  (LoRA adapter + tokenizer)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_BASE_MODEL = "openai/privacy-filter"

# OPF's 8 labels in BIO form. Updated to whatever the actual loaded model has.
DEFAULT_LABELS = [
    "O",
    "B-account_number", "I-account_number",
    "B-private_address", "I-private_address",
    "B-private_email", "I-private_email",
    "B-private_person", "I-private_person",
    "B-private_phone", "I-private_phone",
    "B-private_url", "I-private_url",
    "B-private_date", "I-private_date",
    "B-secret", "I-secret",
]


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def char_offsets_to_token_labels(text: str, entities: list[dict], tokenizer, label2id) -> list[int]:
    """Map character-offset entity spans to token-level BIOES label IDs.

    OPF 用 BIOES 5-tag scheme (33 labels = O + 8 categories × 4)。
    B = begin, I = inside, E = end, S = single-token entity.
    """
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = enc["offset_mapping"]
    labels = ["O"] * len(offsets)
    for ent in entities:
        e_start = ent["start"]
        e_end = ent["end"]
        e_label = ent["label"]
        # 找出落在该实体里的所有 token 索引
        hit = [
            i for i, (s, e) in enumerate(offsets)
            if not (s == e == 0) and not (e <= e_start or s >= e_end)
        ]
        if not hit:
            continue
        if len(hit) == 1:
            labels[hit[0]] = f"S-{e_label}"
        else:
            labels[hit[0]] = f"B-{e_label}"
            labels[hit[-1]] = f"E-{e_label}"
            for i in hit[1:-1]:
                labels[i] = f"I-{e_label}"
    return [label2id.get(l, label2id["O"]) for l in labels]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--train", required=True, help="train JSONL")
    p.add_argument("--eval", required=True, help="eval JSONL")
    p.add_argument("--output", default="training/checkpoints/opf-zh-v0")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    args = p.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            DataCollatorForTokenClassification,
        )
    except ImportError as e:
        print(f"Missing dependency: {e}\nInstall with: pip install sift-privacy[train]", file=sys.stderr)
        return 1

    print(f"loading base model {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    # bf16 on CUDA, fp32 elsewhere (MPS bf16 不稳定)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        dtype=dtype,
    )

    label2id = model.config.label2id or {l: i for i, l in enumerate(DEFAULT_LABELS)}
    id2label = model.config.id2label or {i: l for l, i in label2id.items()}

    print("loading training data")
    train_samples = load_jsonl(args.train)
    eval_samples = load_jsonl(args.eval)
    print(f"  train={len(train_samples)}  eval={len(eval_samples)}")

    def encode(sample):
        enc = tokenizer(
            sample["text"],
            return_offsets_mapping=False,
            truncation=True,
            max_length=512,
        )
        enc["labels"] = char_offsets_to_token_labels(
            sample["text"], sample["entities"], tokenizer, label2id
        )
        return enc

    train_ds = Dataset.from_list(train_samples).map(encode, remove_columns=["text", "entities"])
    eval_ds = Dataset.from_list(eval_samples).map(encode, remove_columns=["text", "entities"])

    print(f"wrapping with LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="TOKEN_CLS",
        # Targets — vary by base model architecture; defaults work for most
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 估计 warmup_steps (transformers 5.x 把 warmup_ratio 标 deprecated)
    steps_per_epoch = max(1, len(train_ds) // args.batch)
    warmup_steps = max(10, int(steps_per_epoch * args.epochs * 0.05))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to=[],
        bf16=torch.cuda.is_available(),
    )

    collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,  # transformers 5.x: tokenizer → processing_class
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"saved adapter + tokenizer to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
