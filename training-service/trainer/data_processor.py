"""
Dataset preprocessing for HF fine-tuning.

Previous version trained a from-scratch BPE tokenizer and wrote raw token-id
arrays (train.npy/val.npy). That only works for the from-scratch GPT/LLaMA
models that used to live in model_builder.py. Real fine-tuning uses the base
model's own pretrained tokenizer, so preprocessing's job now is just: extract
clean text, dedupe, split train/val, and write JSONL — tokenization happens
lazily (and is cached) inside the training job, once we know which base model
the user picked.
"""
import json
import re
import hashlib
from pathlib import Path
from typing import List, Tuple


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def deduplicate(lines: List[str]) -> Tuple[List[str], int]:
    seen, unique, removed = set(), [], 0
    for line in lines:
        h = hashlib.md5(line.lower().strip().encode()).hexdigest()
        if h not in seen and len(line.strip()) > 20:
            seen.add(h)
            unique.append(line)
        else:
            removed += 1
    return unique, removed


def process_files(file_paths: List[str], output_dir: str, val_ratio: float = 0.1) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_lines = []
    for path in file_paths:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix.lower() == ".jsonl":
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # Accept either {"text": ...} or simple instruction/response pairs
                        if "text" in obj:
                            raw_lines.append(obj["text"])
                        elif "instruction" in obj and "response" in obj:
                            raw_lines.append(f"### Instruction:\n{obj['instruction']}\n\n### Response:\n{obj['response']}")
                    except json.JSONDecodeError:
                        continue
        elif p.suffix.lower() == ".pdf":
            try:
                import PyPDF2
                with open(p, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        raw_lines.extend((page.extract_text() or "").splitlines())
            except Exception:
                pass
        else:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                raw_lines.extend(f.readlines())

    cleaned = [clean_text(l) for l in raw_lines]
    cleaned = [l for l in cleaned if len(l.strip()) > 10]
    unique, removed = deduplicate(cleaned)

    split_idx = int(len(unique) * (1 - val_ratio))
    train_lines, val_lines = unique[:split_idx], unique[split_idx:]

    with open(output_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(json.dumps({"text": line}) + "\n")

    with open(output_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(json.dumps({"text": line}) + "\n")

    # Rough token estimate (whitespace split) purely for the pre-flight time
    # estimator on the frontend — the real count comes from the base model's
    # tokenizer once the user starts the job.
    approx_tokens = sum(len(l.split()) for l in unique)

    return {
        "approx_total_tokens": approx_tokens,
        "train_examples": len(train_lines),
        "val_examples": len(val_lines),
        "lines_raw": len(raw_lines),
        "duplicates_removed": removed,
    }
