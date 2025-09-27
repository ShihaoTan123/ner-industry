#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path

def read_ann(ann_path: Path):
    spans = []
    if not ann_path.exists(): return spans
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("T"): 
            continue
        m = re.match(r"^T\d+\t([A-Za-z_]+)\s+(\d+)\s+(\d+)\t", line)
        if m:
            typ, s, e = m.group(1), int(m.group(2)), int(m.group(3))
            spans.append((s, e, typ))
            continue
        m2 = re.match(r"^T\d+\t([A-Za-z_]+)\s+([\d; ]+)\t", line)
        if m2:
            typ, ranges = m2.groups()
            for seg in ranges.split(";"):
                s, e = map(int, seg.strip().split())
                spans.append((s, e, typ))
    return spans

def simple_tokenize_with_offsets(text: str):
    tokens, offsets = [], []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets

def spans_to_bio(tokens, offsets, spans):
    labs = ["O"] * len(tokens)
    for s_char, e_char, typ in spans:
        covered = [i for i, (s, e) in enumerate(offsets) if not (e <= s_char or e_char <= s)]
        if not covered:
            continue
        labs[covered[0]] = f"B-{typ}"
        for i in covered[1:]:
            labs[i] = f"I-{typ}"
    return labs

def convert_split(txt_dir: Path, ann_dir: Path, out_jsonl: Path, label_set: set):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_docs, n_tokens, n_ents = 0, 0, 0
    with out_jsonl.open("w", encoding="utf-8") as w:
        for txt_path in sorted(txt_dir.glob("*.txt")):
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            ann_path = ann_dir / (txt_path.stem + ".ann")
            spans = read_ann(ann_path)
            toks, offs = simple_tokenize_with_offsets(text)
            labs = spans_to_bio(toks, offs, spans)
            for l in labs:
                if l != "O": label_set.add(l[2:])
            w.write(json.dumps({"tokens": toks, "labels": labs}, ensure_ascii=False) + "\n")
            n_docs += 1
            n_tokens += len(toks)
            n_ents += len(spans)
    return n_docs, n_tokens, n_ents

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="e.g., data/scienceie/raw")
    ap.add_argument("--train_dir", default="scienceie2017_train/train2")
    ap.add_argument("--dev_dir",   default="dev")
    ap.add_argument("--test_dir",  default="semeval_articles_test")
    ap.add_argument("--out_dir",   required=True, help="e.g., data/scienceie/prepared")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    label_set = set()
    stats = {}
    # train
    tr_txt = raw / args.train_dir
    tr_ann = raw / args.train_dir
    stats["train"] = convert_split(tr_txt, tr_ann, out / "train.jsonl", label_set)
    # dev
    dv_txt = raw / args.dev_dir
    dv_ann = raw / args.dev_dir
    stats["dev"] = convert_split(dv_txt, dv_ann, out / "dev.jsonl", label_set)
    # test
    te_txt = raw / args.test_dir
    te_ann = raw / args.test_dir
    stats["test"] = convert_split(te_txt, te_ann, out / "test.jsonl", label_set)

    labels = ["O"] + sorted({f"B-{t}" for t in label_set} | {f"I-{t}" for t in label_set})
    (out / "label_list.txt").write_text("\n".join(labels), encoding="utf-8")

    print("[ScienceIE] labels:", labels)
    for split, (nd, ntok, nent) in stats.items():
        print(f"[ScienceIE] {split}: docs={nd}, tokens={ntok}, entities={nent}")

if __name__ == "__main__":
    main()
