#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path

def to_bio(sent_tokens, spans):
    labels = ["O"] * len(sent_tokens)
    for s, e, t in spans:
        labels[s] = f"B-{t}"
        for k in range(s + 1, e + 1):
            labels[k] = f"I-{t}"
    return labels

def convert_split(in_json_path: Path, out_jsonl_path: Path, label_set: set):
    n_sent, n_ent = 0, 0
    with in_json_path.open("r", encoding="utf-8") as f, out_jsonl_path.open("w", encoding="utf-8") as w:
        for line in f:
            ex = json.loads(line)
            sents = ex["sentences"]
            ners  = ex["ner"]
            for toks, spans in zip(sents, ners):
                labs = to_bio(toks, spans)
                for l in labs:
                    if l != "O": label_set.add(l[2:])
                w.write(json.dumps({"tokens": toks, "labels": labs}, ensure_ascii=False) + "\n")
                n_sent += 1
                n_ent  += len(spans)
    return n_sent, n_ent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json_dir", required=True, help="e.g., data/scierc/processed/json")
    ap.add_argument("--out_dir",     required=True, help="e.g., data/scierc/prepared")
    args = ap.parse_args()

    in_dir  = Path(args.in_json_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    label_set = set()
    stats = {}
    for split in ["train", "dev", "test"]:
        ns, ne = convert_split(in_dir / f"{split}.json", out_dir / f"{split}.jsonl", label_set)
        stats[split] = (ns, ne)

    labels = ["O"] + sorted({f"B-{t}" for t in label_set} | {f"I-{t}" for t in label_set})
    (out_dir / "label_list.txt").write_text("\n".join(labels), encoding="utf-8")

    print("[SciERC] labels:", labels)
    for k, (ns, ne) in stats.items():
        print(f"[SciERC] {k}: sentences={ns}, entities={ne}")

if __name__ == "__main__":
    main()
