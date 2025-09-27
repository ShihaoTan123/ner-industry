#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import json
from pathlib import Path
from typing import List, Tuple


def spans_doc_to_sent_local(
    sent_start: int,
    sent_len: int,
    spans_docidx: List[Tuple[int, int, str]],
    resolve_overlap: bool = True,
):

    sent_end = sent_start + sent_len - 1
    candidates = []

    for s_doc, e_doc, typ in spans_docidx:
        if s_doc > e_doc:
            continue
        if s_doc < sent_start or e_doc > sent_end:
            continue
        s_local = s_doc - sent_start
        e_local = e_doc - sent_start
        if 0 <= s_local < sent_len and 0 <= e_local < sent_len:
            candidates.append((s_local, e_local, typ))

    if not resolve_overlap:
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates


    candidates.sort(key=lambda x: (-(x[1] - x[0] + 1), x[0], x[1]))
    selected = []
    occupied = [False] * sent_len

    for s, e, t in candidates:
        if any(occupied[i] for i in range(s, e + 1)):
            continue
        selected.append((s, e, t))
        for i in range(s, e + 1):
            occupied[i] = True

    selected.sort(key=lambda x: (x[0], x[1]))
    return selected


def to_bio(sent_tokens: List[str], spans_local: List[Tuple[int, int, str]]) -> List[str]:

    labels = ["O"] * len(sent_tokens)
    for s, e, t in spans_local:
        if not (0 <= s < len(sent_tokens)) or not (0 <= e < len(sent_tokens)) or s > e:
            continue
        labels[s] = f"B-{t}"
        for k in range(s + 1, e + 1):
            labels[k] = f"I-{t}"
    return labels


def convert_split(in_json_path: Path, out_jsonl_path: Path, label_set: set):

    n_sent = 0
    n_ent_written = 0
    n_cross_sent = 0
    n_overlap_dropped = 0

    with in_json_path.open("r", encoding="utf-8") as f, out_jsonl_path.open("w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            sents: List[List[str]] = ex["sentences"]
            ners: List[List[List]] = ex["ner"]

            sent_starts = []
            cum = 0
            for toks in sents:
                sent_starts.append(cum)
                cum += len(toks)

            if len(sents) != len(ners):
                L = min(len(sents), len(ners))
                sents = sents[:L]
                ners = ners[:L]

            for i, (toks, spans_docidx) in enumerate(zip(sents, ners)):
                sent_start = sent_starts[i]
                sent_len = len(toks)

                before = len(spans_docidx)
                spans_local = spans_doc_to_sent_local(
                    sent_start=sent_start,
                    sent_len=sent_len,
                    spans_docidx=spans_docidx,
                    resolve_overlap=True,
                )

                fully_in = [
                    s for s in spans_docidx
                    if (s[0] >= sent_start and s[1] <= sent_start + sent_len - 1 and s[0] <= s[1])
                ]
                n_cross_sent += (before - len(fully_in))
                n_overlap_dropped += (max(len(fully_in), 0) - len(spans_local))

                labs = to_bio(toks, spans_local)
                for L_ in labs:
                    if L_ != "O":
                        label_set.add(L_[2:])
                w.write(json.dumps({"tokens": toks, "labels": labs}, ensure_ascii=False) + "\n")
                n_sent += 1
                n_ent_written += len(spans_local)

    return n_sent, n_ent_written, n_cross_sent, n_overlap_dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json_dir", required=True, help="e.g., data/scierc/processed/json")
    ap.add_argument("--out_dir",     required=True, help="e.g., data/scierc/prepared")
    args = ap.parse_args()

    in_dir = Path(args.in_json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_set = set()
    stats = {}

    for split in ["train", "dev", "test"]:
        in_path = in_dir / f"{split}.json"
        out_path = out_dir / f"{split}.jsonl"
        ns, ne, n_cross, n_ov = convert_split(in_path, out_path, label_set)
        stats[split] = (ns, ne, n_cross, n_ov)

    labels = ["O"] + sorted({f"B-{t}" for t in label_set} | {f"I-{t}" for t in label_set})
    (out_dir / "label_list.txt").write_text("\n".join(labels), encoding="utf-8")

    print("[SciERC] labels:", labels)
    for split, (ns, ne, n_cross, n_ov) in stats.items():
        print(f"[SciERC] {split}: sentences={ns}, entities_written={ne}, "
              f"cross_sentence_skipped={n_cross}, overlap_dropped={n_ov}")


if __name__ == "__main__":
    main()

