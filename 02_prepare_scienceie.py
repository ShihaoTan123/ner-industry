import argparse
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

def get_sentence_spans(text: str) -> List[Tuple[int, int]]:
    try:
        import nltk
        try:
            tok = nltk.data.load('tokenizers/punkt/english.pickle')
        except LookupError:
            nltk.download('punkt', quiet=True)
            tok = nltk.data.load('tokenizers/punkt/english.pickle')
        spans = list(tok.span_tokenize(text))
        return spans
    except Exception:
        spans = []
        start = 0
        for m in re.finditer(r'([.!?]+)(\s+)', text):
            end = m.end()
            if end - start > 0:
                spans.append((start, end))
                start = end
        if start < len(text):
            spans.append((start, len(text)))
        refined = []
        for s, e in spans:
            chunk = text[s:e]
            offs = 0
            for seg in re.split(r'(\n\s*\n)+', chunk):
                if not seg or seg.isspace():
                    offs += len(seg or "")
                    continue
                refined.append((s + offs, s + offs + len(seg)))
                offs += len(seg)
        return refined or [(0, len(text))]

TOKEN_PATTERN = re.compile(r'\w+|[^\w\s]', re.UNICODE)

def doc_tokens_with_spans(text: str) -> List[Tuple[str, int, int]]:
    toks = []
    for m in TOKEN_PATTERN.finditer(text):
        toks.append((m.group(0), m.start(), m.end()))
    return toks

def read_brat_entities(ann_path: Path, keep_labels={'Material','Process','Task'}) -> List[Tuple[int,int,str]]:
    ents = []
    if not ann_path.exists():
        return ents
    for line in ann_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line or not line.startswith('T'):
            continue
        try:
            _id, rest = line.split('\t', 1)
            tag, mention = rest.split('\t', 1)
            parts = tag.split()
            label = parts[0]
            span_str = " ".join(parts[1:])
            seg = span_str.split(';')[0].strip()
            start_str, end_str = seg.split()[:2]
            start, end = int(start_str), int(end_str)
            if end > start and label in keep_labels:
                ents.append((start, end, label))
        except Exception:
            continue
    ents.sort(key=lambda x: (x[0], x[1]))
    return ents

def slice_tokens_to_sentence(tokens: List[Tuple[str,int,int]], sent_span: Tuple[int,int]) -> Tuple[List[str], List[Tuple[int,int]]]:
    s_start, s_end = sent_span
    sent_toks = []
    spans_local = []
    for tok, a, b in tokens:
        if b <= s_start:
            continue
        if a >= s_end:
            break
        la = max(a, s_start)
        lb = min(b, s_end)
        if lb > la:
            sent_toks.append(tok)
            spans_local.append((a, b))
    return sent_toks, spans_local

def map_entities_to_local_token_spans(
    sent_tokens: List[str],
    sent_tok_doc_spans: List[Tuple[int,int]],
    ent_spans_doc: List[Tuple[int,int,str]],
    sent_span: Tuple[int,int]
) -> List[Tuple[int,int,str]]:
    s_start, s_end = sent_span
    cand = [(a,b,t) for (a,b,t) in ent_spans_doc if a >= s_start and b <= s_end and a < b]
    mapped = []
    for a,b,t in cand:
        st = None; ed = None
        for i,(ta,tb) in enumerate(sent_tok_doc_spans):
            if tb <= a:
                continue
            if ta >= b:
                break
            if ta < b and tb > a:
                if st is None:
                    st = i
                ed = i
        if st is not None and ed is not None and 0 <= st <= ed < len(sent_tokens):
            mapped.append((st, ed, t))
    mapped.sort(key=lambda x: (-(x[1]-x[0]+1), x[0], x[1]))
    selected = []
    used = [False]*len(sent_tokens)
    for s,e,t in mapped:
        if any(used[i] for i in range(s,e+1)):
            continue
        selected.append((s,e,t))
        for i in range(s,e+1): used[i]=True
    selected.sort(key=lambda x: (x[0], x[1]))
    return selected

def to_bio(sent_tokens: List[str], spans_local: List[Tuple[int,int,str]]) -> List[str]:
    labs = ["O"]*len(sent_tokens)
    for s,e,t in spans_local:
        if not (0 <= s < len(sent_tokens)):
            continue
        labs[s] = f"B-{t}"
        for k in range(s+1, e+1):
            if 0 <= k < len(sent_tokens):
                labs[k] = f"I-{t}"
    return labs

def iter_split_examples(split_dir: Path, keep_labels={'Material','Process','Task'}) -> Iterable[Dict]:
    txts = sorted(split_dir.glob("*.txt"))
    for txt_path in txts:
        ann_path = txt_path.with_suffix(".ann")
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        ents = read_brat_entities(ann_path, keep_labels=keep_labels)
        sent_spans = get_sentence_spans(text)
        doc_toks = doc_tokens_with_spans(text)
        for s_span in sent_spans:
            toks, tok_doc_spans = slice_tokens_to_sentence(doc_toks, s_span)
            if not toks:
                continue
            local_spans = map_entities_to_local_token_spans(
                sent_tokens=toks,
                sent_tok_doc_spans=tok_doc_spans,
                ent_spans_doc=ents,
                sent_span=s_span
            )
            labels = to_bio(toks, local_spans)
            yield {"tokens": toks, "labels": labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--dev_dir",   required=True)
    ap.add_argument("--test_dir",  required=True)
    ap.add_argument("--out_dir",   required=True, help="e.g., data/scienceie/prepared")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    label_set = {"Material","Process","Task"}
    splits = {
        "train": Path(args.train_dir),
        "dev":   Path(args.dev_dir),
        "test":  Path(args.test_dir),
    }

    stats = {}
    for name, d in splits.items():
        n_sent = 0
        n_ent  = 0
        out_path = out / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as w:
            for ex in iter_split_examples(d, keep_labels=label_set):
                w.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_sent += 1
                n_ent  += sum(1 for lab in ex["labels"] if lab.startswith("B-"))
        stats[name] = (n_sent, n_ent)

    labels = ["O"] + sorted({f"B-{t}" for t in label_set} | {f"I-{t}" for t in label_set})
    (out / "label_list.txt").write_text("\n".join(labels), encoding="utf-8")

    print("[ScienceIE sentence-level] labels:", labels)
    for k,(ns,ne) in stats.items():
        print(f"[{k}] sentences={ns}, entities={ne}")

if __name__ == "__main__":
    main()

