import os, re, json, argparse, pathlib, glob
from typing import List, Tuple, Dict, Any

def get_sentence_splitter():
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        def splitter(text: str):
            spans = []
            start = 0
            for sent in sent_tokenize(text):
                idx = text.find(sent, start)
                if idx == -1:
                    idx = start
                spans.append((idx, idx+len(sent)))
                start = idx + len(sent)
            return spans
        return splitter
    except Exception:
        SENT_RE = re.compile(r'[^.!?]+[.!?]|\S[^.!?]*$|[\n]+', re.UNICODE)
        def splitter(text: str):
            spans = []
            for m in SENT_RE.finditer(text):
                st, ed = m.span()
                if st != ed and text[st:ed].strip():
                    spans.append((st, ed))
            return spans
        return splitter

TOK_RE = re.compile(r'\w+|[^\w\s]', re.UNICODE)

def tokenize_with_offsets(text: str):
    """Return list of (token, start_char, end_char). end_char exclusive."""
    toks = []
    for m in TOK_RE.finditer(text):
        s, e = m.span()
        toks.append((text[s:e], s, e))
    return toks

def parse_brat_ann(ann_text: str):
    """Return list of dicts: {'id','label','spans':[(s,e),...],'text'}"""
    ents = []
    for line in ann_text.splitlines():
        if not line or not line.startswith('T'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        tid = parts[0]
        type_and_spans = parts[1]
        text = parts[2] if len(parts) >= 3 else ""
        first_space = type_and_spans.find(' ')
        if first_space <= 0:
            continue
        label = type_and_spans[:first_space]
        spans_part = type_and_spans[first_space+1:]
        spans = []
        for chunk in spans_part.split(';'):
            nums = chunk.strip().split()
            if len(nums) >= 2 and nums[0].isdigit() and nums[1].isdigit():
                s, e = int(nums[0]), int(nums[1])
                if e > s >= 0:
                    spans.append((s, e))
        if spans:
            ents.append({'id': tid, 'label': label, 'spans': spans, 'text': text})
    return ents

SCIENCEIE_LABELS = ["Material", "Process", "Task"]
SCIERC_LABELS     = ["Generic", "Material", "Method", "Metric", "OtherScientificTerm", "Task"]

def make_label_mapper(target_schema: str):
    if target_schema == "scienceie":
        keep = set(SCIENCEIE_LABELS)
        def mapper(lbl: str):
            if lbl == "Method":
                return "Process"
            return lbl if lbl in keep else None
        return keep, mapper
    elif target_schema == "scierc":
        keep = set(SCIERC_LABELS)
        def mapper(lbl: str):
            return lbl if lbl in keep else None
        return keep, mapper
    else:
        keep = None
        def mapper(lbl: str):
            return lbl
        return keep, mapper

def align_sentence_bio(text: str,
                       sent_span: Tuple[int,int],
                       tokens: List[Tuple[str,int,int]],
                       entities: List[Dict[str,Any]]):

    s0, e0 = sent_span
    idxs = [i for i,(_,s,e) in enumerate(tokens) if s >= s0 and e <= e0]
    sent_tokens = [tokens[i][0] for i in idxs]
    sent_offsets = [(tokens[i][1], tokens[i][2]) for i in idxs]
    labels = ["O"] * len(sent_tokens)

    ent_spans = []
    for ent in entities:
        for (es, ee) in ent["spans"]:
            if es >= s0 and ee <= e0:
                ent_spans.append((es, ee, ent["label"]))
    ent_spans.sort(key=lambda x: (x[1]-x[0]), reverse=True)

    used = [False]*len(sent_tokens)
    for (es, ee, lab) in ent_spans:
        tok_ids = [j for j,(ts,te) in enumerate(sent_offsets) if ts >= es and te <= ee]
        if not tok_ids:
            for j,(ts,te) in enumerate(sent_offsets):
                mid = (ts+te)//2
                if es <= mid < ee:
                    tok_ids.append(j)
            tok_ids = sorted(set(tok_ids))
        if not tok_ids:
            continue
        if any(used[j] for j in tok_ids):
            continue
        labels[tok_ids[0]] = f"B-{lab}"
        used[tok_ids[0]] = True
        for j in tok_ids[1:]:
            labels[j] = f"I-{lab}"
            used[j] = True
    return sent_tokens, labels

def collect_pairs(root: str):
    files = {}
    for p in glob.glob(os.path.join(root, "**"), recursive=True):
        if not os.path.isfile(p):
            continue
        base = os.path.basename(p)
        name, ext = os.path.splitext(base)
        ext = ext.lower()
        if ext not in [".txt",".ann"]:
            continue
        files.setdefault((name.lower(), ext), []).append(p)
    seen = {}
    for (name, ext), paths in files.items():
        for p in paths:
            seen.setdefault(name, {})[ext] = p
    pairs = []
    for name, m in seen.items():
        if ".txt" in m and ".ann" in m and os.path.getsize(m[".txt"]) > 0:
            pairs.append((m[".txt"], m[".ann"]))
    return sorted(pairs)

def process_split(in_dir: str, out_path: str, schema: str):
    splitter = get_sentence_splitter()
    keep_set, mapper = make_label_mapper(schema)
    n_sent = 0
    n_ents = 0
    by_label = {}

    with open(out_path, "w", encoding="utf-8") as out_f:
        pairs = collect_pairs(in_dir)
        for txt_path, ann_path in pairs:
            text = open(txt_path, "r", encoding="utf-8", errors="ignore").read()
            ann  = open(ann_path, "r", encoding="utf-8", errors="ignore").read()
            ents_raw = parse_brat_ann(ann)

            ents = []
            for e in ents_raw:
                mapped = mapper(e["label"])
                if mapped is None:
                    continue
                ents.append({'label': mapped, 'spans': e["spans"]})
            if not text.strip():
                continue

            sent_spans = splitter(text)
            tokens = tokenize_with_offsets(text)

            for s_span in sent_spans:
                s0,e0 = s_span
                if not text[s0:e0].strip():
                    continue
                toks, labs = align_sentence_bio(text, s_span, tokens, ents)
                if not toks:
                    continue
                for lab in labs:
                    if lab.startswith("B-"):
                        labn = lab[2:]
                        by_label[labn] = by_label.get(labn, 0) + 1
                        n_ents += 1
                out_f.write(json.dumps({"tokens": toks, "labels": labs}, ensure_ascii=False) + "\n")
                n_sent += 1

    return {"sentences": n_sent, "entities": n_ents, "by_label": by_label}

def write_label_list(out_dir: str, schema: str, seen_labels):
    outp = os.path.join(out_dir, "label_list.txt")
    if schema == "scienceie":
        labels = SCIENCEIE_LABELS
    elif schema == "scierc":
        labels = SCIERC_LABELS
    else:
        labels = sorted(seen_labels)
    with open(outp, "w", encoding="utf-8") as f:
        f.write("O\n")
        for lab in labels:
            f.write(f"B-{lab}\n")
            f.write(f"I-{lab}\n")
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--dev_dir",   required=True)
    ap.add_argument("--test_dir",  required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--target_schema", choices=["scienceie","scierc","union"], default="scienceie",
                    help="Output label schema: scienceie (3 classes; Method→Process), scierc (6 classes), union (keep all labels that appear)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    seen = set()
    for split, src in [("train", args.train_dir), ("dev", args.dev_dir), ("test", args.test_dir)]:
        outp = str(out_dir / f"{split}.jsonl")
        s = process_split(src, outp, schema=args.target_schema)
        stats[split] = s
        for k in s["by_label"].keys():
            seen.add(k)
        print(f"[{split}] sentences={s['sentences']}, entities={s['entities']}, by_label={s['by_label']}")

    final_labels = write_label_list(str(out_dir), args.target_schema, sorted(seen))
    labshow = ["O"] + sum(([f"B-{l}", f"I-{l}"] for l in final_labels), [])
    print(f"[Unified BRAT] target_schema={args.target_schema}, labels={labshow}")
    print(f"Output written to: {out_dir}")

if __name__ == "__main__":
    main()
