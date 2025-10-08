import argparse, re, json
from pathlib import Path
from typing import List, Tuple

TOKEN_PAT = re.compile(r"\w+|[^\w\s]")

def tokenize_with_offsets(text:str):
    toks, spans = [], []
    for m in TOKEN_PAT.finditer(text):
        toks.append(m.group(0))
        spans.append((m.start(), m.end()))
    return toks, spans

def read_brat_ann(path:Path):
    ents=[]
    if not path.exists(): return ents
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("T"): continue
        m=re.match(r"^T\d+\t([A-Za-z_]+)\s+(\d+)\s+(\d+)\t", line)
        if m:
            typ, s, e = m.group(1), int(m.group(2)), int(m.group(3))
            ents.append((s,e,typ)); continue
        m2=re.match(r"^T\d+\t([A-Za-z_]+)\s+([\d; ]+)\t", line)
        if m2:
            typ=m2.group(1)
            parts=[p.strip() for p in m2.group(2).split(";")]
            for p in parts:
                a,b=p.split()
                ents.append((int(a), int(b), typ))
    return ents

def char2token_spans(ent_span:Tuple[int,int], tok_spans:List[Tuple[int,int]]):
    s_char,e_char=ent_span
    covered=[]
    for i,(ts,te) in enumerate(tok_spans):
        if te<=s_char or ts>=e_char: continue
        covered.append(i)
    if not covered: return None
    return (covered[0], covered[-1])

def greedy_resolve(spans):
    spans=sorted(spans, key=lambda x: (-(x[1]-x[0]+1), x[0], x[1]))
    selected=[]; used=set()
    for s,e,t in spans:
        if any(i in used for i in range(s, e+1)): continue
        selected.append((s,e,t))
        for i in range(s, e+1): used.add(i)
    return sorted(selected, key=lambda x:(x[0],x[1]))

def to_bio(n:int, spans):
    labs=["O"]*n
    for s,e,t in spans:
        labs[s]=f"B-{t}"
        for i in range(s+1, e+1): labs[i]=f"I-{t}"
    return labs

def prepare_split(src_dir:Path, out_path:Path, label_set:set):
    docs=toks_total=ents_total=0
    with out_path.open("w",encoding="utf-8") as w:
        for txt in sorted(src_dir.glob("*.txt")):
            ann = txt.with_suffix(".ann")
            text=txt.read_text(encoding="utf-8", errors="ignore")
            ents=read_brat_ann(ann)
            tokens, tok_spans = tokenize_with_offsets(text)
            spans_tok=[]
            for s_char,e_char,typ in ents:
                ts=char2token_spans((s_char,e_char), tok_spans)
                if ts is None: continue
                spans_tok.append((ts[0], ts[1], {"Material":"Material","Task":"Task","Process":"Process"}.get(typ, typ)))
            spans_tok = greedy_resolve(spans_tok)
            labels = to_bio(len(tokens), spans_tok)
            for L in labels:
                if L!="O": label_set.add(L[2:])
            w.write(json.dumps({"tokens":tokens, "labels":labels}, ensure_ascii=False)+"\n")
            docs+=1; toks_total+=len(tokens); ents_total+=sum(1 for _ in spans_tok)
    return docs, toks_total, ents_total

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_dir", default="scienceie2017_train/train2")
    ap.add_argument("--dev_dir",   default="dev")
    ap.add_argument("--test_dir",  default="semeval_articles_test")
    args=ap.parse_args()

    raw=Path(args.raw_dir); out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    label_set=set()
    for name, sub in [("train",args.train_dir),("dev",args.dev_dir),("test",args.test_dir)]:
        d, t, e = prepare_split(raw/sub, out/f"{name}.jsonl", label_set)
        print(f"[ScienceIE] {name}: docs={d}, tokens={t}, entities={e}")
    labels=["O"]+["B-Material","B-Process","B-Task","I-Material","I-Process","I-Task"]
    (out/"label_list.txt").write_text("\n".join(labels), encoding="utf-8")
    print("[ScienceIE] labels:", labels)

if __name__=="__main__": main()
