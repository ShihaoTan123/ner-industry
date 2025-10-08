import argparse, json, random, numpy as np, torch
from pathlib import Path
from collections import Counter
from torch import nn
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import f1_score, classification_report
from torchcrf import CRF

PAD="<PAD>"; UNK="<UNK>"

def read_jsonl(p):
    data=[]
    with open(p,encoding="utf-8") as f:
        for line in f:
            ex=json.loads(line); data.append((ex["tokens"], ex["labels"]))
    return data

class NERSet(Dataset):
    def __init__(self,data,word2id,label2id):
        self.w2i=word2id; self.l2i=label2id
        self.X=[[self.w2i.get(w, self.w2i[UNK]) for w in x] for x,_ in data]
        self.Y=[[self.l2i[l] for l in y] for _,y in data]
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        return torch.tensor(self.X[i]), torch.tensor(self.Y[i])

def pad_batch(batch, pad_id, pad_label_id):
    xs, ys = zip(*batch)
    maxlen = max(len(x) for x in xs)
    Xp = torch.full((len(xs), maxlen), pad_id, dtype=torch.long)
    Yp = torch.full((len(xs), maxlen), pad_label_id, dtype=torch.long)
    mask = torch.zeros((len(xs), maxlen), dtype=torch.bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        L = len(x)
        Xp[i, :L] = x
        Yp[i, :L] = y
        mask[i, :L] = 1
    return Xp, Yp, mask

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_labels, pad_id, dropout=0.33):
        super().__init__()
        self.emb=nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm=nn.LSTM(emb_dim, hidden//2, batch_first=True, bidirectional=True)
        self.drop=nn.Dropout(dropout)
        self.fc=nn.Linear(hidden, num_labels)
        self.crf=CRF(num_labels, batch_first=True)
    def forward(self, X, mask, Y=None):
        em=self.emb(X); o,_=self.lstm(em); o=self.drop(o); ems=self.fc(o)
        if Y is not None:
            return -self.crf(ems, Y, mask=mask, reduction='mean')
        else:
            return self.crf.decode(ems, mask=mask)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--labels", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--emb_dim", type=int, default=100); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-3); ap.add_argument("--seed", type=int, default=13)
    args=ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    labels=[l.strip() for l in open(args.labels,encoding="utf-8")]
    label2id={l:i for i,l in enumerate(labels)}; id2label={i:l for l,i in label2id.items()}

    train=read_jsonl(args.train); dev=read_jsonl(args.dev); test=read_jsonl(args.test)
    cnt=Counter([w for x,_ in train for w in x]); vocab=[PAD,UNK]+[w for w,c in cnt.items() if c>=1]
    word2id={w:i for i,w in enumerate(vocab)}

    pad_label_id = label2id['O']
    train_loader=DataLoader(NERSet(train,word2id,label2id), batch_size=args.batch, shuffle=True,  collate_fn=lambda b:pad_batch(b, word2id[PAD], pad_label_id), num_workers=2, pin_memory=True)
    dev_loader  =DataLoader(NERSet(dev,  word2id,label2id), batch_size=args.batch, shuffle=False, collate_fn=lambda b:pad_batch(b, word2id[PAD], pad_label_id), num_workers=2, pin_memory=True)
    test_loader =DataLoader(NERSet(test, word2id,label2id), batch_size=args.batch, shuffle=False, collate_fn=lambda b:pad_batch(b, word2id[PAD], pad_label_id), num_workers=2, pin_memory=True)

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=BiLSTM_CRF(vocab_size=len(vocab), emb_dim=args.emb_dim, hidden=args.hidden, num_labels=len(labels),
                     pad_id=word2id[PAD], dropout=0.33).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=args.lr)

    def evaluate(loader):
        model.eval(); P=[]; L=[]
        with torch.no_grad():
            for X,Y,mask in loader:
                X=X.to(device); Y=Y.to(device); m=mask.to(device)
                pred=model(X,m)
                for i in range(len(pred)):
                    gold=[id2label[int(t)] for t in Y[i][m[i]].tolist()]
                    out =[id2label[j] for j in pred[i]]
                    L.append(gold); P.append(out)
        return f1_score(L,P), classification_report(L,P,digits=4)

    best, bad, patience = 0.0, 0, 3
    for ep in range(1, args.epochs+1):
        model.train(); total=0.0
        for X,Y,mask in train_loader:
            X=X.to(device); Y=Y.to(device); m=mask.to(device)
            opt.zero_grad(); loss=model(X,m,Y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); total += float(loss)
        f1, _ = evaluate(dev_loader)
        print(f"[epoch {ep}] loss={total/len(train_loader):.4f}  dev_f1={f1:.4f}")
        if f1>best:
            best=f1; bad=0
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{args.out_dir}/best.pt")
        else:
            bad+=1
            if bad>=patience: print("early stop"); break

    model.load_state_dict(torch.load(f"{args.out_dir}/best.pt", map_location=device))
    f1, rep = evaluate(test_loader)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.out_dir}/test_report.txt").write_text(rep, encoding="utf-8")
    print("[TEST] micro-F1:", f1); print(rep)

if __name__=="__main__": main()
