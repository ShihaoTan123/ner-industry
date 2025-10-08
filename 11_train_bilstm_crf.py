import argparse, json, random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torchcrf import CRF
from tqdm import tqdm

PAD = "<PAD>"
UNK = "<UNK>"

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def build_vocab(train_examples: List[List[str]], min_freq=1):
    cnt = Counter()
    for toks in train_examples: cnt.update(toks)
    vocab = [PAD, UNK] + [w for w, c in cnt.items() if c >= min_freq]
    return {w: i for i, w in enumerate(vocab)}

class NERSet(Dataset):
    def __init__(self, tokens_list, labels_list, word2id, label2id):
        self.X, self.Y = [], []
        for toks, labs in zip(tokens_list, labels_list):
            x = [word2id.get(w, word2id[UNK]) for w in toks]
            y = [label2id[l] for l in labs]
            if len(x) == len(y):
                self.X.append(torch.tensor(x, dtype=torch.long))
                self.Y.append(torch.tensor(y, dtype=torch.long))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

def collate(batch, pad_id, pad_lab):
    xs, ys = zip(*batch)
    X = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_id)
    Y = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=pad_lab)
    mask = (X != pad_id)
    return X, Y, mask

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_labels, pad_id, dropout=0.5, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden // 2, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, X, mask, Y=None):
        h, _ = self.lstm(self.drop(self.emb(X)))
        emit = self.fc(self.drop(h))
        if Y is not None:
            return -self.crf(emit, Y, mask=mask, reduction='mean')
        return self.crf.decode(emit, mask=mask)

@torch.no_grad()
def evaluate(model, loader, id2label, device):
    model.eval()
    gold, pred = [], []
    for X, Y, M in loader:
        X, Y, M = X.to(device), Y.to(device), M.to(device)
        pred_ids = model(X, M)
        for i, seq in enumerate(pred_ids):
            m = M[i].bool()
            gold.append([id2label[int(t)] for t in Y[i][m].tolist()])
            pred.append([id2label[int(t)] for t in seq])
    return {
        "f1": f1_score(gold, pred, average="micro"),
        "precision": precision_score(gold, pred, average="micro"),
        "recall": recall_score(gold, pred, average="micro"),
        "report": classification_report(gold, pred, digits=4),
    }

def train_one_epoch(model, loader, opt, device, clip=5.0):
    model.train(); tot = 0.0
    for X, Y, M in tqdm(loader, desc="train", leave=False):
        X, Y, M = X.to(device), Y.to(device), M.to(device)
        opt.zero_grad()
        loss = model(X, M, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        tot += float(loss)
    return tot / len(loader)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--labels", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--emb_dim", type=int, default=100); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2); ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-3); ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5); ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    labels = [l.strip() for l in open(args.labels, encoding="utf-8") if l.strip()]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    pad_lab = label2id.get("O", 0)

    ds = load_dataset("json", data_files={"train": args.train, "dev": args.dev, "test": args.test})
    train_tokens = [ex["tokens"] for ex in ds["train"]]
    word2id = build_vocab(train_tokens, min_freq=args.min_freq)
    pad_id = word2id[PAD]

    def make_dataset(split):
        toks = [ex["tokens"] for ex in ds[split]]
        labs = [ex["labels"] for ex in ds[split]]
        return NERSet(toks, labs, word2id, label2id)

    train_set, dev_set, test_set = make_dataset("train"), make_dataset("dev"), make_dataset("test")
    mk_loader = lambda s, sh: DataLoader(s, batch_size=args.batch, shuffle=sh,
                                         collate_fn=lambda b: collate(b, pad_id, pad_lab))
    tr_loader, dv_loader, te_loader = mk_loader(train_set, True), mk_loader(dev_set, False), mk_loader(test_set, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_CRF(len(word2id), args.emb_dim, args.hidden, len(labels),
                       pad_id, dropout=args.dropout, num_layers=args.num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1, bad = 0.0, 0
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, tr_loader, opt, device)
        dev_metrics = evaluate(model, dv_loader, id2label, device)
        print(f"[epoch {ep}] loss={tr_loss:.4f}  dev_f1={dev_metrics['f1']:.4f}")

        if dev_metrics["f1"] > best_f1:
            best_f1, bad = dev_metrics["f1"], 0
            torch.save(model.state_dict(), out / "best_model.pt")
            meta = {"word2id": word2id, "label2id": label2id, "args": vars(args)}
            json.dump(meta, open(out / "meta.json", "w"), indent=2)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break

    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, te_loader, id2label, device)
    json.dump(
        {"best_dev_f1": best_f1, "test_f1": test_metrics["f1"],
         "test_precision": test_metrics["precision"], "test_recall": test_metrics["recall"]},
        open(out / "test_metrics.json", "w"), indent=2
    )
    open(out / "test_report.txt", "w").write(test_metrics["report"])
    print("\n=== Final (test) ===")
    print(f"F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  R={test_metrics['recall']:.4f}")
    print(test_metrics["report"])

if __name__ == "__main__":
    main()
