import argparse
import json
import random
import numpy as np
import torch
import logging
from pathlib import Path
from collections import Counter
from torch import nn
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torchcrf import CRF
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PAD = "<PAD>"
UNK = "<UNK>"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_jsonl(file_path):
    data = []
    with open(file_path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            try:
                ex = json.loads(line)
                if "tokens" in ex and "labels" in ex:
                    data.append((ex["tokens"], ex["labels"]))
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {line_idx} in {file_path}: {e}")
    logger.info(f"Loaded {len(data)} examples from {file_path}")
    return data

class NERDataset(Dataset):
    
    def __init__(self, data, word2id, label2id):
        self.word2id = word2id
        self.label2id = label2id
        
        self.X = []
        self.Y = []
        
        for tokens, labels in data:
            token_ids = [self.word2id.get(w, self.word2id[UNK]) for w in tokens]
            label_ids = [self.label2id[l] for l in labels]
            
            if len(token_ids) != len(label_ids):
                logger.warning(f"Length mismatch: {len(token_ids)} tokens vs {len(label_ids)} labels")
                continue
                
            self.X.append(token_ids)
            self.Y.append(label_ids)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.Y[idx], dtype=torch.long)

def collate_fn(batch, pad_id, pad_label_id):
    sequences, labels = zip(*batch)
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    
    X_padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    Y_padded = torch.full((batch_size, max_len), pad_label_id, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    for i, (seq, lab) in enumerate(zip(sequences, labels)):
        seq_len = len(seq)
        X_padded[i, :seq_len] = seq
        Y_padded[i, :seq_len] = lab
        mask[i, :seq_len] = True
    
    return X_padded, Y_padded, mask

class BiLSTM_CRF(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, 
                 pad_idx, dropout=0.5, num_layers=2):
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        
        self.crf = CRF(num_labels, batch_first=True)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, sentences, mask, labels=None):
        embeddings = self.embedding(sentences)
        embeddings = self.dropout(embeddings)
        
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        
        emissions = self.hidden2tag(lstm_out)
        
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)

def build_vocabulary(train_data, min_freq=2):
    word_counter = Counter()
    
    for tokens, _ in train_data:
        word_counter.update(tokens)
    
    vocab = [PAD, UNK]
    vocab.extend([word for word, count in word_counter.items() if count >= min_freq])
    
    word2id = {word: idx for idx, word in enumerate(vocab)}
    logger.info(f"Vocabulary size: {len(vocab)} (min_freq={min_freq})")
    logger.info(f"Filtered out {len(word_counter) - len(vocab) + 2} low-frequency words")
    
    return vocab, word2id

def evaluate(model, data_loader, id2label, device):
    model.eval()
    predictions = []
    gold_labels = []
    
    with torch.no_grad():
        for X, Y, mask in tqdm(data_loader, desc="Evaluating", leave=False):
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)
            
            pred_ids = model(X, mask)
            
            for i in range(len(pred_ids)):
                seq_mask = mask[i]
                
                gold = [id2label[int(label_id)] for label_id in Y[i][seq_mask].tolist()]
                pred = [id2label[pred_id] for pred_id in pred_ids[i]]
                
                gold_labels.append(gold)
                predictions.append(pred)
    
    f1 = f1_score(gold_labels, predictions, average='micro')
    precision = precision_score(gold_labels, predictions, average='micro')
    recall = recall_score(gold_labels, predictions, average='micro')
    report = classification_report(gold_labels, predictions, digits=4)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': report
    }

def train_epoch(model, train_loader, optimizer, device, max_grad_norm=5.0):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for X, Y, mask in progress_bar:
        X = X.to(device)
        Y = Y.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        loss = model(X, mask, Y)
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)

def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM-CRF model for NER")
    
    parser.add_argument("--train", required=True, help="Path to training data")
    parser.add_argument("--dev", required=True, help="Path to dev data")
    parser.add_argument("--test", required=True, help="Path to test data")
    parser.add_argument("--labels", required=True, help="Path to labels file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    
    parser.add_argument("--emb_dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum word frequency")
    
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading labels from {args.labels}")
    with open(args.labels, encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    logger.info(f"Number of labels: {len(labels)}")
    
    logger.info("Loading datasets...")
    train_data = read_jsonl(args.train)
    dev_data = read_jsonl(args.dev)
    test_data = read_jsonl(args.test)
    
    vocab, word2id = build_vocabulary(train_data, min_freq=args.min_freq)
    
    train_dataset = NERDataset(train_data, word2id, label2id)
    dev_dataset = NERDataset(dev_data, word2id, label2id)
    test_dataset = NERDataset(test_data, word2id, label2id)
    
    pad_label_id = label2id.get('O', 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, word2id[PAD], pad_label_id),
        num_workers=0
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, word2id[PAD], pad_label_id),
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, word2id[PAD], pad_label_id),
        num_workers=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = BiLSTM_CRF(
        vocab_size=len(vocab),
        embedding_dim=args.emb_dim,
        hidden_dim=args.hidden,
        num_labels=len(labels),
        pad_idx=word2id[PAD],
        dropout=args.dropout,
        num_layers=args.num_layers
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    best_f1 = 0.0
    bad_epochs = 0
    
    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, device)
        
        dev_metrics = evaluate(model, dev_loader, id2label, device)
        
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"  Train Loss: {avg_loss:.4f}")
        logger.info(f"  Dev F1: {dev_metrics['f1']:.4f}")
        logger.info(f"  Dev Precision: {dev_metrics['precision']:.4f}")
        logger.info(f"  Dev Recall: {dev_metrics['recall']:.4f}")
        
        if dev_metrics['f1'] > best_f1:
            best_f1 = dev_metrics['f1']
            bad_epochs = 0
            
            model_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'vocab': vocab,
                'word2id': word2id,
                'label2id': label2id,
                'args': args
            }, model_path)
            
            logger.info(f"  New best model saved with F1: {best_f1:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
    
    logger.info("Loading best model for final evaluation...")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, id2label, device)
    
    results = {
        'best_dev_f1': best_f1,
        'test_f1': test_metrics['f1'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall']
    }
    
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / "test_report.txt", "w", encoding="utf-8") as f:
        f.write(test_metrics['report'])
    
    print("\n" + "="*50)
    print("Final Test Results")
    print("="*50)
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print("\nDetailed Report:")
    print(test_metrics['report'])

if __name__ == "__main__":
    main()
