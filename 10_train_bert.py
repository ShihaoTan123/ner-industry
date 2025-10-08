import argparse, json, logging
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, Trainer, TrainingArguments,
    EarlyStoppingCallback, set_seed
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

def align_labels_with_tokens(tokenized_inputs, labels, label2id):
    out = []
    for i, labs in enumerate(labels):
        ids = tokenized_inputs.word_ids(batch_index=i)
        prev = None; arr = []
        for wid in ids:
            if wid is None: arr.append(-100)
            elif wid != prev:
                arr.append(label2id[labs[wid]] if wid < len(labs) else -100)
            else:
                arr.append(-100)
            prev = wid
        out.append(arr)
    return out

def make_metrics_fn(id2label):
    def _metrics_fn(eval_preds):
        preds, gold = eval_preds
        preds = np.argmax(preds, axis=-1)
        t_labels, t_preds = [], []
        for p, g in zip(preds, gold):
            pl, gl = [], []
            for pi, gi in zip(p, g):
                if gi != -100:
                    gl.append(id2label[int(gi)])
                    pl.append(id2label[int(pi)])
            if gl:
                t_labels.append(gl); t_preds.append(pl)
        return {
            "f1": f1_score(t_labels, t_preds, average="micro"),
            "precision": precision_score(t_labels, t_preds, average="micro"),
            "recall": recall_score(t_labels, t_preds, average="micro"),
        }
    return _metrics_fn

def main():
    ap = argparse.ArgumentParser(description="Minimal BERT NER Trainer")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev",   required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Cannot enable both FP16 and BF16.")

    set_seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    labels = [l.strip() for l in open(args.labels, encoding="utf-8") if l.strip()]
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in id2label.items()}
    log.info(f"#labels={len(labels)} -> {labels}")

    ds = load_dataset("json", data_files={"train": args.train, "validation": args.dev, "test": args.test})
    tok = AutoTokenizer.from_pretrained(args.model_name)

    def proc(batch):
        enc = tok(
            batch["tokens"], is_split_into_words=True,
            truncation=True, max_length=args.max_len, padding=False
        )
        enc["labels"] = align_labels_with_tokens(enc, batch["labels"], label2id)
        return enc

    cols = ds["train"].column_names
    ds_tok = ds.map(proc, batched=True, remove_columns=cols, desc="Tokenize+Align")

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    collator = DataCollatorForTokenClassification(
        tokenizer=tok, pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None
    )

    targs = TrainingArguments(
        output_dir=str(out),
        seed=args.seed,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(out / "logs"),
        logging_steps=50,
        save_total_limit=3,
        report_to=[],
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)] if args.early_stopping else None

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=make_metrics_fn(id2label),
        callbacks=callbacks,
    )

    log.info("== train ==")
    trainer.train()

    log.info("== test ==")
    test_metrics = trainer.evaluate(ds_tok["test"])
    pred_out = trainer.predict(ds_tok["test"])

    preds = np.argmax(pred_out.predictions, axis=-1)
    gold  = pred_out.label_ids

    y_true, y_pred = [], []
    for p, g in zip(preds, gold):
        tl, tp = [], []
        for pi, gi in zip(p, g):
            if gi != -100:
                tl.append(id2label[int(gi)])
                tp.append(id2label[int(pi)])
        if tl:
            y_true.append(tl); y_pred.append(tp)

    rep = classification_report(y_true, y_pred, digits=4)

    (out / "final_model").mkdir(parents=True, exist_ok=True)
    trainer.save_model(out / "final_model")
    json.dump(test_metrics, open(out / "test_metrics.json", "w"), indent=2, ensure_ascii=False)
    open(out / "test_report.txt", "w").write(rep)

    print("\n===== Test =====")
    print(f"Micro F1: {test_metrics.get('eval_f1', 0):.4f}")
    print(f"Precision: {test_metrics.get('eval_precision', 0):.4f}")
    print(f"Recall: {test_metrics.get('eval_recall', 0):.4f}")
    print(rep)

if __name__ == "__main__":
    main()
