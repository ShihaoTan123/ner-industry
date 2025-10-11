import argparse, json, logging
import torch, torch.nn as nn
from pathlib import Path
import numpy as np
from datasets import load_dataset
from torchcrf import CRF
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, Trainer, TrainingArguments,
    EarlyStoppingCallback, set_seed,
    AutoModel, AutoConfig
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

class WeightedTokenTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is None:
            raise ValueError("class_weights is required")
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: int | None = None, **kwargs):
        labels = inputs.get("labels")
        if "labels" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**inputs)
        logits  = outputs.logits
        cw = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        loss_fct = nn.CrossEntropyLoss(weight=cw, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class CRFTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        has_labels = "labels" in inputs
        labels = inputs.get("labels")
        with torch.no_grad():
            net_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**net_inputs)
            emissions = outputs["logits"].float()
            mask = inputs["attention_mask"].bool()
            paths = model.crf.decode(emissions, mask=mask)
        max_len = emissions.size(1)
        pred_ids = []
        for seq in paths:
            if len(seq) < max_len:
                seq = seq + [-100] * (max_len - len(seq))
            pred_ids.append(seq)
        pred = torch.tensor(pred_ids, device=emissions.device, dtype=torch.long)
        loss = None
        if has_labels:
            O_ID = model.config.label2id.get("O", 0)
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = O_ID
            with torch.no_grad():
                loss = - model.crf(emissions, labels_crf.long(), mask=mask, reduction='mean')
            loss = loss.detach()
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, pred, labels)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

class BertCRFForTokenClassification(nn.Module):
    def __init__(self, base_model_name, num_labels, id2label, label2id, scheme="BIOES", constraints=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.config = AutoConfig.from_pretrained(base_model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)
        self.scheme = scheme
        if constraints is not None:
            BIG_NEG = -1e4
            with torch.no_grad():
                self.crf.transitions[:] = BIG_NEG
                for (i, j) in constraints["trans"]:
                    self.crf.transitions[i, j] = 0.0
                self.crf.start_transitions[:] = BIG_NEG
                self.crf.end_transitions[:] = BIG_NEG
                for i in constraints["start_ok"]:
                    self.crf.start_transitions[i] = 0.0
                for i in constraints["end_ok"]:
                    self.crf.end_transitions[i] = 0.0
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        emissions = self.classifier(out.last_hidden_state).float()
        mask = attention_mask.bool() if attention_mask is not None else torch.ones(emissions.size()[:2], dtype=torch.bool, device=emissions.device)
        loss = None
        if labels is not None:
            O_ID = self.config.label2id.get("O", 0)
            labels_filled = labels.clone()
            labels_filled[labels_filled == -100] = O_ID
            loss = - self.crf(emissions, labels_filled.long(), mask=mask, reduction='mean')
        return {"loss": loss, "logits": emissions, "mask": mask}

def align_labels_with_tokens(tokenized_inputs, labels, label2id):
    out = []
    for i, labs in enumerate(labels):
        ids = tokenized_inputs.word_ids(batch_index=i)
        prev = None
        arr = []
        for wid in ids:
            if wid is None:
                arr.append(-100)
            elif wid != prev:
                arr.append(label2id[labs[wid]] if wid < len(labs) else -100)
            else:
                arr.append(-100)
            prev = wid
        out.append(arr)
    return out

def parse_tag_type(label: str):
    if label == "O":
        return "O", None
    t, typ = label.split("-", 1)
    return t, typ

def allowed_transitions(labels, scheme="BIOES"):
    ok = set()
    for i, L1 in enumerate(labels):
        t1, c1 = parse_tag_type(L1)
        for j, L2 in enumerate(labels):
            t2, c2 = parse_tag_type(L2)
            legal = False
            if scheme == "BIO":
                if t1 == "O":
                    legal = (t2 in ("O", "B"))
                elif t1 == "B":
                    legal = (t2 in ("I", "O", "B")) and (t2 != "I" or c2 == c1)
                elif t1 == "I":
                    legal = (t2 in ("I", "O", "B")) and (t2 != "I" or c2 == c1)
            else:
                if t1 == "O":
                    legal = (t2 in ("O", "B", "S"))
                elif t1 == "B":
                    legal = (t2 in ("I", "E")) and (c2 == c1)
                elif t1 == "I":
                    legal = (t2 in ("I", "E")) and (c2 == c1)
                elif t1 == "E":
                    legal = (t2 in ("O", "B", "S"))
                elif t1 == "S":
                    legal = (t2 in ("O", "B", "S"))
            if legal:
                ok.add((i, j))
    return ok

def start_end_allowed(labels, scheme="BIOES"):
    start_ok, end_ok = set(), set()
    for i, L in enumerate(labels):
        t, _ = parse_tag_type(L)
        if scheme == "BIO":
            if L == "O" or t in ("B",):
                start_ok.add(i)
            if L == "O" or t in ("I",):
                end_ok.add(i)
        else:
            if L == "O" or t in ("B", "S"):
                start_ok.add(i)
            if L == "O" or t in ("E", "S"):
                end_ok.add(i)
    return start_ok, end_ok

def make_metrics_fn(id2label, num_labels):
    def _metrics_fn(eval_preds):
        if hasattr(eval_preds, "predictions"):
            preds, gold = eval_preds.predictions, eval_preds.label_ids
        else:
            preds, gold = eval_preds
        preds = np.array(preds)
        if preds.ndim == 3 and preds.shape[-1] == num_labels:
            preds = np.argmax(preds, axis=-1)
        t_labels, t_preds = [], []
        for p, g in zip(preds, gold):
            pl, gl = [], []
            for pi, gi in zip(p, g):
                if gi != -100:
                    gl.append(id2label[int(gi)])
                    pl.append(id2label[int(pi)])
            if gl:
                t_labels.append(gl)
                t_preds.append(pl)
        return {
            "f1": f1_score(t_labels, t_preds, average="micro"),
            "precision": precision_score(t_labels, t_preds, average="micro"),
            "recall": recall_score(t_labels, t_preds, average="micro"),
        }
    return _metrics_fn

def main():
    ap = argparse.ArgumentParser(description="Minimal BERT NER Trainer")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
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
    ap.add_argument("--b_weight", type=float, default=2.0)
    ap.add_argument("--task_weight", type=float, default=1.5)
    ap.add_argument("--use_crf", action="store_true")
    ap.add_argument("--scheme", choices=["BIO", "BIOES"], default="BIOES")
    args = ap.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Cannot enable both FP16 and BF16.")
    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = [l.strip() for l in open(args.labels, encoding="utf-8") if l.strip()]
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in id2label.items()}
    log.info(f"#labels={len(labels)} -> {labels}")

    ds = load_dataset("json", data_files={"train": args.train, "validation": args.dev, "test": args.test})
    tok = AutoTokenizer.from_pretrained(args.model_name)

    def proc(batch):
        enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=args.max_len, padding=False)
        enc["labels"] = align_labels_with_tokens(enc, batch["labels"], label2id)
        return enc

    cols = ds["train"].column_names
    ds_tok = ds.map(proc, batched=True, remove_columns=cols, desc="Tokenize+Align")

    collator = DataCollatorForTokenClassification(tokenizer=tok, pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None)

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

    if args.use_crf:
        trans_ok = allowed_transitions(labels, scheme=args.scheme)
        start_ok, end_ok = start_end_allowed(labels, scheme=args.scheme)
        constraints = {"trans": trans_ok, "start_ok": start_ok, "end_ok": end_ok}
        model = BertCRFForTokenClassification(
            base_model_name=args.model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            scheme=args.scheme,
            constraints=constraints,
        )
        trainer = CRFTrainer(
            model=model,
            args=targs,
            train_dataset=ds_tok["train"],
            eval_dataset=ds_tok["validation"],
            processing_class=tok,
            data_collator=collator,
            compute_metrics=make_metrics_fn(id2label, num_labels=len(labels)),
            callbacks=callbacks,
        )
    else:
        class_weights = torch.ones(len(labels))
        for lab, idx in label2id.items():
            if lab.startswith("B-"):
                class_weights[idx] *= args.b_weight
            if lab.endswith("Task"):
                class_weights[idx] *= args.task_weight
        log.info("Class weights: " + ", ".join(f"{lab}:{float(class_weights[label2id[lab]]):.2f}" for lab in labels))
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
        )
        trainer = WeightedTokenTrainer(
            model=model,
            args=targs,
            train_dataset=ds_tok["train"],
            eval_dataset=ds_tok["validation"],
            processing_class=tok,
            data_collator=collator,
            compute_metrics=make_metrics_fn(id2label, num_labels=len(labels)),
            callbacks=callbacks,
            class_weights=class_weights,
        )

    log.info("== train ==")
    trainer.train()

    log.info("== test ==")
    test_metrics = trainer.evaluate(ds_tok["test"])
    pred_out = trainer.predict(ds_tok["test"])

    preds = pred_out.predictions
    gold = pred_out.label_ids
    preds = np.array(preds)
    if preds.ndim == 3 and preds.shape[-1] == len(labels):
        preds = np.argmax(preds, axis=-1)

    y_true, y_pred = [], []
    for p, g in zip(preds, gold):
        tl, tp = [], []
        for pi, gi in zip(p, g):
            if gi != -100:
                tl.append(id2label[int(gi)])
                tp.append(id2label[int(pi)])
        if tl:
            y_true.append(tl)
            y_pred.append(tp)

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

