import argparse, json, numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback)
from seqeval.metrics import f1_score, classification_report

def build(args):
    labels=[l.strip() for l in open(args.labels,encoding="utf-8")]
    id2={i:l for i,l in enumerate(labels)}; l2={l:i for i,l in enumerate(labels)}
    ds=load_dataset("json",data_files={"train":args.train,"validation":args.dev,"test":args.test})
    tok=AutoTokenizer.from_pretrained(args.model_name)

    def align(ex):
        out=tok(ex["tokens"],is_split_into_words=True,truncation=True,max_length=args.max_len)
        labs=[]
        for i in range(len(ex["tokens"])):
            ids=out.word_ids(i); L=ex["labels"][i]; a=[]; prev=None
            for wid in ids:
                if wid is None: a.append(-100)
                elif wid!=prev: a.append(l2[L[wid]])
                else: a.append(-100)
                prev=wid
            labs.append(a)
        out["labels"]=labs; return out

    cols=ds["train"].column_names
    ds=ds.map(align, batched=True, remove_columns=cols)
    model=AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(labels), id2label=id2, label2id=l2)
    collator=DataCollatorForTokenClassification(tokenizer=tok)

    def metrics(p):
        pred=np.argmax(p.predictions,axis=-1); lab=p.label_ids; P,L=[],[]
        for pr,lb in zip(pred,lab):
            pl,ll=[],[]
            for pi,li in zip(pr,lb):
                if li==-100: continue
                pl.append(id2[pi]); ll.append(id2[li])
            P.append(pl); L.append(ll)
        return {"f1": f1_score(L,P)}

    args_hf=TrainingArguments(
        output_dir=args.out_dir, seed=args.seed, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs, warmup_ratio=0.06, weight_decay=0.01,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps,
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        logging_steps=50, report_to="none",
        fp16=args.fp16, bf16=args.bf16
    )
    trainer=Trainer(model=model,args=args_hf,train_dataset=ds["train"],eval_dataset=ds["validation"],
                    tokenizer=tok,data_collator=collator,compute_metrics=metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    return trainer, ds, labels

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--labels", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=13); ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--bf16", action="store_true")
    args=ap.parse_args()

    trainer, ds, labels = build(args)
    trainer.train()
    test_metrics=trainer.evaluate(ds["test"])

    pred=trainer.predict(ds["test"]).predictions
    lab =trainer.predict(ds["test"]).label_ids
    import numpy as np
    pred=np.argmax(pred,axis=-1)
    id2={i:l for i,l in enumerate(labels)}
    P,L=[],[]
    for pr,lb in zip(pred,lab):
        pl,ll=[],[]
        for pi,li in zip(pr,lb):
            if li==-100: continue
            pl.append(id2[pi]); ll.append(id2[li])
        P.append(pl); L.append(ll)
    from seqeval.metrics import classification_report
    rep=classification_report(L,P,digits=4)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.out_dir}/test_metrics.json").write_text(json.dumps(test_metrics,indent=2),encoding="utf-8")
    Path(f"{args.out_dir}/test_report.txt").write_text(rep,encoding="utf-8")
    print("[TEST] micro-F1:", test_metrics.get("eval_f1"))
    print(rep)

if __name__=="__main__": main()
