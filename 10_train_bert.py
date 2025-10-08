import argparse
import json
import numpy as np
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback,
    set_seed
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def align_labels_with_tokens(tokenized_inputs, labels, label2id):
    aligned_labels = []
    
    for i, label_sequence in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(label2id[label_sequence[word_idx]])
                except IndexError:
                    logger.warning(f"Label index out of range: word_idx={word_idx}, len(labels)={len(label_sequence)}")
                    label_ids.append(-100)
            else:
                label_ids.append(-100)
            
            previous_word_idx = word_idx
        
        aligned_labels.append(label_ids)
    
    return aligned_labels

def build_trainer(args):
    set_seed(args.seed)
    
    logger.info(f"Loading labels from {args.labels}")
    with open(args.labels, encoding="utf-8") as f:
        label_list = [line.strip() for line in f if line.strip()]
    
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    logger.info(f"Number of labels: {len(label_list)}")
    logger.info(f"Labels: {label_list}")
    
    logger.info("Loading datasets...")
    data_files = {
        "train": args.train,
        "validation": args.dev,
        "test": args.test
    }
    
    datasets = load_dataset("json", data_files=data_files)
    logger.info(f"Dataset sizes - Train: {len(datasets['train'])}, Val: {len(datasets['validation'])}, Test: {len(datasets['test'])}")
    
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_len,
            padding=False
        )
        
        tokenized_inputs["labels"] = align_labels_with_tokens(
            tokenized_inputs, 
            examples["labels"], 
            label2id
        )
        
        return tokenized_inputs
    
    logger.info("Processing datasets...")
    column_names = datasets["train"].column_names
    tokenized_datasets = datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing and aligning labels"
    )
    
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None
    )
    
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=-1)
        
        true_labels = []
        true_predictions = []
        
        for prediction, label in zip(predictions, labels):
            true_label = []
            true_prediction = []
            
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:
                    true_label.append(id2label[label_id])
                    true_prediction.append(id2label[pred_id])
            
            if true_label:
                true_labels.append(true_label)
                true_predictions.append(true_prediction)
        
        results = {
            "f1": f1_score(true_labels, true_predictions, average="micro"),
            "precision": precision_score(true_labels, true_predictions, average="micro"),
            "recall": recall_score(true_labels, true_predictions, average="micro"),
        }
        
        return results
    
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        seed=args.seed,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        warmup_ratio=0.06,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"{args.out_dir}/logs",
        logging_steps=50,
        save_total_limit=3,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=4 if not args.debug else 0,
        remove_unused_columns=False,
    )
    
    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    return trainer, tokenized_datasets, label_list

def evaluate_and_save_results(trainer, test_dataset, label_list, output_dir):
    logger.info("Evaluating on test set...")
    
    test_metrics = trainer.evaluate(test_dataset)
    
    predictions_output = trainer.predict(test_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids
    
    id2label = {i: label for i, label in enumerate(label_list)}
    
    true_labels = []
    true_predictions = []
    
    for prediction, label in zip(predictions, labels):
        true_label = []
        true_prediction = []
        
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:
                true_label.append(id2label[label_id])
                true_prediction.append(id2label[pred_id])
        
        if true_label:
            true_labels.append(true_label)
            true_predictions.append(true_prediction)
    
    report = classification_report(true_labels, true_predictions, digits=4)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_path / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics saved to {metrics_path}")
    
    report_path = output_path / "test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    
    print("\n" + "="*50)
    print("Test Results")
    print("="*50)
    print(f"Micro F1: {test_metrics.get('eval_f1', 0):.4f}")
    print(f"Precision: {test_metrics.get('eval_precision', 0):.4f}")
    print(f"Recall: {test_metrics.get('eval_recall', 0):.4f}")
    print("\nDetailed Classification Report:")
    print(report)
    
    return test_metrics

def main():
    parser = argparse.ArgumentParser(description="Train NER model using Transformers")
    
    parser.add_argument("--train", required=True, help="Path to training data (JSONL)")
    parser.add_argument("--dev", required=True, help="Path to validation data (JSONL)")
    parser.add_argument("--test", required=True, help="Path to test data (JSONL)")
    parser.add_argument("--labels", required=True, help="Path to label list file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    
    parser.add_argument("--model_name", default="bert-base-uncased", 
                        help="Pretrained model name or path")
    parser.add_argument("--max_len", type=int, default=256, 
                        help="Maximum sequence length")
    
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-5, 
                        help="Learning rate")
    parser.add_argument("--batch", type=int, default=16, 
                        help="Batch size")
    parser.add_argument("--eval_steps", type=int, default=200, 
                        help="Evaluation frequency (steps)")
    parser.add_argument("--seed", type=int, default=13, 
                        help="Random seed")
    
    parser.add_argument("--fp16", action="store_true", 
                        help="Use FP16 mixed precision training")
    parser.add_argument("--bf16", action="store_true", 
                        help="Use BF16 mixed precision training")
    parser.add_argument("--early_stopping", action="store_true", 
                        help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=3, 
                        help="Early stopping patience")
    
    parser.add_argument("--debug", action="store_true", 
                        help="Debug mode (disable multiprocessing)")
    
    args = parser.parse_args()
    
    if args.fp16 and args.bf16:
        raise ValueError("Cannot use both FP16 and BF16 at the same time")
    
    trainer, tokenized_datasets, label_list = build_trainer(args)
    
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")
    
    evaluate_and_save_results(
        trainer, 
        tokenized_datasets["test"], 
        label_list, 
        args.out_dir
    )
    
    final_model_path = Path(args.out_dir) / "final_model"
    trainer.save_model(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
