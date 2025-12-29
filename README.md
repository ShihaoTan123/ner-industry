# Scientific NER (Industry Project) — Preprocessing + Training Code

This repository contains the **data preprocessing** and **model training** code for a scientific Named Entity Recognition (NER) project.
It supports:
- Preparing datasets (SciERC, ScienceIE) into a unified format
- Training Transformer baselines (BERT) and a BiLSTM-CRF baseline

> Note: This repository currently focuses on preprocessing and training scripts only.
> Diagnostic/error-analysis code and report assets are maintained separately (not included here).

---

## What’s Included
- Dataset preprocessing:
  - `01_prepare_scierc.py`
  - `02_prepare_scienceie.py`
  - `03_prepare_both.py`
- Training:
  - `10_train_bert.py`
  - `11_train_bilstm_crf.py`
- Repository notes:
  - `repo structure` (a planning/structure note)

---

## Datasets
This project uses **public datasets**:
- SciERC
- ScienceIE

Datasets are **not redistributed** in this repository. Please download them from official sources and place them in your local data directory.

---

## Environment
Python 3.x is recommended.

Install dependencies (example):
```bash
pip install -r requirements.txt
