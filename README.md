<div align="center">

# Towards Robust Online Sexism Detection for the EXIST 2023 Shared Task

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8144300.svg)](https://doi.org/10.5281/zenodo.8144300)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*BERT + XLM-RoBERTa + DistilBERT voting ensemble for sexism identification and categorization in English and Spanish tweets.*

</div>

## Paper

|                  |                                                                          |
| ---------------- | ------------------------------------------------------------------------ |
| **Title**        | Towards Robust Online Sexism Detection: A Multi-Model Approach with BERT, XLM-RoBERTa, and DistilBERT for EXIST 2023 Tasks |
| **Authors**      | Hadi Mohammadi, Anastasia Giachanou, Robert A. Bagheri |
| **Affiliation**  | Department of Methodology and Statistics, Utrecht University, The Netherlands |
| **Venue**        | Working Notes of CLEF 2023 (CEUR Workshop Proceedings, Vol. 3497), pp. 1000–1011 |
| **Paper**        | [ceur-ws.org/Vol-3497/paper-085.pdf](https://ceur-ws.org/Vol-3497/paper-085.pdf) |
| **Code archive** | [10.5281/zenodo.8144300](https://doi.org/10.5281/zenodo.8144300) (this repository) |

> System of the **M&S NLP team** for the [EXIST 2023 shared task](http://nlp.uned.es/exist2023/) (sEXism Identification in Social neTworks) at CLEF 2023.
> The repository [Sexism-Detection-in-Social-Media](https://github.com/mohammadi-hadi/Sexism-Detection-in-Social-Media) contains an identical copy of this notebook.

## Abstract

The EXIST 2023 shared task targets the identification and categorization of sexism in English and Spanish tweets across three subtasks: binary sexism identification, source-intention classification (direct, reported, or judgemental), and multi-label sexism categorization. Our approach combines three multilingual transformer encoders — BERT, XLM-RoBERTa, and DistilBERT — with a voting ensemble, on top of text preprocessing, data augmentation, and class-imbalance handling. As reported in the paper, the system performed best on English, and combining the outputs of multiple models via voting improved robustness. Detailed results on the official EXIST 2023 test set are reported in the [paper](https://ceur-ws.org/Vol-3497/paper-085.pdf).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{mohammadi2023towards,
  title     = {Towards Robust Online Sexism Detection: A Multi-Model Approach with BERT, XLM-RoBERTa, and DistilBERT for EXIST 2023 Tasks},
  author    = {Mohammadi, Hadi and Giachanou, Anastasia and Bagheri, Robert A.},
  booktitle = {Working Notes of CLEF 2023 -- Conference and Labs of the Evaluation Forum},
  series    = {CEUR Workshop Proceedings},
  volume    = {3497},
  pages     = {1000--1011},
  year      = {2023},
  url       = {https://ceur-ws.org/Vol-3497/paper-085.pdf}
}
```

---

## Overview

This repository contains the code of the M&S NLP team (Utrecht University) for the **EXIST 2023 shared task** at CLEF 2023, which targets the identification and categorization of sexism in English and Spanish tweets:

- **Task 1 — Sexism identification**: binary classification (sexist vs. non-sexist)
- **Task 2 — Source intention**: direct, reported, or judgemental
- **Task 3 — Sexism categorization**: multi-label classification over sexism categories

## Methods

All methods below are implemented in the notebook (TensorFlow/Keras + Hugging Face Transformers):

- **Preprocessing**: regex-based text cleaning, NLTK tokenization, and WordNet lemmatization
- **Class imbalance handling**: `nlpaug` data augmentation (synonym replacement, random insertion/swap), SMOTE / random oversampling (`imbalanced-learn`), and class weights
- **Models**: frozen `bert-base-multilingual-uncased`, `xlm-roberta-base`, and `distilbert-base-multilingual-cased` encoders with Keras dense classification heads (sigmoid for Task 1, softmax for Tasks 2 and 3; `MultiLabelBinarizer` for the multi-label setting)
- **Ensemble voting**: combining predictions from the three models
- **Tuning and validation**: Keras Tuner random search, stratified k-fold cross-validation, early stopping
- **Baseline**: TF-IDF + random forest for comparison
- **Submission output**: JSON predictions with hard and soft labels in the EXIST evaluation format

## Quick Start

```bash
git clone https://github.com/mohammadi-hadi/Exist-2023.git
cd Exist-2023
pip install tensorflow transformers keras-tuner scikit-learn imbalanced-learn nlpaug nltk pandas numpy
jupyter notebook "M&S_NLP technical report for Exist 2023 (new version).ipynb"
```

Update the data paths at the top of the notebook to point to your local copies of the EXIST 2023 files. A GPU is recommended for training.

## Repository Structure

```
Exist-2023/
├── M&S_NLP technical report for Exist 2023 (new version).ipynb  # All code (preprocessing, training, ensembling, submission)
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Data

The **EXIST 2023 dataset** (English and Spanish tweets with hard and soft labels for all three tasks) is distributed by the task organizers and is **not included in this repository**; see the [EXIST 2023 website](http://nlp.uned.es/exist2023/) for access conditions. The notebook expects local copies of the training and test JSON files (e.g. `EXIST2023_training.json`, `EXIST2023_test_clean.json`) and optionally merges an additional sexism dataset (`train_all_tasks.csv`) into the training data.

## Related Work

- [Sexism-Detection-in-Social-Media](https://github.com/mohammadi-hadi/Sexism-Detection-in-Social-Media) — mirror of this repository (identical notebook)
- [Explainable-Sexism-Detection](https://github.com/mohammadi-hadi/Explainable-Sexism-Detection) — transparent sexism-detection pipeline with SHAP explanations (Applied Sciences, 2024)
- [BehAv-PO](https://github.com/mohammadi-hadi/BehAv-PO) — behavioral cluster-driven multi-agent preference optimization for sexism detection (EXIST 2024)

## License

MIT License — see [LICENSE](LICENSE).

## Contact

- **Hadi Mohammadi** — Utrecht University
- Website: [mohammadi.cv](https://mohammadi.cv)

## Acknowledgments

Thanks to the EXIST 2023 organizers for providing the dataset and evaluation framework, and to CLEF 2023 for hosting the shared task.
