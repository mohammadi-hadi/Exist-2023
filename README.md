# EXIST 2023: Online Sexism Detection

Code for the paper *"Towards Robust Online Sexism Detection: A Multi-Model Approach with BERT, XLM-RoBERTa, and DistilBERT for EXIST 2023 Tasks"* (CLEF 2023 Working Notes).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8144300.svg)](https://doi.org/10.5281/zenodo.8144300)

[Paper](https://ceur-ws.org/Vol-3497/paper-085.pdf) • [EXIST 2023 task](http://nlp.uned.es/exist2023/) • [Code archive on Zenodo](https://doi.org/10.5281/zenodo.8144300)

## Overview

This repository contains the code of the M&S NLP team (Utrecht University) for the **EXIST 2023 shared task** at CLEF 2023, which targets the identification and categorization of sexism in English and Spanish tweets:

- **Task 1 — Sexism identification**: binary classification (sexist vs. non-sexist)
- **Task 2 — Source intention**: direct, reported, or judgemental
- **Task 3 — Sexism categorization**: multi-label classification over sexism categories

The approach combines three multilingual transformer encoders with a voting ensemble. As reported in the paper, the system performed best on English, and combining the outputs of multiple models via voting improved robustness.

**Authors:** Hadi Mohammadi, Anastasia Giachanou, Ayoub Bagheri — Department of Methodology and Statistics, Utrecht University.

Note: the repository [Sexism-Detection-in-Social-Media](https://github.com/mohammadi-hadi/Sexism-Detection-in-Social-Media) contains an identical copy of this notebook.

## Data

The **EXIST 2023 dataset** (English and Spanish tweets with hard and soft labels for all three tasks) is distributed by the task organizers and is **not included in this repository**; see the [EXIST 2023 website](http://nlp.uned.es/exist2023/) for access conditions. The notebook expects local copies of the training and test JSON files (e.g. `EXIST2023_training.json`, `EXIST2023_test_clean.json`) and optionally merges an additional sexism dataset (`train_all_tasks.csv`) into the training data.

## Methods

All methods below are implemented in the notebook (TensorFlow/Keras + Hugging Face Transformers):

- **Preprocessing**: regex-based text cleaning, NLTK tokenization, and WordNet lemmatization
- **Class imbalance handling**: `nlpaug` data augmentation (synonym replacement, random insertion/swap), SMOTE / random oversampling (`imbalanced-learn`), and class weights
- **Models**: frozen `bert-base-multilingual-uncased`, `xlm-roberta-base`, and `distilbert-base-multilingual-cased` encoders with Keras dense classification heads (sigmoid for Task 1, softmax for Tasks 2 and 3; `MultiLabelBinarizer` for the multi-label setting)
- **Ensemble voting**: combining predictions from the three models
- **Tuning and validation**: Keras Tuner random search, stratified k-fold cross-validation, early stopping
- **Baseline**: TF-IDF + random forest for comparison
- **Submission output**: JSON predictions with hard and soft labels in the EXIST evaluation format

Detailed results on the official EXIST 2023 test set are reported in the [paper](https://ceur-ws.org/Vol-3497/paper-085.pdf).

## Repository structure

```
Exist-2023/
├── M&S_NLP technical report for Exist 2023 (new version).ipynb  # All code (preprocessing, training, ensembling, submission)
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## How to run

```bash
git clone https://github.com/mohammadi-hadi/Exist-2023.git
cd Exist-2023
pip install tensorflow transformers keras-tuner scikit-learn imbalanced-learn nlpaug nltk pandas numpy
jupyter notebook "M&S_NLP technical report for Exist 2023 (new version).ipynb"
```

Update the data paths at the top of the notebook to point to your local copies of the EXIST 2023 files. A GPU is recommended for training.

## Citation

```bibtex
@inproceedings{mohammadi2023towards,
  title     = {Towards Robust Online Sexism Detection: A Multi-Model Approach with BERT, XLM-RoBERTa, and DistilBERT for EXIST 2023 Tasks},
  author    = {Mohammadi, Hadi and Giachanou, Anastasia and Bagheri, Ayoub},
  booktitle = {Working Notes of CLEF 2023 -- Conference and Labs of the Evaluation Forum},
  series    = {CEUR Workshop Proceedings},
  volume    = {3497},
  pages     = {1000--1011},
  year      = {2023},
  url       = {https://ceur-ws.org/Vol-3497/paper-085.pdf}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

Thanks to the EXIST 2023 organizers for providing the dataset and evaluation framework, and to CLEF 2023 for hosting the shared task.

For questions, see [mohammadi.cv](https://mohammadi.cv).
