<div align="center">

# EXIST 2023: Online Sexism Detection

### Multi-Model Approach with BERT, XLM-RoBERTa, and DistilBERT

[![EXIST 2023](https://img.shields.io/badge/EXIST-2023-blue.svg)](http://nlp.uned.es/exist2023/)
[![CLEF 2023](https://img.shields.io/badge/CLEF-2023-green.svg)](https://clef2023.clef-initiative.eu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8144300.svg)](https://doi.org/10.5281/zenodo.8144300)

*Robust detection of online sexism using ensemble transformer models*

[Paper](https://ceur-ws.org/Vol-3497/) • [Competition](http://nlp.uned.es/exist2023/) • [Data](https://doi.org/10.5281/zenodo.8144300)

---

</div>

## Overview

This repository contains the code for our participation in the **EXIST 2023 shared task** at CLEF 2023, focusing on identifying and categorizing online sexism. Our approach combines multiple pre-trained transformer-based models (BERT, XLM-RoBERTa, and DistilBERT) with ensemble voting to achieve robust classification.

## Key Features

- **Multi-Model Ensemble**: Combines BERT, XLM-RoBERTa, and DistilBERT predictions
- **Bilingual Support**: Handles both English and Spanish text
- **Data Augmentation**: Implements preprocessing and augmentation techniques
- **Voting System**: Ensemble approach for improved robustness

## Authors

- **Hadi Mohammadi** - Utrecht University ([h.mohammadi@uu.nl](mailto:h.mohammadi@uu.nl))
- **Anastasia Giachanou** - Utrecht University
- **Ayoub Bagheri** - Utrecht University

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mohammadi-hadi/Exist-2023.git
cd Exist-2023

# Install dependencies
pip install transformers torch pandas numpy scikit-learn

# Open the Jupyter notebook
jupyter notebook "M&S_NLP technical report for Exist 2023 (new version).ipynb"
```

## Repository Structure

```
Exist-2023/
├── M&S_NLP technical report for Exist 2023 (new version).ipynb  # Main notebook
├── LICENSE                                                        # MIT License
├── CONTRIBUTING.md                                                # Contribution guidelines
└── README.md                                                      # This file
```

## Methodology

1. **Data Preprocessing**: Text cleaning, normalization, and encoding
2. **Data Augmentation**: Techniques to handle class imbalance
3. **Model Training**: Fine-tuning BERT, XLM-RoBERTa, and DistilBERT
4. **Ensemble Voting**: Combining predictions from multiple models
5. **Evaluation**: Performance assessment on EXIST 2023 test set

## Results

Our ensemble approach achieved competitive results in the EXIST 2023 competition, with particularly strong performance on English text classification.

## Citation

```bibtex
@inproceedings{mohammadi2023exist,
  title={Towards Robust Online Sexism Detection: A Multi-Model Approach with BERT, XLM-RoBERTa, and DistilBERT for EXIST 2023 Tasks},
  author={Mohammadi, Hadi and Giachanou, Anastasia and Bagheri, Ayoub},
  booktitle={Proceedings of the Working Notes of CLEF 2023},
  year={2023},
  publisher={CEUR Workshop Proceedings}
}
```

## Related Work

- [Explainable Sexism Detection](https://github.com/mohammadi-hadi/Explainable-Sexism-Detection) - Extended work on explainable classification

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- EXIST 2023 organizers for providing the dataset and evaluation framework
- CLEF 2023 for hosting the shared task

## Contact

For questions or collaboration inquiries:
- **Email**: [h.mohammadi@uu.nl](mailto:h.mohammadi@uu.nl)
- **Website**: [mohammadi.cv](https://mohammadi.cv)
