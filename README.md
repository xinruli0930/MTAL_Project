# Multi-Task Adversarial Learning (MTAL) for Treatment Effect Estimation in Basket Trials

## Overview
This repository contains the implementation and reproduction of the Multi-Task Adversarial Learning (MTAL) framework, originally introduced by Chu, Rathbun, and Li (2022). The MTAL model uniquely integrates multi-task neural networks with adversarial learning to enhance the estimation of subgroup-specific treatment effects in basket trials, addressing significant challenges such as the absence of control groups and tumor heterogeneity.

## Repository Structure

```
MTAL_Project/
├── data/
│   ├── IHDP/
│   │   ├── ihdp_npci_1-1000.train.npz
│   │   └── ihdp_npci_1-1000.test.npz
│   ├── NEWS/
│   │   ├── vocab.nips.txt
│   │   └── docword.nips.txt
│   └── Synthetic/
│       └── synthetic_basket_trial.csv
├── src/
│   ├── data_preprocessing.py
│   ├── MTAL_model.py
│   ├── training_loop.py
│   ├── evaluation.py
│   ├── generate_synthetic_data.py
│   └── generate_figures.py
├── results/
│   ├── figures/
│   └── metrics_results.csv
├── requirements.txt
└── README.md
```

## Installation

Clone this repository:
```bash
git clone https://github.com/xinruli0930/MTAL_Project.git
cd MTAL_Project
```

Install required packages:
```bash
pip install -r requirements.txt
```

## Datasets

- **IHDP Dataset:** Downloaded from [NPCI GitHub](https://github.com/vdorie/npci)
- **News Dataset:** Downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/164/bag+of+words)
- **Synthetic Dataset:** Generated following guidelines specified in this repository (included in `data/Synthetic/synthetic_basket_trial.csv`).

Place the datasets as shown in the repository structure.

## Synthetic Dataset Generation

Run the following script to generate the synthetic dataset:

```bash
python src/generate_synthetic_data.py
```

## Usage

### Data Preprocessing
```bash
python src/data_preprocessing.py
```

### Training MTAL Model
```bash
python src/training_loop.py
```

### Evaluation
```bash
python src/evaluation.py
```

### Generating Figures

Run the following script to generate figures for your results:

```bash
python src/generate_figures.py
```

Generated figures will be saved in the `results/figures/` directory.

## Results
The results include PEHE and ATE scores comparing our reproduction with original MTAL performance, demonstrating high reproducibility and robustness of the MTAL approach.

## Ablation Study
An ablation experiment was conducted to investigate the role of adversarial learning in MTAL. Results show significant performance degradation without the adversarial discriminator, confirming its critical contribution.

## Contributions
This repository was developed entirely by the author as part of a research project focused on reproducibility and extensions of existing methodologies in causal inference for clinical trials.

## Reference
Chu, T., Rathbun, S., & Li, Y. (2022). Multi-Task Adversarial Learning for Treatment Effect Estimation in Basket Trials. 
https://arxiv.org/abs/2203.05123

## Contact
For questions or further information, please contact Xinru Li at xinruli4@illinois.edu.
