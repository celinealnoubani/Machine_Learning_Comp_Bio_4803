# CX/CS 4803/7643 - Machine Learning for Computational Biology


## Overview

This repository contains my coursework for Machine Learning for Computational Biology, covering deep learning techniques applied to biological data including protein sequences, molecular structures, and graph-based representations of biological systems.

## Course Structure

| Assignment | Topics | Key Implementations |
|------------|--------|---------------------|
| [HW1](./HW1) | PyTorch Fundamentals | Tensor operations, matrix manipulation |
| [HW2](./HW2) | Classification & Regression | MLP, Random Forest, SVM, Gradient Descent |
| [HW3](./HW3) | Neural Networks for Proteins | Transformer, 1D-CNN, ESM-2 embeddings |
| [HW4](./HW4) | Graph Neural Networks | GCN for node/graph classification |

## Technical Skills Demonstrated

### Deep Learning Architectures
- **Multi-Layer Perceptrons (MLP)** - Fully connected networks for classification/regression
- **Transformers** - Self-attention mechanisms for sequence modeling
- **1D Convolutional Neural Networks** - Local pattern recognition in sequences
- **Graph Convolutional Networks (GCN)** - Message passing on graph-structured data

### Biological Applications
- **Protein Function Prediction** - Enzyme Commission (EC) number classification from sequences
- **Protein Language Models** - ESM-2 embeddings for transfer learning
- **Molecular Graph Analysis** - Node and graph classification tasks

### Machine Learning Techniques
- PyTorch neural network implementation
- Scikit-learn classifiers (Random Forest, SVM, Logistic Regression)
- Gradient descent optimization from scratch
- One-hot encoding for biological sequences
- Graph pooling for graph-level predictions

## Technologies Used

- **PyTorch** - Deep learning framework
- **PyTorch Geometric** - Graph neural networks
- **Scikit-learn** - Classical ML models
- **ESM-2** - Pretrained protein language model
- **NumPy/Pandas** - Data manipulation

## Repository Structure

```
Machine_Learning_Comp_Bio_4803/
├── HW1/                          # PyTorch Introduction
│   ├── hw1_torch_intro.ipynb
│   └── README.md
├── HW2/                          # Classification & Regression
│   ├── Classification/
│   │   ├── hw2_classification.ipynb
│   │   └── *.csv                 # Prediction files
│   ├── Regression/
│   │   ├── hw2_regression.ipynb
│   │   └── *.csv                 # Prediction files
│   └── README.md
├── HW3/                          # Neural Networks for Proteins
│   ├── hw3_nn.ipynb
│   ├── test_preds_*.csv          # Model predictions
│   └── README.md
└── HW4/                          # Graph Neural Networks
    ├── hw4_gnn.ipynb
    ├── predictions_*.txt         # Model predictions
    └── README.md
```

## Highlights

### Protein Function Prediction (HW3)
- Implemented 4 different neural network architectures
- Achieved **97.6% accuracy** using ESM-2 pretrained embeddings
- Compared Transformer vs CNN performance on sequence data

### Graph Neural Networks (HW4)
- Demonstrated **2x improvement** in node classification using GCN over MLP (46% vs 21%)
- Applied graph classification to CIFAR10 super-pixel representations

## Key Concepts

- Sequence tokenization strategies (one-hot encoding)
- Attention mechanisms and positional encoding
- Transfer learning with pretrained biological models
- Graph representation learning and message passing
- Handling variable-length biological sequences


