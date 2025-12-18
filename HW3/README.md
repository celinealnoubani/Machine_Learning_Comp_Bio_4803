# HW3: Neural Networks for Protein Function Prediction

## Course: CX/CS 4803/7643 - Machine Learning for Computational Biology (Georgia Tech)

## Overview
This assignment explores deep learning architectures for enzyme function prediction from protein sequences. We implement various neural network models including Transformers, CNNs, and leverage pretrained protein language models (ESM-2) to classify proteins into Enzyme Commission (EC) numbers.

## Problem Description
- **Task**: Multi-class classification of enzyme functions (200 EC number classes)
- **Input**: Protein sequences (amino acid strings)
- **Training samples**: 20,000 sequences
- **Challenge**: Variable-length sequences, complex sequence-function relationships

## Topics Covered
- **Sequence Tokenization** - One-hot encoding of amino acids
- **Transformer Architecture** - Self-attention mechanisms for sequence modeling
- **Convolutional Neural Networks** - 1D convolutions for local pattern recognition
- **Transfer Learning** - Pretrained protein language model embeddings (ESM-2)

---

## Implementation Details

### Task 1: One-Hot Tokenizer (4 points)
```python
amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids

def one_hot_encode(sequence):
    """Convert amino acid sequence to L×20 one-hot tensor"""
    one_hot = torch.zeros(len(sequence), 20)
    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            one_hot[i, aa_to_idx[aa]] = 1
    return one_hot  # Unknown amino acids 'X' encoded as all-zero
```

### Task 2: Transformer with Multi-Head Attention (9 points)

**AttentionClassifier Architecture:**
```
Input (L×20) → Embedding (L×64) → TransformerBlock → AvgPool → FC → Output (200)
```

| Component | Implementation |
|-----------|----------------|
| Self-Attention | `nn.MultiheadAttention(embed_dim, num_heads)` |
| Feed-Forward | Linear → ReLU → Linear |
| Normalization | `nn.LayerNorm` (pre-norm architecture) |
| Masking | Key padding mask for variable-length sequences |

**Best Validation Accuracy**: 70.90%

### Task 3: Transformer with TransformerEncoder (9 points)

**TransformerClassifier Architecture:**
```
Input → Embedding → TransformerEncoder → AvgPool → FC → Output
```

Uses PyTorch's wrapped modules:
- `nn.TransformerEncoderLayer`
- `nn.TransformerEncoder`

**Best Validation Accuracy**: 72.65%

### Task 4: 1D-CNN Model (9 points)

**CNNClassifier Architecture:**
```
Input → Embedding → Conv1D×3 → AvgPool → FC → Output
```

| Layer | Configuration |
|-------|---------------|
| Conv1D | kernel_size=3, padding=1 |
| Activation | ReLU |
| Normalization | BatchNorm1d |
| Pooling | AdaptiveAvgPool1d(1) |

**Best Validation Accuracy**: 94.93%

### Task 5: ESM-2 Pretrained Embeddings (9 points)

**Approach:**
1. Extract embeddings using ESM-2 model (`esm2_t33_650M_UR50D`)
2. Average over sequence length for sequence-level representation
3. Train MLP classifier on 1280-dimensional embeddings

**MLPClassifier Architecture:**
```
Input (1280) → Hidden (640) → Hidden (320) → Output (200)
```

**Best Validation Accuracy**: 97.60%

---

## Training Configuration

| Model | Epochs | Batch Size | Learning Rate | Best Val Acc |
|-------|--------|------------|---------------|--------------|
| AttentionClassifier | 28 | 14 | 5e-4 | 70.90% |
| TransformerClassifier | 40 | 32 | 7e-4 | 72.65% |
| CNNClassifier | 20 | 28 | 5e-4 | 94.93% |
| ESM + MLP | 20 | 32 | 1e-4 | 97.60% |

## Key Findings
- **1D-CNN outperforms Transformers** on this task, likely due to the importance of local sequence motifs for enzyme function
- **ESM-2 embeddings achieve best results**, demonstrating the power of transfer learning from protein language models
- **Proper padding masks** are crucial for handling variable-length sequences in attention mechanisms

## Files
```
HW3/
├── hw3_nn (1).ipynb
├── test_preds_attention.csv
├── test_preds_transformer.csv
├── test_preds_cnn.csv
├── test_preds_esm.csv
└── weights.csv
```

## Key Concepts Demonstrated
- Sequence tokenization strategies for biological data
- Implementing attention mechanisms from components
- Handling variable-length sequences with padding and masking
- Transfer learning with pretrained biological models
- Early stopping and hyperparameter tuning

## Skills Gained
- Building Transformer models in PyTorch
- Designing CNN architectures for sequence data
- Using pretrained protein language models (ESM-2)
- Training deep learning models for biological sequence analysis
