# HW2: Classification and Regression

## Course: CX/CS 4803/7643 - Machine Learning for Computational Biology (Georgia Tech)

## Overview
This assignment covers supervised learning fundamentals through two practice problems: multi-class classification on protein expression data and regression on diabetes progression data. Implementations include both from-scratch models and scikit-learn classifiers.

## Topics Covered
- **Data Preprocessing** - Normalization, train/test splitting
- **Neural Networks** - MLP implementation in PyTorch
- **Classical ML** - Random Forest, SVM, Logistic Regression
- **Linear Regression** - Gradient descent optimization
- **Model Evaluation** - Accuracy, MSE, Pearson correlation

---

## Part 1: Classification (20 points)

### Dataset
- **Protein Expression Dataset**: 77 features, 8 classes (0-7)
- Training samples: 880
- Task: Multi-class classification

### Implementations

#### Task 1: Min-Max Normalization (2 points)
```python
def normalize(X):
    """Min-max normalization: x' = (x - min) / (max - min)"""
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)
```

#### Task 2: MLP Classifier in PyTorch (6 points)
| Component | Description |
|-----------|-------------|
| `MLP.forward()` | Forward pass: fc1 → ReLU → fc2 |
| Backpropagation | optimizer.zero_grad() → loss.backward() → optimizer.step() |
| `predict()` | Inference with outputs.argmax(dim=1) |

**Architecture**: Input(77) → Hidden(500×77) → Output(8)

#### Task 3: Scikit-learn Classifiers (9 points)

| Model | Validation Accuracy |
|-------|---------------------|
| Random Forest | 93.75% |
| SVM (Linear) | 100% |
| Logistic Regression | 98.86% |

---

## Part 2: Regression (15 points)

### Dataset
- **Diabetes Dataset**: 10 features
- Training samples: 362
- Task: Predict disease progression

### Implementations

#### Task 1: Feature Exploration (1 point)
Identified top 3 correlated features: `bmi`, `s5`, `s4`

#### Task 2: Train/Test Split (2 points)
```python
def random_split(X, y, seed=0, train_ratio=0.8):
    """Random 80/20 split using np.random.permutation"""
    indices = np.random.permutation(len(X))
    # Split at train_ratio
```

#### Task 3: Linear Regression from Scratch (8 points)
| Function | Description |
|----------|-------------|
| `linear_model(X, θ)` | Compute Y = Xθ |
| `mse(y_true, y_pred)` | MSE loss: (1/2n) \|\|y_true - y_pred\|\|² |
| `mse_grad(θ, X, y)` | Gradient: (1/n) Xᵀ(Xθ - y) |
| `gradient_descent()` | Iterative optimization with learning rate |

**Hyperparameters**: lr=0.05, epochs=15000, convergence ε=1e-3

#### Task 4: Scikit-learn Linear Regression (4 points)
```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
```

## Evaluation Metrics
- **Classification**: Test accuracy
- **Regression**: Pearson correlation (achieved ~0.72)

## Files
```
HW2/
├── Classification/
│   ├── hw2_classification (1).ipynb
│   ├── cls_pred.csv          # MLP predictions
│   ├── cls_pred_rf.csv       # Random Forest predictions
│   ├── cls_pred_svm.csv      # SVM predictions
│   └── cls_pred_lr.csv       # Logistic Regression predictions
└── Regression/
    ├── hw2_regression (1).ipynb
    ├── reg_pred.csv          # From-scratch predictions
    └── reg_pred_sklearn_LR.csv  # Sklearn predictions
```

## Key Concepts Demonstrated
- Feature normalization for improved model performance
- PyTorch training loop (forward, loss, backward, optimize)
- Gradient descent implementation from mathematical formulation
- K-fold cross-validation for model evaluation
- Hyperparameter tuning (learning rate, epochs, batch size)

## Skills Gained
- Building and training neural networks in PyTorch
- Implementing optimization algorithms from scratch
- Using scikit-learn for classical ML models
- Understanding bias-variance tradeoff through model comparison
