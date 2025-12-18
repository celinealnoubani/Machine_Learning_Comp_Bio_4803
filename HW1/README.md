# HW1: Introduction to PyTorch

## Course: CX/CS 4803/7643 - Machine Learning for Computational Biology (Georgia Tech)

## Overview
This assignment serves as an introduction to PyTorch fundamentals, covering tensor operations, matrix manipulations, and basic deep learning building blocks essential for computational biology applications.

## Topics Covered
- **Tensor Creation** - Creating tensors from lists, NumPy arrays, and special initializations
- **Tensor Operations** - Indexing, slicing, arithmetic operations
- **Matrix Operations** - Inverse, pseudoinverse, transpose, concatenation
- **PyTorch Functions** - Built-in utilities for tensor manipulation

## Implementation Details

### Task 1: Diagonal Matrix (2 points)
```python
def diag_matrix(n, x):
    """Create n×n diagonal matrix with value x on diagonal"""
    matrix = torch.zeros(n, n)
    matrix.fill_diagonal_(x)
    return matrix
```

### Task 2: Mean Square Error Loss (2 points)
```python
def mean_square_loss(y_true, y_pred):
    """Compute MSE: L = (1/n) * ||y_true - y_pred||²"""
    squared_diff = (y_true - y_pred) ** 2
    return torch.mean(squared_diff)
```

### Task 3: Extract Even Rows (2 points)
```python
def extract_even_rows(matrix):
    """Extract rows at even indices (0, 2, 4, ...)"""
    return matrix[::2]
```

### Task 4: Filter Column Values (2 points)
```python
def filter_column(tensor, column_idx, threshold):
    """Return elements from specified column greater than threshold"""
    column = tensor[:, column_idx]
    return column[column > threshold]
```

### Task 5: Remove Dimensions (1 point)
```python
def remove_dim(tensor):
    """Remove all dimensions of size 1 using torch.squeeze"""
    return torch.squeeze(tensor)
```

### Task 6: Clamp Tensor Values (1 point)
```python
def clamp_tensor(tensor, min_val, max_val):
    """Clamp tensor values to [min_val, max_val] range"""
    return torch.clamp(tensor, min=min_val, max=max_val)
```

## Key Concepts Demonstrated
- Tensor creation and initialization methods
- Element-wise and matrix operations
- Broadcasting in PyTorch
- In-place operations (fill_diagonal_)
- Boolean indexing and masking

## Files
- `hw1_torch_intro.ipynb` - Jupyter notebook with exercises and implementations

## Skills Gained
- PyTorch tensor manipulation
- Understanding PyTorch documentation
- Foundation for building neural networks
