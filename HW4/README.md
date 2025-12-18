# HW4: Graph Neural Networks

## Course: CX/CS 4803/7643 - Machine Learning for Computational Biology (Georgia Tech)

## Overview
This assignment introduces Graph Neural Networks (GNNs) for graph-structured data analysis. We implement node classification on community detection tasks and graph classification on image super-pixel representations using PyTorch Geometric.

## Topics Covered
- **Graph Representation** - Node features, edge indices, graph batching
- **Graph Convolutions** - Message passing with GCNConv
- **Node Classification** - Predicting labels for individual nodes
- **Graph Classification** - Predicting labels for entire graphs
- **Graph Pooling** - Aggregating node embeddings to graph-level representations

---

## Datasets

### CLUSTER Dataset (Node Classification)
- **Source**: Stochastic Block Model (SBM) benchmark
- **Task**: Semi-supervised community detection
- **Node features**: 7-dimensional
- **Classes**: 6 communities
- **Training graphs**: 10,000
- **Challenge**: Identifying community structure from noisy connections

### CIFAR10 Super-Pixel Dataset (Graph Classification)
- **Source**: CIFAR10 images converted to graphs via SLIC super-pixel segmentation
- **Task**: Image classification (10 classes)
- **Node features**: 3-dimensional (RGB)
- **Training graphs**: 45,000

---

## Implementation Details

### Task 1: MLP for Node Classification (5 points)

**MLPNodeClassifier Architecture:**
```
Input (7) → Linear (32) → ReLU → Linear (6) → Output
```

```python
class MLPNodeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

**Validation Accuracy**: ~21% (baseline without graph structure)

### Task 2: GCN for Node Classification (10 points)

**GNNNodeClassifier Architecture:**
```
Input (7) → GCNConv (128) → ReLU → GCNConv (6) → Output
```

| Component | Description |
|-----------|-------------|
| `GCNConv` | Graph Convolutional layer from PyTorch Geometric |
| Message Passing | Aggregates neighbor features |
| Activation | ReLU between layers |

```python
class GNNNodeClassifier(nn.Module):
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x
```

**Validation Accuracy**: ~46% (significantly improved with graph structure)

### Task 3: GCN for Graph Classification (10 points)

**GCNGraphClassifier Architecture:**
```
Input → Embedding (512) → GCNConv×4 (256) → GlobalMeanPool → MLP → Output (10)
```

| Component | Configuration |
|-----------|---------------|
| Embedding | Linear + BatchNorm + ReLU + Dropout(0.3) |
| GCN Layers | 4 layers with BatchNorm and Dropout(0.2) |
| Pooling | `global_mean_pool` for graph-level embedding |
| Classifier | MLP with BatchNorm, ReLU, and Dropout layers |

```python
def forward(self, x, edge_index, batch):
    x = self.embedding(x)
    for i, conv in enumerate(self.convs):
        x = conv(x, edge_index)
        x = self.batch_norms[i](x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
    x = global_mean_pool(x, batch)  # Graph-level pooling
    return self.mlp(x)
```

**Validation Accuracy**: ~42%

---

## Training Configuration

| Model | Dataset | Epochs | Batch Size | LR | Val Acc |
|-------|---------|--------|------------|-----|---------|
| MLP | CLUSTER | 5 | 32 | 1e-2 | 21% |
| GCN Node | CLUSTER | 5 | 32 | 1e-2 | 46% |
| GCN Graph | CIFAR10 | 12 | 60 | 5e-3 | 42% |

## Key Findings
- **GCN significantly outperforms MLP** on node classification (46% vs 21%), demonstrating the importance of graph structure
- **Message passing** enables nodes to leverage neighbor information for better predictions
- **Graph pooling** is essential for aggregating node features into graph-level representations
- **Learning rate scheduling** (ReduceLROnPlateau) helps convergence

## Files
```
HW4/
├── hw4_gnn (1).ipynb
├── predictions_mlp_cluster.txt
├── predictions_gcn_cluster.txt
└── predictions_gcn_cifar10.txt
```

## Key Concepts Demonstrated
- Graph data representation (node features, edge index, batch tensor)
- Graph convolutional operations and message passing
- Difference between node-level and graph-level tasks
- Global pooling strategies for graph classification
- Handling batched graphs in PyTorch Geometric

## Skills Gained
- Using PyTorch Geometric for graph deep learning
- Implementing GNN architectures (GCN)
- Training models on graph-structured data
- Understanding community detection and graph classification tasks
