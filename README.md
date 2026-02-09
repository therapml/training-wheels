# Training Modules - TherapML

A training module for implementing core machine learning and tensor operations from scratch using PyTorch and NumPy.

## Overview

This project contains phase 01 training focused on:
- **Tensor operations**: Batch matrix multiplication and dot products
- **Optimizers**: Stochastic Gradient Descent (SGD) implementation

## Project Structure

```
therapml/                    # Main implementation folder - write your code here
├── __init__.py
tests/                       # Test suite
├── adapters.py             # Glue logic only - connects therapml to tests
├── test_*.py               # Test files
├── common.py               # Shared test utilities
└── fixtures/               # Test fixtures
```

## Implementation Guide

### Key Rules
- **Write all implementation code in the `therapml/` folder**
- **Only write glue/adapter code in `tests/adapters.py`** - this file should only contain imports and function wrappers that connect your implementations to the tests

### Phase 01: Core Operations

#### 1. Tensor Operations

Implement batch matrix multiplication and tensor dot products.

**Implementation location**: `therapml/` (create new modules as needed)

**What to implement**:
- `tensor_multiply(arr1, arr2)`: Batched matrix multiplication
  - Input: Two 3D arrays with shapes `(batch, x, y)` and `(batch, y, z)`
  - Output: Result with shape `(batch, x, z)`
  - Use einsum pattern: `"bij,bjk->bik"`
  
- `tensor_dot(arr1, arr2, dim)`: Tensor dot product along specified dimension
  - Implement according to test requirements

**Testing**: Run tests with
```bash
pytest tests/test_basic_tensor.py -v
```

#### 2. Optimizer - SGD

Implement Stochastic Gradient Descent optimizer compatible with PyTorch.

**Implementation location**: `therapml/` (create new modules as needed)

**What to implement**:
- `SGD` class that behaves like `torch.optim.SGD`
  - Constructor: `__init__(params, lr, weight_decay)`
  - Methods: `zero_grad()`, `step()`
  - Should match PyTorch's SGD behavior when optimizing a simple linear model

**Testing**: Run tests with
```bash
pytest tests/test_optimizer.py -v
```

## Getting Started

### 1. Set up environment
```bash
# Install dependencies
uv sync
```

### 2. Start implementing
- Create modules inside `therapml/` folder for your implementations
- Update `tests/adapters.py` with import statements pointing to your code
- Run tests to validate your implementation

### 3. Run tests
```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_basic_tensor.py -v
pytest tests/test_optimizer.py -v
```