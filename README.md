# Character Recognition Neural Network

A neural network built from scratch using only NumPy to classify handwritten characters into 35 classes: uppercase letters A-Z and digits 1-9.

## Overview

The project implements, by hand:
- Forward propagation
- ReLU activation
- Softmax output layer
- Cross-entropy loss
- Backpropagation
- Mini-batch gradient descent
- A training loop with validation and early stopping

Trained and evaluated on real handwritten samples from the EMNIST ByClass dataset.

## Results

- Test accuracy: 72.38%
- Validation accuracy (best epoch): 69.33%
- Training accuracy (best epoch): 95.84%
- Random-guess baseline (35 classes): 2.9%


## Project structure

```
├── data/
│   ├── emnist-byclass-train.csv
│   ├── emnist-byclass-test.csv
│   └── emnist-byclass-mapping.txt
├── prepare_data.py      # Filters EMNIST to 35 classes, normalizes, splits, saves processed_data.npz
├── network.py            # Weight init, forward prop, ReLU, softmax, cross-entropy loss, backprop
├── train.py               # Training loop with mini-batch gradient descent and early stopping
├── evaluate.py           # Test set evaluation, confusion matrix, error analysis
```

## How to run

```bash
pip install numpy pandas matplotlib

python prepare_data.py
python train.py
python evaluate.py
```

## Architecture

Input (784) -> Hidden layer (128, ReLU) -> Output layer (35, Softmax)

- 784 = 28x28 flattened pixel values (fixed by image size)
- 128 = hidden layer size (chosen)
- 35 = one output per class (fixed by the task)

Weights are initialized with small random values scaled by layer input size; biases start at zero.

## Dataset

EMNIST ByClass, filtered to 35 classes (digits 1-9, uppercase A-Z). 100 samples per class, normalized to 0-1, split 70/15/15 into train/validation/test.

## Notes

- Early stopping (patience = 5 epochs) keeps the weights from the epoch with the lowest validation loss, instead of the final epoch, to reduce overfitting.
- EMNIST images are stored in transposed order; a transpose is applied before displaying them so they appear upright.
