import numpy as np
from network import forward_propagation, softmax
from train import compute_accuracy

# Load processed data and the trained model
data = np.load('data/processed_data.npz')
X_test, y_test = data['X_test'], data['y_test']

model = np.load('model_weights.npz')
params = {"W1": model["W1"], "b1": model["b1"], "W2": model["W2"], "b2": model["b2"]}

# Run test set through the trained network
cache = forward_propagation(X_test, params)
probs = softmax(cache["Z2"])

test_acc = compute_accuracy(probs, y_test)
print(f"Test accuracy: {test_acc:.4f}")

predictions = np.argmax(probs, axis=1)

import matplotlib.pyplot as plt

# Build the 35x35 confusion matrix
num_classes = 35
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for true_label, pred_label in zip(y_test, predictions):
    conf_matrix[true_label, pred_label] += 1

# Class names in order (0-8 = digits 1-9, 9-34 = A-Z) - matches our earlier relabeling
class_names = [str(d) for d in range(1, 10)] + [chr(c) for c in range(65, 91)]

# Plot it
plt.figure(figsize=(14, 12))
plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar(label='Number of samples')
plt.xticks(range(num_classes), class_names, rotation=90)
plt.yticks(range(num_classes), class_names)
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion Matrix - Character Recognition')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved to confusion_matrix.png")

# Find indices where the model got it wrong
wrong_indices = np.where(predictions != y_test)[0]
print(f"Total wrong predictions: {len(wrong_indices)} out of {len(y_test)}")

# Show 5 examples: the image, true label, predicted label, and confidence
num_examples = 5
fig, axes = plt.subplots(1, num_examples, figsize=(15, 4))

for i, idx in enumerate(wrong_indices[:num_examples]):
    image = X_test[idx].reshape(28, 28).T

    true_char = class_names[y_test[idx]]
    pred_char = class_names[predictions[idx]]
    confidence = probs[idx, predictions[idx]]

    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f"True: {true_char}\nPred: {pred_char} ({confidence:.2f})")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('error_examples.png', dpi=150)
plt.show()
print("Error examples saved to error_examples.png")