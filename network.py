import numpy as np

def initialize_parameters(input_size, hidden_size, output_size):
    """
    Initializes weights and biases for a 2-layer neural network
    (one hidden layer, one output layer).

    Weights are initialized with small random values scaled by the
    number of inputs to each layer, which helps keep early activations
    in a reasonable range and avoids the "symmetry problem" where all
    neurons learn identically.

    Biases are initialized to zero, which is standard practice since
    random weights already ensure neurons start out different from
    each other.

    Parameters:
        input_size (int): number of input features (784 for our 28x28 images)
        hidden_size (int): number of neurons in the hidden layer (128)
        output_size (int): number of output classes (35)

    Returns:
        dict: contains W1, b1, W2, b2 - the initial parameters
    """
    np.random.seed(42)  # reproducibility, same reasoning as the data shuffle

    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))

    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# Quick test
if __name__ == "__main__":
    params = initialize_parameters(784, 128, 35)
    for name, array in params.items():
        print(name, "shape:", array.shape)

def relu(Z):
    """
    Applies the ReLU (Rectified Linear Unit) activation function.

    Replaces all negative values with 0, leaves positive values unchanged.
    This introduces non-linearity into the network, which is what allows
    it to learn complex patterns rather than being limited to straight-line
    relationships between inputs and outputs.

    Parameters:
        Z (numpy array): raw weighted sums from a layer, any shape

    Returns:
        numpy array: same shape as Z, negative values replaced with 0
    """
    return np.maximum(0, Z)


def forward_propagation(X, params):
    """
    Runs a batch of input images through the network to produce
    raw output scores for each of the 35 classes.

    Parameters:
        X (numpy array): input images, shape (batch_size, 784)
        params (dict): W1, b1, W2, b2 from initialize_parameters()

    Returns:
        dict: contains Z1, A1, Z2 (intermediate values, needed later
              for backpropagation) and the final raw output scores
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    # Input layer -> hidden layer
    Z1 = X @ W1 + b1        # weighted sums, shape (batch_size, 128)
    A1 = relu(Z1)            # activated hidden layer output, same shape

    # Hidden layer -> output layer
    Z2 = A1 @ W2 + b2       # raw output scores, shape (batch_size, 35)

    return {"Z1": Z1, "A1": A1, "Z2": Z2}


# Quick test
if __name__ == "__main__":
    params = initialize_parameters(784, 128, 35)

    # Fake a small batch of 5 images, just for testing the shapes
    X_fake = np.random.rand(5, 784)

    cache = forward_propagation(X_fake, params)
    for name, array in cache.items():
        print(name, "shape:", array.shape)

def softmax(Z):
    """
    Converts raw output scores into probabilities that sum to 1
    across all classes, for each sample in the batch.

    Subtracts the max value per row before exponentiating, for
    numerical stability - this prevents overflow errors without
    changing the final probabilities (a standard technique).

    Parameters:
        Z (numpy array): raw scores, shape (batch_size, num_classes)

    Returns:
        numpy array: same shape, each row sums to 1.0
    """
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


# Quick test
if __name__ == "__main__":
    params = initialize_parameters(784, 128, 35)
    X_fake = np.random.rand(5, 784)
    cache = forward_propagation(X_fake, params)

    probs = softmax(cache["Z2"])
    print("Probabilities shape:", probs.shape)
    print("Row sums (should all be ~1.0):", probs.sum(axis=1))

def cross_entropy_loss(probs, y_true):
    """
    Computes the average cross-entropy loss across a batch.

    For each sample, looks up the probability the network assigned
    to the correct class, and penalizes low probabilities heavily
    (via -log), especially confident wrong predictions.

    Parameters:
        probs (numpy array): predicted probabilities, shape (batch_size, num_classes)
        y_true (numpy array): true class labels (integers), shape (batch_size,)

    Returns:
        float: average loss across the batch
    """
    batch_size = probs.shape[0]

    # For each sample i, grab probs[i, y_true[i]] - the probability
    # assigned to that sample's correct class
    correct_class_probs = probs[np.arange(batch_size), y_true]

    # Avoid log(0) errors with a tiny safety value
    correct_class_probs = np.clip(correct_class_probs, 1e-12, 1.0)

    loss = -np.mean(np.log(correct_class_probs))
    return loss


# Quick test
if __name__ == "__main__":
    params = initialize_parameters(784, 128, 35)
    X_fake = np.random.rand(5, 784)
    y_fake = np.array([3, 10, 22, 0, 34])  # fake true labels for our 5 fake images

    cache = forward_propagation(X_fake, params)
    probs = softmax(cache["Z2"])

    loss = cross_entropy_loss(probs, y_fake)
    print("Loss:", loss)

def backward_propagation(X, y_true, params, cache):
    """
    Computes gradients of the loss with respect to every weight and
    bias, by propagating the error backward from the output layer
    to the input layer using the chain rule.

    Parameters:
        X (numpy array): input images, shape (batch_size, 784)
        y_true (numpy array): true labels, shape (batch_size,)
        params (dict): current W1, b1, W2, b2
        cache (dict): Z1, A1, Z2 from forward_propagation, plus probs

    Returns:
        dict: gradients dW1, db1, dW2, db2 - same shapes as the
              corresponding parameters
    """
    batch_size = X.shape[0]
    W2 = params["W2"]
    Z1, A1 = cache["Z1"], cache["A1"]
    probs = cache["probs"]

    # One-hot encode y_true: shape (batch_size, 35), 1 at the true class, 0 elsewhere
    one_hot = np.zeros((batch_size, 35))
    one_hot[np.arange(batch_size), y_true] = 1

    # Step A: output layer error (the softmax + cross-entropy shortcut)
    dZ2 = probs - one_hot

    # Step B: gradients for W2, b2
    dW2 = A1.T @ dZ2 / batch_size
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    # Step C: propagate error back to hidden layer
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)   # ReLU derivative applied elementwise

    # Step D: gradients for W1, b1
    dW1 = X.T @ dZ1 / batch_size
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


# Quick test
if __name__ == "__main__":
    params = initialize_parameters(784, 128, 35)
    X_fake = np.random.rand(5, 784)
    y_fake = np.array([3, 10, 22, 0, 34])

    cache = forward_propagation(X_fake, params)
    cache["probs"] = softmax(cache["Z2"])

    grads = backward_propagation(X_fake, y_fake, params, cache)
    for name, array in grads.items():
        print(name, "shape:", array.shape)

def update_parameters(params, grads, learning_rate):
    """
    Updates weights and biases by taking one gradient descent step:
    moves each parameter in the opposite direction of its gradient,
    scaled by the learning rate.

    Parameters:
        params (dict): current W1, b1, W2, b2
        grads (dict): dW1, db1, dW2, db2 from backward_propagation()
        learning_rate (float): step size for the update

    Returns:
        dict: updated W1, b1, W2, b2
    """
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params


# Quick test: confirm loss decreases over a few manual steps on the same fake batch
if __name__ == "__main__":
    params = initialize_parameters(784, 128, 35)
    X_fake = np.random.rand(5, 784)
    y_fake = np.array([3, 10, 22, 0, 34])

    for step in range(5):
        cache = forward_propagation(X_fake, params)
        cache["probs"] = softmax(cache["Z2"])
        loss = cross_entropy_loss(cache["probs"], y_fake)
        print(f"Step {step}, loss: {loss:.4f}")

        grads = backward_propagation(X_fake, y_fake, params, cache)
        params = update_parameters(params, grads, learning_rate=0.1)