import numpy as np
from network import initialize_parameters, forward_propagation, softmax, cross_entropy_loss, backward_propagation, update_parameters


def compute_accuracy(probs, y_true):
    """
    Computes the percentage of correct predictions.

    Parameters:
        probs (numpy array): predicted probabilities, shape (batch_size, 35)
        y_true (numpy array): true labels, shape (batch_size,)

    Returns:
        float: accuracy as a fraction between 0 and 1
    """
    predictions = np.argmax(probs, axis=1)  # index of highest probability = predicted class
    return np.mean(predictions == y_true)


import copy

def train(X_train, y_train, X_val, y_val, hidden_size=128, num_classes=35,
          learning_rate=0.1, num_epochs=50, batch_size=32, patience=5):
    """
    Trains the neural network using mini-batch gradient descent,
    tracking training and validation loss/accuracy each epoch.

    Uses early stopping: keeps a saved copy of the weights from
    whichever epoch had the lowest validation loss, and stops
    training if validation loss hasn't improved for `patience`
    consecutive epochs - this prevents the returned model from
    overfitting past its best generalization point.

    Parameters:
        X_train, y_train: training data and labels
        X_val, y_val: validation data and labels
        hidden_size (int): number of hidden neurons
        num_classes (int): number of output classes (35)
        learning_rate (float): step size for gradient descent
        num_epochs (int): maximum number of full passes through training data
        batch_size (int): number of samples per gradient update
        patience (int): stop early if val_loss doesn't improve for this many epochs

    Returns:
        best_params (dict): weights/biases from the best-validation-loss epoch
        history (dict): per-epoch train/val loss and accuracy, for plotting
    """
    input_size = X_train.shape[1]
    params = initialize_parameters(input_size, hidden_size, num_classes)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    n_samples = X_train.shape[0]

    best_val_loss = float('inf')
    best_params = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        perm = np.random.permutation(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            cache = forward_propagation(X_batch, params)
            cache["probs"] = softmax(cache["Z2"])

            grads = backward_propagation(X_batch, y_batch, params, cache)
            params = update_parameters(params, grads, learning_rate)

        train_cache = forward_propagation(X_train, params)
        train_probs = softmax(train_cache["Z2"])
        train_loss = cross_entropy_loss(train_probs, y_train)
        train_acc = compute_accuracy(train_probs, y_train)

        val_cache = forward_propagation(X_val, params)
        val_probs = softmax(val_cache["Z2"])
        val_loss = cross_entropy_loss(val_probs, y_val)
        val_acc = compute_accuracy(val_probs, y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {train_loss:.4f}, "
              f"train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = copy.deepcopy(params)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after epoch {epoch+1} "
                  f"(no improvement for {patience} epochs). Best val_loss: {best_val_loss:.4f}")
            break

    return best_params, history

if __name__ == "__main__":
    data = np.load('data/processed_data.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    params, history = train(X_train, y_train, X_val, y_val,
                             learning_rate=0.1, num_epochs=50, batch_size=32)

    np.savez('model_weights.npz', **params)
    np.savez('training_history.npz', **history)
    print("Training complete. Model and history saved.")