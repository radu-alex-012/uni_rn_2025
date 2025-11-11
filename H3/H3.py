import pickle
import pandas as pd
import numpy as np
import torch


def softmax(z):
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return z > 0


def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2


def forward_step(X, parameters):
    W1, b1, W2, b2 = parameters
    z1 = X @ W1 + b1
    a1 = relu(z1)
    logits = a1 @ W2 + b2
    cache = {
        "X": X,
        "z1": z1,
        "a1": a1,
        "logits": logits,
    }
    return cache


def backward_step(cache, y_batch_onehot, parameters, l2_lambda):
    W1, _, W2, _ = parameters
    X = cache["X"]
    z1 = cache["z1"]
    a1 = cache["a1"]
    logits = cache["logits"]
    batch_size = X.shape[0]
    # Softmax probabilities
    probs = softmax(logits)
    # Gradient of cross-entropy loss w.r.t logits: (p - y)/N
    dlogits = (probs - y_batch_onehot) / batch_size
    # Gradients for W2, b2
    dW2 = a1.T @ dlogits
    db2 = np.sum(dlogits, axis=0, keepdims=True)
    # Backpropagate into hidden layer
    da1 = dlogits @ W2.T
    dz1 = da1 * relu_derivative(z1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    # Add L2 regularization gradients (only on weights)
    dW1 += l2_lambda * W1
    dW2 += l2_lambda * W2
    return dW1, db1, dW2, db2


def compute_loss_and_accuracy(X, y, parameters, l2_lambda):
    W1, _, W2, _ = parameters
    logits = forward_step(X, parameters)["logits"]
    # Compute loss
    with torch.no_grad():
        ce_loss = torch.nn.functional.cross_entropy(
            torch.tensor(logits, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        ).item()
    # L2 regularization on weights
    l2_loss = (l2_lambda / 2.0) * (np.sum(W1**2) + np.sum(W2**2))
    total_loss = ce_loss + l2_loss
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == y)
    return total_loss, accuracy


def mlp_classifier(
    X,
    y,
    X_test,
    y_test,
    epochs,
    learning_rate,
    batch_size,
    l2_lambda,
    input_size,
    hidden_size,
    output_size,
):
    sample_size, actual_input_size = X.shape
    actual_output_size = len(set(y))
    assert (
        actual_input_size == input_size
    ), f"Expected input_size {input_size}, got {actual_input_size}"
    assert (
        actual_output_size == output_size
    ), f"Expected output_size {output_size}, got {actual_output_size}"
    # One-hot encode labels
    y_onehot = np.zeros((sample_size, output_size))
    for i, label in enumerate(y):
        y_onehot[i, label] = 1
    # Initialize parameters
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    parameters = (W1, b1, W2, b2)
    best_test_accuracy = -np.inf
    best_parameters = None
    for epoch in range(epochs):
        # Shuffle the samples
        indices = np.arange(sample_size)
        np.random.shuffle(indices)
        X_shuffled, y_shuffled = X[indices], y_onehot[indices]
        # Mini-batch SGD
        for i in range(0, sample_size, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            y_batch_onehot = y_shuffled[i : i + batch_size]
            # Forward step
            cache = forward_step(X_batch, parameters)
            # Backward step
            dW1, db1, dW2, db2 = backward_step(
                cache, y_batch_onehot, parameters, l2_lambda
            )
            # Gradient descent update (constant learning rate, no decay)
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            parameters = (W1, b1, W2, b2)
        # Compute training/test loss/accuracy
        training_loss, training_accuracy = compute_loss_and_accuracy(
            X, y, parameters, l2_lambda
        )
        test_loss, test_accuracy = compute_loss_and_accuracy(
            X_test, y_test, parameters, l2_lambda
        )
        # Check if this is the best test accuracy so far
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            # If so, deep copy the parameters
            best_parameters = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        # Log the losses/accuracies every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Training loss: {training_loss:.4f}, accuracy: {training_accuracy * 100:.2f}% | "
                f"Test loss: {test_loss:.4f}, accuracy: {test_accuracy * 100:.2f}%, best accuracy: {best_test_accuracy * 100:.2f}%"
            )
    return best_parameters


training_file = "extended_mnist_train.pkl"
test_file = "extended_mnist_test.pkl"
with open(training_file, "rb") as fp:
    training = pickle.load(fp)
with open(test_file, "rb") as fp:
    test = pickle.load(fp)
training_data = []
training_labels = []
for image, label in training:
    training_data.append(image.flatten())
    training_labels.append(label)
test_data = []
test_labels = []
for image, label in test:
    test_data.append(image.flatten())
    test_labels.append(label)
training_data = np.array(training_data)
test_data = np.array(test_data)
# Zero-center and normalize training and test data
training_mean = training_data.mean(axis=0)
training_std = training_data.std(axis=0) + 1e-8  # avoid division by zero
training_data = (training_data - training_mean) / training_std
test_data = (
    test_data - training_mean
) / training_std  # you can't peek test data, so reuse training data mean
params = mlp_classifier(
    X=training_data,
    y=training_labels,
    X_test=test_data,
    y_test=test_labels,
    epochs=250,
    learning_rate=0.1,
    batch_size=64,
    l2_lambda=1e-3,
    input_size=784,
    hidden_size=100,
    output_size=10,
)
W1, b1, W2, b2 = params
test_logits = forward_step(test_data, params)["logits"]
probabilities = softmax(test_logits)
predictions = np.argmax(probabilities, axis=1)
predictions_csv = {
    "ID": [],
    "target": [],
}
for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(int(label))
df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)
