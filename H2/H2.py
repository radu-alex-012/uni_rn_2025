import pickle
import pandas as pd
import numpy as np
from math import pi, cos


def softmax(z):
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def compute_gradients(X_batch, y_batch, weights, biases):
    logits = X_batch @ weights + biases
    probabilities = softmax(logits)
    delta_logits = (
        probabilities - y_batch
    )  # we compute the gradient this way because we use softmax and cross-entropy loss
    delta_weights = X_batch.T @ delta_logits / X_batch.shape[0]
    delta_biases = np.mean(delta_logits, axis=0, keepdims=True)
    return delta_weights, delta_biases


def slp_classifier(
    X,
    y,
    epochs,
    initial_learning_rate,
    batch_size,
    momentum,
    decay_rate,
    warmup_epochs,
    alpha,
    X_test=None,
    y_test=None,
):
    no_of_samples, input_size = X.shape
    no_of_classes = len(set(y))
    # One-hot encode labels
    y_onehot = np.zeros((no_of_samples, no_of_classes))
    for i, label in enumerate(y):
        y_onehot[i, label] = 1
    # Xavier initialization
    weights = np.random.randn(input_size, no_of_classes) * np.sqrt(1 / input_size)
    biases = np.zeros((1, no_of_classes))
    # Momentum buffers initialization - momentum helps avoid local minima and speed up convergence in consistent directions
    weights_velocities = np.zeros_like(weights)
    biases_velocities = np.zeros_like(biases)
    learning_rate = initial_learning_rate
    for epoch in range(epochs):
        # Learning rate scheduling
        # Warmup - prevents unstable large updates at start
        if epoch < warmup_epochs:
            learning_rate = initial_learning_rate * (epoch + 1) / warmup_epochs
        # Learning rate decay - fine-tune learning rate as the algorithm converges
        else:
            learning_rate = initial_learning_rate * (
                decay_rate ** (epoch - warmup_epochs + 1)
            )
        # Shuffle the samples
        indices = np.arange(no_of_samples)
        np.random.shuffle(indices)
        X_shuffled, y_shuffled = X[indices], y_onehot[indices]
        # Prepare the batches
        batches = [
            (
                X_shuffled[i : i + batch_size],
                y_shuffled[i : i + batch_size],
                weights,
                biases,
            )
            for i in range(0, no_of_samples, batch_size)
        ]
        # Compute gradients in parallel
        gradients = [compute_gradients(*batch) for batch in batches]
        # Aggregate gradients
        delta_weights = (
            np.mean([gradient[0] for gradient in gradients], axis=0) + alpha * weights
        )  # L2 regularization or weight decay - prevents overfitting
        delta_biases = np.mean([gradient[1] for gradient in gradients], axis=0)
        # Apply stochastic gradient descent with momentum
        weights_velocities = (
            momentum * weights_velocities - learning_rate * delta_weights
        )
        biases_velocities = momentum * biases_velocities - learning_rate * delta_biases
        weights += weights_velocities
        biases += biases_velocities
        # Log loss, training and test accuracies and learning rate every 10 epochs
        if (epoch + 1) % 10 == 0 and X_test is not None and y_test is not None:
            training_probabilities = softmax(X @ weights + biases)
            loss = -np.mean(
                np.sum(y_onehot * np.log(training_probabilities + 1e-8), axis=1)
            )
            loss += (alpha / 2) * np.sum(weights**2)
            training_accuracy = np.mean(np.argmax(training_probabilities, axis=1) == y)
            test_probabilities = softmax(X_test @ weights + biases)
            test_accuracy = np.mean(np.argmax(test_probabilities, axis=1) == y_test)
            print(
                f"Epoch {epoch+1}/{epochs}: loss = {loss:.4f}, training accuracy = {training_accuracy*100:.2f}%, test accuracy = {test_accuracy*100:.2f}%, learning rate = {learning_rate:.5f}"
            )
    return weights, biases


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
training_std = training_data.std(axis=0) + 1e-8  # to avoid division by zero
training_data = (training_data - training_mean) / training_std
test_data = (
    test_data - training_mean
) / training_std  # you can't peek test data, so reuse training data mean
weights, biases = slp_classifier(
    X=training_data,
    y=training_labels,
    epochs=100,
    initial_learning_rate=0.45,
    batch_size=64,
    momentum=0.92,
    warmup_epochs=8,
    decay_rate=0.995,
    alpha=1e-4,
    X_test=test_data,
    y_test=test_labels,
)
probabilities = softmax(test_data @ weights + biases)
predictions = np.argmax(probabilities, axis=1)
predictions_csv = {
    "ID": [],
    "target": [],
}
for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)
df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)
