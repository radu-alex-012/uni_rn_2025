import pickle
import numpy as np
from multiprocessing import Pool


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]


def train(training_data, training_labels, alpha, weights, beta, out_weights, out_beta):
    delta_w = np.zeros_like(weights)
    delta_b = np.zeros_like(beta)
    delta_out_w = np.zeros_like(out_weights)
    delta_out_b = np.zeros_like(out_beta)
    for sample, label in zip(training_data, training_labels):
        hidden_logits = np.dot(sample, weights) + beta
        hidden_activation = np.tanh(hidden_logits)
        output_logits = np.dot(hidden_activation, out_weights) + out_beta
        probs = softmax(output_logits)
        error_out = probs - label
        delta_out_w += np.outer(hidden_activation, error_out) * alpha
        delta_out_b += error_out * alpha
        error_hidden = (1 - hidden_activation**2) * np.dot(out_weights, error_out)
        delta_w += np.outer(sample, error_hidden) * alpha
        delta_b += error_hidden * alpha
    return delta_w, delta_b, delta_out_w, delta_out_b


def batch_training(
    training_data, training_labels, hidden_size, alpha, no_of_threads, epochs
):
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)
    weights = np.random.normal(0, 1e-4, size=(784, hidden_size))
    beta = np.random.normal(0, 1e-4, size=(hidden_size,))
    out_weights = np.random.normal(0, 1e-4, size=(hidden_size, 10))
    out_beta = np.random.normal(0, 1e-4, size=(10,))
    data_batches = np.array_split(training_data, no_of_threads)
    label_batches = np.array_split(training_labels, no_of_threads)
    for epoch in range(1, epochs + 1):
        with Pool(no_of_threads) as pool:
            results = pool.starmap(
                train,
                [
                    (
                        data_batches[i],
                        label_batches[i],
                        alpha,
                        weights,
                        beta,
                        out_weights,
                        out_beta,
                    )
                    for i in range(no_of_threads)
                ],
            )
        total_delta_w, total_delta_b, total_delta_out_w, total_delta_out_b = 0, 0, 0, 0
        for delta_w, delta_b, delta_out_w, delta_out_b in results:
            total_delta_w += delta_w
            total_delta_b += delta_b
            total_delta_out_w += delta_out_w
            total_delta_out_b += delta_out_b
        weights -= total_delta_w
        beta -= total_delta_b
        out_weights -= total_delta_out_w
        out_beta -= total_delta_out_b
        hidden_logits = np.dot(training_data, weights) + beta
        hidden_probs = np.tanh(hidden_logits)  # hidden activation
        logits = np.dot(hidden_probs, out_weights) + out_beta
        probs = softmax(logits.T).T
        loss = cross_entropy(training_labels, probs)
        predicted_classes = np.argmax(probs, axis=1)
        true_classes = np.argmax(training_labels, axis=1)
        accuracy = np.mean(predicted_classes == true_classes) * 100
        print(f"Epoch {epoch}/{epochs}, loss = {loss:.8f}, accuracy = {accuracy:.2f}%")

    return weights, beta, out_weights, out_beta


if __name__ == "__main__":
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
    for image, label in test:
        test_data.append(image.flatten())
    training_labels = np.eye(10)[training_labels]
    training_data = np.array(training_data) / 255.0
    batch_training(training_data, training_labels, 40, 1e-4, 16, 100)
    # predictions_csv = {
    #     "ID": [],
    #     "target": [],
    # }
    # for i, label in enumerate(predictions):
    #     predictions_csv["ID"].append(i)
    #     predictions_csv["target"].append(label)
    # df = pd.DataFrame(predictions_csv)
    # df.to_csv("submission.csv", index=False)
