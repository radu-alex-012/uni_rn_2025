import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

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
training_data = np.array(training_data) / 255.0
test_data = []
test_labels = []
for image, label in test:
    test_data.append(image.flatten())
    test_labels.append(label)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
mlp = MLPClassifier(
    hidden_layer_sizes=(40,),
    solver="sgd",
    alpha=1e-4,
    learning_rate_init=0.2,
    max_iter=1,
    random_state=1,
    warm_start=True,
)
target_accuracy = 98.0
current_accuracy = 0.0
step = 1
while current_accuracy < target_accuracy:
    mlp.fit(training_data, training_labels)
    training_preds = mlp.predict(training_data)
    current_accuracy = accuracy_score(training_labels, training_preds) * 100
    print(f"{step}. Training accuracy: {current_accuracy:.2f}%")
    step += 1
test_preds = mlp.predict(test_data)
test_accuracy = accuracy_score(test_labels, test_preds) * 100
print(f"Test accuracy: {test_accuracy:.2f}%")
predictions_csv = {
    "ID": [],
    "target": [],
}
for i, label in enumerate(test_preds):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)
df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)
