import csv
import random
from math import exp, log
from collections import Counter

# ---------------------------------------------------
# Step 1: Load Dataset (no numpy/pandas)
# ---------------------------------------------------

def load_breast_cancer(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 0:
                continue
            # last column is class: "no-recurrence-events" or "recurrence-events"
            features = row[:-1]
            label = row[-1].strip()
            dataset.append((features, label))
    return dataset

# Convert categorical values to integers manually
def encode_dataset(dataset):
    column_values = {}
    # collect unique values per column
    for row in dataset:
        for col in range(len(row[0])):
            if col not in column_values:
                column_values[col] = set()
            column_values[col].add(row[0][col])

    # build mapping
    mappings = {col: {val: i for i, val in enumerate(sorted(list(vals)))} 
                for col, vals in column_values.items()}

    # encode
    encoded = []
    for features, label in dataset:
        new_feat = [mappings[i][features[i]] for i in range(len(features))]
        encoded.append((new_feat, label))
    return encoded

# Binary label mapping to {-1, +1}
def encode_labels(dataset):
    new_data = []
    for features, label in dataset:
        y = 1 if label == "no-recurrence-events" else -1
        new_data.append((features, y))
    return new_data


# ---------------------------------------------------
# Step 2: Train/Val/Test Split (80:10:10)
# ---------------------------------------------------

def split_data(dataset, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(dataset)
    n = len(dataset)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train = dataset[:n_train]
    val = dataset[n_train:n_train+n_val]
    test = dataset[n_train+n_val:]
    return train, val, test


# ---------------------------------------------------
# Step 3: Decision Stump (base learner)
# ---------------------------------------------------

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        # X is a single sample: [f1,f2,...]
        if X[self.feature_index] < self.threshold:
            return self.polarity
        else:
            return -self.polarity


# ---------------------------------------------------
# Step 4: AdaBoost Implementation
# ---------------------------------------------------

class AdaBoost:
    def __init__(self, n_clf=10):
        self.n_clf = n_clf
        self.classifiers = []

    def fit(self, dataset):
        X = [row[0] for row in dataset]
        y = [row[1] for row in dataset]
        n = len(X)

        # Initialize uniform weights
        w = [1.0 / n] * n

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            # Find best stump
            for feature in range(len(X[0])):
                feature_values = sorted(list(set([x[feature] for x in X])))

                # try each possible threshold
                for thr in feature_values:
                    for polarity in [1, -1]:
                        error = 0
                        for i in range(n):
                            prediction = polarity if X[i][feature] < thr else -polarity
                            if prediction != y[i]:
                                error += w[i]

                        if error < min_error:
                            min_error = error
                            clf.feature_index = feature
                            clf.threshold = thr
                            clf.polarity = polarity

            # compute alpha
            eps = 1e-10
            clf.alpha = 0.5 * log((1 - min_error + eps) / (min_error + eps))

            # update weights
            for i in range(n):
                prediction = clf.predict(X[i])
                w[i] *= exp(-clf.alpha * y[i] * prediction)

            # normalize
            total = sum(w)
            w = [wi / total for wi in w]

            self.classifiers.append(clf)

    def predict(self, X):
        total = 0
        for clf in self.classifiers:
            total += clf.alpha * clf.predict(X)
        return 1 if total >= 0 else -1


# ---------------------------------------------------
# Step 5: Accuracy Evaluation
# ---------------------------------------------------

def accuracy(dataset, model):
    correct = 0
    for x, y in dataset:
        if model.predict(x) == y:
            correct += 1
    return correct / len(dataset) * 100


# ---------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------

if __name__ == "__main__":
    dataset = load_breast_cancer("Data/breast-cancer.data")
    dataset = encode_dataset(dataset)
    dataset = encode_labels(dataset)

    train, val, test = split_data(dataset)

    model = AdaBoost(n_clf=20)
    model.fit(train)

    print("\nAdaBoost (From Scratch) on Breast Cancer Dataset")
    print("------------------------------------------------")
    print(f"Training Accuracy:   {accuracy(train, model):.2f}%")
    print(f"Validation Accuracy: {accuracy(val, model):.2f}%")
    print(f"Testing Accuracy:    {accuracy(test, model):.2f}%")
