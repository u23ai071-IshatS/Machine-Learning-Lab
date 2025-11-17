import csv
import random
import math
from collections import Counter, defaultdict

# -------------------------------
# STEP 1: Load and preprocess data
# -------------------------------

def load_iris_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            # Convert features to float and label to string
            features = list(map(float, row[:-1]))
            label = row[-1]
            dataset.append((features, label))
    return dataset

def train_val_test_split(dataset, train_ratio=0.6, val_ratio=0.2):
    random.shuffle(dataset)
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train + n_val]
    test_data = dataset[n_train + n_val:]
    return train_data, val_data, test_data

# ----------------------------------
# STEP 2: Implement Decision Tree
# ----------------------------------

def gini_index(groups, classes):
    total_samples = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        labels = [row[-1] for row in group]
        for class_val in classes:
            p = labels.count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / total_samples)
    return gini

def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[0][index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = 999, 999, 999, None
    for index in range(len(dataset[0][0])):  # features
        for row in dataset:
            groups = test_split(index, row[0][index], dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[0][index], gini, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return Counter(outcomes).most_common(1)[0][0]

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # if no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right)
        split(node['right'], max_depth, min_size, depth + 1)

def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    if row[0][node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# ----------------------------------
# STEP 3: Implement Bagging
# ----------------------------------

class BaggingClassifier:
    def __init__(self, n_trees=5, max_depth=5, min_size=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.trees = []

    def subsample(self, dataset):
        sample = []
        while len(sample) < len(dataset):
            index = random.randrange(len(dataset))
            sample.append(dataset[index])
        return sample

    def fit(self, dataset):
        self.trees = []
        for _ in range(self.n_trees):
            sample = self.subsample(dataset)
            tree = build_tree(sample, self.max_depth, self.min_size)
            self.trees.append(tree)

    def predict(self, row):
        predictions = [predict(tree, row) for tree in self.trees]
        return Counter(predictions).most_common(1)[0][0]

# ----------------------------------
# STEP 4: Evaluate Model
# ----------------------------------

def accuracy(dataset, model):
    correct = 0
    for row in dataset:
        prediction = model.predict(row)
        if prediction == row[-1]:
            correct += 1
    return correct / len(dataset) * 100

# ----------------------------------
# MAIN EXECUTION
# ----------------------------------

if __name__ == "__main__":
    dataset = load_iris_dataset("Data/Iris.csv")
    train, val, test = train_val_test_split(dataset)

    model = BaggingClassifier(n_trees=7, max_depth=5, min_size=5)
    model.fit(train)

    print("\nBagging Ensemble Classifier (from scratch)")
    print("-----------------------------------------")
    print(f"Training Accuracy:   {accuracy(train, model):.2f}%")
    print(f"Validation Accuracy: {accuracy(val, model):.2f}%")
    print(f"Testing Accuracy:    {accuracy(test, model):.2f}%")
