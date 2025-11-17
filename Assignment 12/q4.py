import numpy as np

# -------------------------
# Load the Iris dataset manually
# -------------------------
# Iris dataset (150 samples, 4 features, 3 classes)
# We embed the data directly (to avoid sklearn)
data = np.genfromtxt('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv',
                     delimiter=',', dtype=str)

# Remove header
data = data[1:]

# Features
X = data[:, :-1].astype(float)

# Labels (convert strings → integers 0,1,2)
label_map = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}
y = np.array([label_map[label] for label in data[:, -1]])

# One-hot encoding
def one_hot(y, num_classes=3):
    o = np.zeros((len(y), num_classes))
    o[np.arange(len(y)), y] = 1
    return o

y_onehot = one_hot(y)

# Train–test split (manual)
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))

train_idx, test_idx = indices[:train_size], indices[train_size:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
y_train_oh = y_onehot[train_idx]
y_test_oh = y_onehot[test_idx]


# ------------------------
# Utility functions
# ------------------------
def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)


# ------------------------
# Base class: N-layer Neural Network
# ------------------------
class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.01):
        # Example: [4, 10, 3] → 4 inputs, 10 hidden, 3 outputs
        self.lr = lr
        self.L = len(layer_sizes) - 1  # number of weight layers
        
        self.W = []
        self.b = []
        
        for i in range(self.L):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.W.append(w)
            self.b.append(b)

    def forward(self, X):
        a = X
        zs = []
        activations = [X]

        for i in range(self.L - 1):
            z = a @ self.W[i] + self.b[i]
            a = relu(z)
            zs.append(z)
            activations.append(a)

        # Output layer (softmax)
        z = a @ self.W[-1] + self.b[-1]
        out = softmax(z)
        zs.append(z)
        activations.append(out)
        return zs, activations

    def backward(self, zs, activations, y_true):
        m = y_true.shape[0]
        
        # output layer gradient
        delta = activations[-1] - y_true  # softmax + CE derivative

        dW = []
        db = []

        # Backprop through layers
        for i in reversed(range(self.L)):
            a_prev = activations[i]
            dW_i = (a_prev.T @ delta) / m
            db_i = np.sum(delta, axis=0, keepdims=True) / m
            
            dW.insert(0, dW_i)
            db.insert(0, db_i)

            if i != 0:
                z_prev = zs[i-1]
                delta = (delta @ self.W[i].T) * relu_deriv(z_prev)

        # Gradient step
        for i in range(self.L):
            self.W[i] -= self.lr * dW[i]
            self.b[i] -= self.lr * db[i]

    def predict(self, X):
        _, activations = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def train(self, X, y, epochs=500):
        for ep in range(epochs):
            zs, activations = self.forward(X)
            self.backward(zs, activations, y)

            if ep % 100 == 0:
                acc = self.accuracy(X, np.argmax(y, axis=1))
                print(f"Epoch {ep}, Accuracy: {acc:.3f}")


# ------------------------
# 1. MODEL: 0 hidden layers (simple perceptron)
# ------------------------
print("\n=== Training Perceptron (0 hidden layers) ===")
model0 = NeuralNetwork([4, 3], lr=0.01)
model0.train(X_train, y_train_oh, epochs=500)
acc0 = model0.accuracy(X_test, y_test)
print("Test Accuracy (0 layers):", acc0)


# ------------------------
# 2. MODEL: 1 hidden layer
# ------------------------
print("\n=== Training 1-Hidden-Layer MLP ===")
model1 = NeuralNetwork([4, 8, 3], lr=0.01)
model1.train(X_train, y_train_oh, epochs=500)
acc1 = model1.accuracy(X_test, y_test)
print("Test Accuracy (1 hidden layer):", acc1)


# ------------------------
# 3. MODEL: 2 hidden layers
# ------------------------
print("\n=== Training 2-Hidden-Layer MLP ===")
model2 = NeuralNetwork([4, 8, 8, 3], lr=0.01)
model2.train(X_train, y_train_oh, epochs=500)
acc2 = model2.accuracy(X_test, y_test)
print("Test Accuracy (2 hidden layers):", acc2)
