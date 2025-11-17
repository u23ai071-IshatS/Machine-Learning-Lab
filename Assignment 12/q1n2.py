"""
AdaBoost (One-vs-All) and SAMME (multiclass) from scratch
Requirements: numpy, pandas
Usage: put Score.csv (from the Kaggle dataset) into the same folder and run.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import math
import random

RND = 42
np.random.seed(RND)
random.seed(RND)

# ---------------------------
# Utilities: load & preprocess
# ---------------------------
def load_and_preprocess(path="Data/Score.csv", drop_columns=None):
    """
    Load CSV, do minimal preprocessing:
    - Drop columns if requested
    - Convert categorical columns (object / category) to integer labels
    - Return X (ndarray), y (ndarray), feature_names, mapping info
    """
    df = pd.read_csv(path)
    if drop_columns:
        df = df.drop(columns=drop_columns, errors='ignore')

    # assume last column is label if it's named 'Credit_Score' or last column otherwise
    if 'Credit_Score' in df.columns:
        label_col = 'Credit_Score'
    else:
        label_col = df.columns[-1]

    X_df = df.drop(columns=[label_col])
    y_ser = df[label_col].copy()

    # convert labels to integers (0..K-1)
    classes = sorted(y_ser.unique())
    class_to_idx = {c:i for i,c in enumerate(classes)}
    idx_to_class = {i:c for c,i in class_to_idx.items()}
    y = y_ser.map(class_to_idx).values

    # preprocess features: simple label encoding for categorical, keep numeric as is
    feature_names = list(X_df.columns)
    X_proc = pd.DataFrame()
    feature_maps = {}
    for col in feature_names:
        if X_df[col].dtype == object or str(X_df[col].dtype).startswith('category'):
            vals = X_df[col].fillna("NA").astype(str)
            uniq = vals.unique().tolist()
            mapping = {v:i for i,v in enumerate(uniq)}
            feature_maps[col] = mapping
            X_proc[col] = vals.map(mapping).astype(float)
        else:
            # numeric
            X_proc[col] = X_df[col].astype(float).fillna(X_df[col].median())

    return X_proc.values, y.astype(int), feature_names, (class_to_idx, idx_to_class, feature_maps)

def train_val_test_split(X, y, train_frac=0.8, val_frac=0.1, test_frac=0.1, shuffle=True):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

# ---------------------------
# Decision stump (weighted)
# ---------------------------
class DecisionStump:
    """
    Decision stump that returns a class label. Supports numeric splits and categorical equality checks.
    For numeric features, we try thresholds (sample of percentiles).
    For categorical (encoded as ints), we try splits x == category.
    The stump predicts class_a for left side and class_b for right side.
    """

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.is_categorical = False
        self.left_class = None
        self.right_class = None
        self.polarity = 1  # unused but kept for compatibility

    def fit(self, X, y, sample_weight, max_thresholds=25):
        """
        Fit stump by minimizing weighted classification error.
        X: (n_samples, n_features)
        y: integer labels (0..K-1)
        sample_weight: positive numbers same shape as y
        """
        n, m = X.shape
        best_err = float('inf')
        best = None
        classes = np.unique(y)

        for j in range(m):  # try each feature
            col = X[:, j]
            # decide if categorical by checking if values have few unique ints
            uniq = np.unique(col)
            if len(uniq) <= 15 and np.all(np.mod(uniq,1)==0):
                # treat as categorical
                for cat in uniq:
                    mask_left = (col == cat)
                    if mask_left.sum() == 0 or mask_left.sum() == n:
                        continue
                    # compute weighted most common class on left and right
                    left_weights = defaultdict(float)
                    right_weights = defaultdict(float)
                    for i in range(n):
                        if mask_left[i]:
                            left_weights[int(y[i])] += sample_weight[i]
                        else:
                            right_weights[int(y[i])] += sample_weight[i]
                    left_class = max(left_weights.items(), key=lambda p:p[1])[0]
                    right_class = max(right_weights.items(), key=lambda p:p[1])[0]
                    preds = np.where(mask_left, left_class, right_class)
                    err = np.sum(sample_weight * (preds != y))
                    if err < best_err:
                        best_err = err
                        best = (j, cat, True, left_class, right_class)
            else:
                # numeric: try thresholds (use percentiles to limit)
                sorted_vals = np.unique(np.percentile(col, np.linspace(0,100, min(max_thresholds, len(col)) )))
                # produce candidate thresholds as midpoints between adjacent unique sorted values
                thresholds = []
                s = np.unique(sorted_vals)
                if len(s) <= 1:
                    continue
                for a,b in zip(s[:-1], s[1:]):
                    thresholds.append((a+b)/2.0)
                # also try a few direct percentiles
                for thr in thresholds:
                    mask_left = (col <= thr)
                    if mask_left.sum() == 0 or mask_left.sum() == n:
                        continue
                    left_weights = defaultdict(float)
                    right_weights = defaultdict(float)
                    for i in range(n):
                        if mask_left[i]:
                            left_weights[int(y[i])] += sample_weight[i]
                        else:
                            right_weights[int(y[i])] += sample_weight[i]
                    left_class = max(left_weights.items(), key=lambda p:p[1])[0]
                    right_class = max(right_weights.items(), key=lambda p:p[1])[0]
                    preds = np.where(mask_left, left_class, right_class)
                    err = np.sum(sample_weight * (preds != y))
                    if err < best_err:
                        best_err = err
                        best = (j, thr, False, left_class, right_class)

        if best is None:
            # fallback: predict the weighted majority class always
            maj = np.bincount(y, weights=sample_weight).argmax()
            self.feature_index = 0
            self.threshold = 1e9
            self.is_categorical = False
            self.left_class = maj
            self.right_class = maj
            return

        j, thr, is_cat, left_c, right_c = best
        self.feature_index = j
        self.threshold = thr
        self.is_categorical = is_cat
        self.left_class = int(left_c)
        self.right_class = int(right_c)

    def predict(self, X):
        col = X[:, self.feature_index]
        if self.is_categorical:
            mask_left = (col == self.threshold)
        else:
            mask_left = (col <= self.threshold)
        preds = np.where(mask_left, self.left_class, self.right_class)
        return preds

# ---------------------------
# Binary AdaBoost (for One-vs-All)
# ---------------------------
class BinaryAdaBoost:
    """
    Binary AdaBoost (discrete) using decision stumps.
    For labels in {+1, -1}
    """

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.learners = []
        self.alphas = []

    def fit(self, X, y_binary):
        """
        y_binary: array of +1 / -1
        """
        n = X.shape[0]
        w = np.ones(n) / n
        self.learners = []
        self.alphas = []

        for t in range(self.n_estimators):
            stump = DecisionStump()
            # convert multi-label original y to {0,1} style for stump; stump expects integer labels
            # we will pass labels 0 and 1 for stump: map -1->0, +1->1 to let stump pick majority class
            y_for_stump = ((y_binary == 1).astype(int))
            stump.fit(X, y_for_stump, w)
            pred_stump = stump.predict(X)
            # convert stump predictions back to +1/-1
            pred = np.where(pred_stump == 1, 1, -1)

            # weighted error
            miss = (pred != y_binary).astype(float)
            err = np.dot(w, miss) / w.sum()
            # avoid division by zero or err >= 0.5 (weak learner requirement)
            if err >= 0.5:
                # stop early
                # print(f"Stopping early at t={t}, err={err:.4f}")
                break
            alpha = 0.5 * math.log((1 - err) / (err + 1e-12))
            # update weights
            w = w * np.exp(-alpha * y_binary * pred)
            w = w / w.sum()
            self.learners.append(stump)
            self.alphas.append(alpha)
        # if none learners, add a trivial predictor (majority)
        if not self.learners:
            # create a stump that predicts the majority label
            maj_label = 1 if np.sum(y_binary==1) >= np.sum(y_binary==-1) else -1
            trivial = DecisionStump()
            trivial.feature_index = 0
            trivial.threshold = 1e9
            trivial.left_class = 1 if maj_label==1 else 0
            trivial.right_class = trivial.left_class
            self.learners = [trivial]
            self.alphas = [1.0]

    def predict(self, X):
        # aggregate alpha * pred
        agg = np.zeros(X.shape[0])
        for stump, alpha in zip(self.learners, self.alphas):
            # stump returns 0/1 labels; convert to -1/+1
            pred_stump = stump.predict(X)
            pred = np.where(pred_stump == 1, 1, -1)
            agg += alpha * pred
        return np.sign(agg).astype(int)  # returns -1, 0, or 1 (0 if exactly zero)
        # convert zeros to +1
    def predict_strict(self, X):
        p = self.predict(X)
        p[p==0] = 1
        return p

# ---------------------------
# Multiclass via One-vs-All AdaBoost (train K binary)
# ---------------------------
class AdaBoost_OVA:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.classifiers = {}  # class_idx -> BinaryAdaBoost

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            # create binary labels +1 for class c, -1 for others
            y_bin = np.where(y == c, 1, -1)
            clf = BinaryAdaBoost(n_estimators=self.n_estimators)
            clf.fit(X, y_bin)
            self.classifiers[c] = clf

    def predict(self, X):
        # for each class, get score = sum(alpha * pred) where pred in -1/+1
        n = X.shape[0]
        scores = np.zeros((n, len(self.classes_)))
        for i,c in enumerate(self.classes_):
            clf = self.classifiers[c]
            agg = np.zeros(n)
            for stump, alpha in zip(clf.learners, clf.alphas):
                pred_stump = stump.predict(X)
                pred = np.where(pred_stump == 1, 1, -1)
                agg += alpha * pred
            scores[:, i] = agg
        # choose class with max score
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

# ---------------------------
# SAMME multiclass AdaBoost
# ---------------------------
class SAMME:
    """
    Implementation of SAMME algorithm (multiclass AdaBoost) using decision stumps.
    Reference: Zhu et al., "Multi-class AdaBoost", and the SAMME formulation:
    alpha_t = ln((1 - err_t)/err_t) + ln(K - 1)
    weight update: w_i <- w_i * exp(alpha_t * I(y_i != h_t(x_i)))
    """
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.learners = []
        self.alphas = []

    def fit(self, X, y):
        n = X.shape[0]
        K = len(np.unique(y))
        w = np.ones(n) / n
        self.learners = []
        self.alphas = []

        for t in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, w)
            pred = stump.predict(X)
            # weighted error
            miss = (pred != y).astype(float)
            err = np.dot(w, miss) / w.sum()
            # numerical stability
            err = min(max(err, 1e-12), 1 - 1e-12)
            # If perfect classifier achieved
            if err == 0:
                alpha = 1e9
                self.learners.append(stump)
                self.alphas.append(alpha)
                break
            alpha = math.log((1 - err) / err) + math.log(K - 1)
            # Update weights
            w = w * np.exp(alpha * miss)
            w = w / w.sum()
            self.learners.append(stump)
            self.alphas.append(alpha)
            # stopping if alpha huge (perfectly separated)
            if alpha > 1e8:
                break

    def predict(self, X):
        n = X.shape[0]
        # aggregate per class: sum over t of alpha_t * 1_{h_t(x)==k}
        classes = set()
        for stump in self.learners:
            # stump returns class labels; collect
            pass
        # determine K from learned predictions on X (or store classes during fit; here we'll derive)
        # We'll get predictions for each learner
        learner_preds = [stump.predict(X) for stump in self.learners]  # list (T) of arrays (n,)
        classes_all = sorted(set([int(v) for arr in learner_preds for v in arr]))
        classes_all = np.array(classes_all)
        K = len(classes_all)
        scores = np.zeros((n, K))
        class_to_idx = {c:i for i,c in enumerate(classes_all)}
        for alpha, pred in zip(self.alphas, learner_preds):
            for k in class_to_idx:
                # add alpha where pred == k
                mask = (pred == k)
                scores[:, class_to_idx[k]] += alpha * mask
        idx = np.argmax(scores, axis=1)
        return classes_all[idx]

# ---------------------------
# Evaluation helpers
# ---------------------------
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    lab_to_idx = {l:i for i,l in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t,p in zip(y_true, y_pred):
        mat[lab_to_idx[t], lab_to_idx[p]] += 1
    return mat, labels

def precision_recall_per_class(y_true, y_pred, labels=None):
    mat, labels = confusion_matrix(y_true, y_pred, labels)
    precisions = {}
    recalls = {}
    for i,l in enumerate(labels):
        tp = mat[i,i]
        fp = mat[:,i].sum() - tp
        fn = mat[i,:].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions[l] = prec
        recalls[l] = rec
    return precisions, recalls, mat, labels

def print_metrics(y_true, y_pred, label_map=None):
    acc = accuracy(y_true, y_pred)
    prec, rec, cm, labels = precision_recall_per_class(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Per-class precision and recall:")
    for l in labels:
        label_name = label_map[l] if label_map else str(l)
        print(f"  {label_name} : precision={prec[l]:.4f}, recall={rec[l]:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=[label_map[l] if label_map else l for l in labels],
                       columns=[label_map[l] if label_map else l for l in labels]))

# ---------------------------
# Main run: load, split, train, evaluate
# ---------------------------
def main():
    # load
    print("Loading data...")
    X, y, feature_names, maps = load_and_preprocess("Data/Score.csv", drop_columns=None)
    class_to_idx, idx_to_class, feature_maps = maps
    label_map = idx_to_class  # mapping int->original label

    print("Dataset size:", X.shape, "Num classes:", len(class_to_idx))
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, 0.8, 0.1, 0.1, shuffle=True)
    print("Train / Val / Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    # Train AdaBoost One-vs-All
    print("\nTraining One-vs-All AdaBoost (binary AdaBoost per class)...")
    ova = AdaBoost_OVA(n_estimators=50)
    ova.fit(X_train, y_train)
    y_val_pred_ova = ova.predict(X_val)
    print("\nOV A on validation set:")
    print_metrics(y_val, y_val_pred_ova, label_map)

    # Train SAMME
    print("\nTraining SAMME multiclass AdaBoost...")
    samme = SAMME(n_estimators=50)
    samme.fit(X_train, y_train)
    y_val_pred_samme = samme.predict(X_val)
    print("\nSAMME on validation set:")
    print_metrics(y_val, y_val_pred_samme, label_map)

    # Compare on test set
    print("\n--- Final comparison on TEST set ---")
    y_test_pred_ova = ova.predict(X_test)
    print("\nOne-vs-All AdaBoost on test set:")
    print_metrics(y_test, y_test_pred_ova, label_map)

    y_test_pred_samme = samme.predict(X_test)
    print("\nSAMME on test set:")
    print_metrics(y_test, y_test_pred_samme, label_map)

if __name__ == "__main__":
    main()
