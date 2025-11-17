import numpy as np
import pandas as pd
from collections import Counter
from math import sqrt

# -------------------------------------------------------
# Load Iris dataset (from CSV)
# -------------------------------------------------------
df = pd.read_csv("Data/Iris.csv")  # adjust filename as needed
X = df.iloc[:, 1:5].values    # features
y = df.iloc[:, 5].values      # labels


# -------------------------------------------------------
# Utility: Euclidean distance
# -------------------------------------------------------
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# -------------------------------------------------------
# DBSCAN Implementation (No sklearn)
# -------------------------------------------------------
def region_query(X, point_idx, eps):
    neighbors = []
    for i in range(len(X)):
        if euclidean(X[point_idx], X[i]) <= eps:
            neighbors.append(i)
    return neighbors


def expand_cluster(X, labels, point_idx, cluster_id, eps, min_pts):
    neighbors = region_query(X, point_idx, eps)
    
    if len(neighbors) < min_pts:
        labels[point_idx] = -1    # noise
        return False

    labels[point_idx] = cluster_id

    i = 0
    while i < len(neighbors):
        n = neighbors[i]
        if labels[n] == -1:       # noise becomes border point
            labels[n] = cluster_id
        elif labels[n] == 0:
            labels[n] = cluster_id
            new_neighbors = region_query(X, n, eps)
            if len(new_neighbors) >= min_pts:
                neighbors += new_neighbors
        i += 1

    return True


def dbscan(X, eps=0.5, min_pts=5):
    labels = [0] * len(X)   # 0 = unvisited
    cluster_id = 0

    for i in range(len(X)):
        if labels[i] != 0:
            continue

        if expand_cluster(X, labels, i, cluster_id + 1, eps, min_pts):
            cluster_id += 1

    return np.array(labels)


# -------------------------------------------------------
# Silhouette score (manual implementation)
# -------------------------------------------------------
def silhouette_score(X, labels):
    X = np.array(X)
    labels = np.array(labels)
    unique_clusters = [c for c in np.unique(labels) if c != -1]

    def avg_distance(point, cluster_points):
        return np.mean([euclidean(point, p) for p in cluster_points]) if len(cluster_points) > 0 else 0

    silhouettes = []

    for i in range(len(X)):
        if labels[i] == -1:  # ignore noise
            continue

        current_cluster = labels[i]
        same_cluster = X[labels == current_cluster]
        other_clusters = [X[labels == c] for c in unique_clusters if c != current_cluster]

        # a(i): intra-cluster distance
        a = avg_distance(X[i], same_cluster[same_cluster != X[i]].reshape(-1, X.shape[1]))

        # b(i): nearest cluster distance
        b = min(avg_distance(X[i], cluster) for cluster in other_clusters)

        silhouettes.append((b - a) / max(a, b))

    return np.mean(silhouettes)


# -------------------------------------------------------
# Purity (cluster quality measure)
# -------------------------------------------------------
def purity_score(y_true, y_pred):
    clusters = np.unique(y_pred)
    total = len(y_true)
    purity_sum = 0

    for c in clusters:
        if c == -1:
            continue
        idx = np.where(y_pred == c)[0]
        assigned_labels = y_true[idx]
        most_common = Counter(assigned_labels).most_common(1)[0][1]
        purity_sum += most_common

    return purity_sum / total


# -------------------------------------------------------
# Adjusted Rand Index (manual)
# -------------------------------------------------------
def adjusted_rand_index(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    def combinations(n):
        return n * (n - 1) / 2

    # Contingency table
    clusters_true = np.unique(labels_true)
    clusters_pred = np.unique(labels_pred)

    contingency = np.zeros((len(clusters_true), len(clusters_pred)))

    for i, t in enumerate(clusters_true):
        for j, p in enumerate(clusters_pred):
            contingency[i, j] = np.sum((labels_true == t) & (labels_pred == p))

    sum_comb_c = sum(combinations(n) for n in np.sum(contingency, axis=1))
    sum_comb_k = sum(combinations(n) for n in np.sum(contingency, axis=0))
    sum_comb = sum(combinations(n) for row in contingency for n in row)

    expected_index = sum_comb_c * sum_comb_k / combinations(len(labels_true))
    max_index = (sum_comb_c + sum_comb_k) / 2
    ari = (sum_comb - expected_index) / (max_index - expected_index)
    return ari


# -------------------------------------------------------
# Run DBSCAN and evaluate
# -------------------------------------------------------
labels = dbscan(X, eps=0.5, min_pts=5)

print("Cluster Assignments:", np.unique(labels))
print("Purity:", purity_score(y, labels))
print("ARI:", adjusted_rand_index(y, labels))
print("Silhouette:", silhouette_score(X, labels))
