import pandas as pd
from typing import Any, Dict, Sequence, Tuple, Union
import math
import numpy as np
import random
import joblib
from itertools import combinations

# Type alias for nodes in decision tree
DecisionNode = Union["DecisionBranch", "DecisionLeaf"]

class DecisionLeaf:
    """Leaf node in decision tree"""

    def __init__(self, label):
        self.label = label

    def predict(self, x):
        return self.label

class DecisionBranch:
    """Branching node in decision tree"""

    def __init__(self, attr: str, branches: Dict[Any, DecisionNode], default_label: Any):
        self.attr = attr
        self.branches = branches
        self.default_label = default_label

    def predict(self, x: pd.Series):
        if x[self.attr] in self.branches:
            return self.branches[x[self.attr]].predict(x)
        else:
            return self.default_label


def mtry(attrs: Sequence[str]) -> Sequence[str]:
    """Return number of attributes to consider for each split"""
    num_vars = len(attrs)
    num_to_select = max(1, int(math.sqrt(num_vars)))
    return random.sample(attrs, num_to_select)

def information_gain(X: pd.DataFrame, y: pd.Series, attr: str) -> float:
    total_mean = y.mean()
    unique_values = X[attr].unique()
    subsets = []

    for value in unique_values:
        subset_X = X[X[attr] == value]
        subset_y = y[X[attr] == value]
        subsets.append((subset_X, subset_y))

    information_gain = 0
    total_samples = len(y)

    for subset_X, subset_y in subsets:
        subset_mean = subset_y.mean()
        subset_weight = len(subset_y) / total_samples
        information_gain += subset_weight * abs(total_mean - subset_mean)

    return information_gain

def bootStrap(y: pd.Series, x: pd.DataFrame, random_state: int = None):
    n_samples = len(x)
    rng = np.random.default_rng(seed=random_state)
    boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
    x_boot = x.iloc[boot_indices]
    y_boot = y.iloc[boot_indices]
    oob_mask = ~np.isin(range(n_samples), boot_indices)
    x_oob = x.iloc[oob_mask]
    y_oob = y.iloc[oob_mask]
    oob_indices = x.index[oob_mask]
    return x_boot, y_boot, x_oob, y_oob, boot_indices, oob_indices

def learn_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    attrs: Sequence[str],
    y_parent: pd.Series,
    min_samples_split: int = 10
) -> DecisionNode:
    if len(y.unique()) == 1 or len(y) < min_samples_split:
        return DecisionLeaf(y.iloc[0])
    if len(attrs) == 0 or X.empty:
        return DecisionLeaf(round(y_parent.mean()))
    
    best_attr = max(attrs, key=lambda attr: information_gain(X, y, attr))
    attrs = [attr for attr in attrs if attr != best_attr]
    
    branches = {}
    for val in X[best_attr].unique():
        subset_X = X[X[best_attr] == val]
        subset_y = y[X[best_attr] == val]
        branches[val] = learn_decision_tree(subset_X, subset_y, attrs, y, min_samples_split)
    
    default_label = y.mode()[0]
    return DecisionBranch(best_attr, branches, default_label)

def fit(X: pd.DataFrame, y: pd.Series, attrs: Sequence[str], min_samples_split: int = 10) -> DecisionNode:
    return learn_decision_tree(X, y, attrs, y, min_samples_split)

def predict(tree: DecisionNode, X: pd.DataFrame):
    return X.apply(lambda x: tree.predict(x), axis=1)

def kfoldData(X, y, y2, k):
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    X = X.iloc[indices]
    y = y.iloc[indices]
    y2 = y2.iloc[indices]
    fold_size = n // k
    X_folds = []
    y_folds = []
    y2_folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        if i == k - 1:
            end = n
        X_folds.append(X.iloc[start:end])
        y_folds.append(y.iloc[start:end])
        y2_folds.append(y2.iloc[start:end])
    return X_folds, y_folds, y2_folds

def compute_metrics(y_true, y_pred):
    mean_difference = (y_true - y_pred).mean()
    return {"mean_difference": mean_difference}

def compute_acc(y_true, y_pred):
    mean_difference = abs((y_true - y_pred)).mean()
    return {"mean_difference": mean_difference}

def build_forest(X: pd.DataFrame, y: pd.Series, n_trees: int = 10, min_samples_split: int = 10, random_state: int = None) -> Tuple[list, pd.DataFrame]:
    oob_x = []
    oob_y = []
    oob_indices_list = []
    boot_indices_list = []
    unused_attrs = []
    trees = []
    for i in range(n_trees):
        boot_data = bootStrap(y, X, random_state=i)
        attrs = mtry(list(X.columns))

        used_attrs = set(attrs)
        all_attrs = set(X.columns)
        unused_attrs.append(all_attrs - used_attrs)
        
        tree = fit(boot_data[0], boot_data[1], attrs, min_samples_split)
        trees.append(tree)

        # Adding all of the data needed for importance calculation/acc
        oob_x.append(boot_data[2])
        oob_y.append(boot_data[3])
        boot_indices_list.append(boot_data[4])
        oob_indices_list.append(boot_data[5])

    # Align the OOB datasets using the indices
    aligned_oob_preds = pd.DataFrame(index=X.index)
    for i in range(len(trees)):
        aligned_oob_preds.loc[oob_indices_list[i], i] = predict(trees[i], oob_x[i])

    return trees, aligned_oob_preds

# Load the data
data = pd.read_csv('BridgeDataNoTrump.csv')

# Define the target variable and features
target = 'avgTricks'
compare = 'diff_tricks_contract'
features = data.columns.drop([target, compare])

# Split the data into y and x
X = data[features]
y = data[target]
y2 = abs(data[compare])

# Build the forest
trees, aligned_oob_preds = build_forest(X, y, n_trees=100, min_samples_split=10)

# Predict the target variable on the test set using majority voting
final_predictions = aligned_oob_preds.mode(axis=1)[0]

# Ensure the indices of final_predictions match the original indices of X
final_predictions.index = X.index

# Evaluate the model
metrics = compute_acc(y, final_predictions)
y2_mean_diff = y2.mean() - abs(metrics['mean_difference'])

for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value}")
print(y2.mean())
print(f" mean_difference + y2.mean(): {y2_mean_diff}")

# 5-fold cross-validation
k = 5
X_folds, y_folds, y2_folds = kfoldData(X, y, y2, k)

# Collect metrics for each fold
fold_metrics = []

for i in range(k):
    print(f"Fold {i+1}")
    X_train = pd.concat([X_folds[j] for j in range(k) if j != i])
    y_train = pd.concat([y_folds[j] for j in range(k) if j != i])
    y2_train = pd.concat([y2_folds[j] for j in range(k) if j != i])
    X_test = X_folds[i]
    y_test = y_folds[i]
    y2_test = abs(y2_folds[i])  # Apply absolute value to y2_test

    # Build the forest for the current fold
    trees, aligned_oob_preds = build_forest(X_train, y_train, n_trees=100, min_samples_split=10)

    # Predict the target variable on the test set using majority voting
    aligned_oob_preds_test = pd.DataFrame(index=X_test.index)
    for j in range(len(trees)):
        aligned_oob_preds_test.loc[:, j] = predict(trees[j], X_test)

    final_predictions = aligned_oob_preds_test.mode(axis=1)[0]

    # Evaluate the model
    if final_predictions.isna().any():
        final_predictions = final_predictions.fillna(final_predictions.mean())

    metrics = compute_acc(y_test, final_predictions)
    y2_mean_diff = abs(y2_test.mean()) - abs(metrics['mean_difference'])

    fold_metrics.append(metrics['mean_difference'])

# Calculate mean metrics across all folds
mean_fold_metrics = np.mean(fold_metrics)
print(f"Mean difference across all folds: {mean_fold_metrics}")

# Extract predictions for each level of each attribute and compute mean y value for each subset
level_importance_scores = {}
mean_final_predictions = final_predictions.mean()
for col in X.columns:
    level_importance_scores[col] = {}
    for level in X[col].unique():
        row_indices = X.index[X[col] == level].tolist()
        # Ensure indices are in final_predictions
        row_indices = [idx for idx in row_indices if idx in final_predictions.index]
        predictions = final_predictions.loc[row_indices]
        mean_prediction = predictions.mean()
        importance_score = (mean_prediction - mean_final_predictions) * (40 / 13)  # Scale to HCP
        level_importance_scores[col][level] = importance_score

# Print the importance score for each level of each attribute
for col, levels in level_importance_scores.items():
    for level, importance_score in levels.items():
        print(f"Importance score for {col} = {level}: {importance_score}")

# Save the model to a file
joblib.dump(trees, 'forest_model.pkl')