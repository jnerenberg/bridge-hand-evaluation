import pandas as pd
from typing import Any, Dict, Sequence, Tuple, Union
import math
import numpy as np
import random
import joblib

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
    y_parent: pd.Series
) -> DecisionNode:
    if len(y.unique()) == 1:
        return DecisionLeaf(y.iloc[0])
    if len(attrs) == 0 or X.empty:
        return DecisionLeaf(round(y_parent.mean()))
    
    best_attr = max(attrs, key=lambda attr: information_gain(X, y, attr))
    attrs = [attr for attr in attrs if attr != best_attr]
    
    branches = {}
    for val in X[best_attr].unique():
        subset_X = X[X[best_attr] == val]
        subset_y = y[X[best_attr] == val]
        branches[val] = learn_decision_tree(subset_X, subset_y, attrs, y)
    
    default_label = y.mode()[0]
    return DecisionBranch(best_attr, branches, default_label)

def fit(X: pd.DataFrame, y: pd.Series, attrs: Sequence[str]) -> DecisionNode:
    return learn_decision_tree(X, y, attrs, y)

def predict(tree: DecisionNode, X: pd.DataFrame):
    return X.apply(lambda x: tree.predict(x), axis=1)

def compute_metrics(y_true, y_pred):
    mean_difference = (y_true - y_pred).mean()
    return {"mean_difference": mean_difference}

def compute_acc(y_true, y_pred):
    mean_difference = abs((y_true - y_pred)).mean()
    return {"mean_difference": mean_difference}

# Load the data
data = pd.read_csv('BridgeDataTrump.csv')

# Define the target variable and features
target = 'diff_tricks_contract'
features = data.columns.drop(target)

# Split the data into y and x
X = data[features]
y = data[target]

oob_x = []
oob_y = []
oob_indices_list = []
boot_indices_list = []
unused_attrs = []
trees = []
for i in range(100): #number of trees
    boot_data = bootStrap(y, X, random_state=i)
    attrs = mtry(list(X.columns))

    used_attrs = set(attrs)
    all_attrs = set(X.columns)
    unused_attrs.append(all_attrs - used_attrs)
    
    tree = fit(boot_data[0], boot_data[1], attrs)
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

# Predict the target variable on the test set using majority voting
final_predictions = aligned_oob_preds.mode(axis=1)[0]

# Evaluate the model
metrics = compute_acc(y, final_predictions)
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value}")

# Calculate the change in mean for each level of each attribute
level_importance_scores = {}

for feature in features:
    level_importance_scores[feature] = {}
    unique_levels = X[feature].unique()
    
    for level in unique_levels:
        # Filter the dataset to include only rows where the feature has the specific level
        filtered_data = data[data[feature] == level]
        level_importance_scores[feature][level] = filtered_data

# Make predictions for each filtered dataset using the random forest
level_predictions = {}

for feature, levels in level_importance_scores.items():
    level_predictions[feature] = {}
    for level, dataset in levels.items():
        if not dataset.empty:
            X_filtered = dataset[features]
            y_filtered = dataset[target]
            
            # List to store predictions for each tree
            tree_preds = []
            for tree in trees:
                preds = predict(tree, X_filtered)
                tree_preds.append(preds)
            
            # Combine predictions into a DataFrame
            combined_preds = pd.concat(tree_preds, axis=1)
            
            # Majority voting for final predictions
            final_preds = combined_preds.mode(axis=1)[0]
            
            # Store the final predictions
            level_predictions[feature][level] = final_preds

# Compute the mean final y for the complete set
mean_final_y_complete = final_predictions.mean()

# Compare the mean final y for each of the filtered sets to the mean final y for the complete set
mean_final_y_differences = {}

for feature, levels in level_predictions.items():
    mean_final_y_differences[feature] = {}
    for level, preds in levels.items():
        mean_final_y_filtered = preds.mean()
        mean_final_y_diff = mean_final_y_filtered - mean_final_y_complete
        # Multiply the mean difference by 40/13
        mean_final_y_differences[feature][level] = mean_final_y_diff * (40 / 13)

# Print the mean final y differences for each level of each feature
print(f"Mean final y for the complete set: {mean_final_y_complete}")
for feature, levels in mean_final_y_differences.items():
    for level, mean_final_y_diff in levels.items():
        print(f"Mean final y difference for {feature} = {level}: {mean_final_y_diff}")

# Save the model to a file
joblib.dump(trees, 'forest_model.pkl')