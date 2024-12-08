import pandas as pd
from typing import Any, Dict, Sequence, Tuple, Union

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
    return attrs

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
    return x_boot, y_boot, x_oob, y_oob

def learn_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    attrs: Sequence[str],
    y_parent: pd.Series,
) -> DecisionNode:
    if len(y.unique()) == 1:
        return DecisionLeaf(y.iloc[0])
    if len(attrs) == 0 or X.empty:
        return DecisionLeaf(y_parent.mode()[0])
    
    select_attrs = mtry(attrs)
    best_attr = max(select_attrs, key=lambda attr: information_gain(X, y, attr))
    attrs = [attr for attr in attrs if attr != best_attr]
    
    branches = {}
    for val in X[best_attr].unique():
        subset_X = X[X[best_attr] == val]
        subset_y = y[X[best_attr] == val]
        branches[val] = learn_decision_tree(subset_X, subset_y, attrs, y)
    
    default_label = y.mode()[0]
    return DecisionBranch(best_attr, branches, default_label)

def fit(X: pd.DataFrame, y: pd.Series) -> DecisionNode:
    attrs = list(X.columns)
    return learn_decision_tree(X, y, attrs, y)

def predict(tree: DecisionNode, X: pd.DataFrame):
    return X.apply(lambda x: tree.predict(x), axis=1)

def compute_metrics(y_true, y_pred):
    accuracy = (y_true == y_pred).mean()
    return {"accuracy": accuracy}

# Load the data
data = pd.read_csv('BridgeDataTrump.csv')

# Define the target variable and features
target = 'avgTricks'
features = data.columns.drop(target)

# Split the data into training and testing sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Train the decision tree
tree = fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = predict(tree, X_test)

# Evaluate the model
metrics = compute_metrics(y_test, y_pred)
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value}")

# Save the model to a file
import joblib
joblib.dump(tree, 'model.pkl')