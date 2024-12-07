"""
CS311 Final Project 

Full Name: Julia Nerenberg & Lia Smith


"""

import argparse, os, random, sys
from typing import Any, Dict, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import math

# Type alias for nodes in decision tree
DecisionNode = Union["DecisionBranch", "DecisionLeaf"]


class DecisionBranch:
    """Branching node in decision tree"""

    def __init__(self, attr: str, branches: Dict[Any, DecisionNode]):
        """Create branching node in decision tree

        Args:
            attr (str): Splitting attribute
            branches (Dict[Any, DecisionNode]): Children nodes for each possible value of `attr`
        """
        self.attr = attr
        self.branches = branches

    def predict(self, x: pd.Series):
        """Return predicted labeled for array-like example x"""
        # TODO: Implement prediction based on value of self.attr in x
        
        return self.branches[x[self.attr]].predict(x)

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Test Feature", self.attr)
        for val, subtree in self.branches.items():
            print(" " * 4 * indent, self.attr, "=", val, "->", end=" ")
            subtree.display(indent + 1)


class DecisionLeaf:
    """Leaf node in decision tree"""

    def __init__(self, label):
        """Create leaf node in decision tree

        Args:
            label: Label for this node
        """
        self.label = label

    def predict(self, x):
        """Return predicted labeled for array-like example x"""
        return self.label

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Label=", self.label)


def mtry(attrs: Sequence[str]) -> Sequence[str]:
    """Return number of attributes to consider for each split"""
    num_vars = len(attrs)
    num_to_select = math.floor(math.sqrt(num_vars))
    return random.sample(attrs, num_to_select)

def information_gain(X: pd.DataFrame, y: pd.Series, attr: str) -> float:
    """Return the expected change in mean from splitting X,y by attr"""
    # TODO: Implement information gain metric for selecting attributes

    total_mean = y.mean() #mean tricks from big set
    unique_values = X[attr].unique() #unique values of attribute
    subsets = []

    for value in unique_values: #splitting the data
        subset_X = X[X[attr] == value]
        subset_y = y[X[attr] == value]
        subsets.append((subset_X, subset_y))

    information_gain = 0
    total_samples = len(y)

    #caclulating the information gained from changes in mean after split
    for (subset_X, subset_y) in subsets:
        subset_mean = subset_y.mean()
        subset_weight = len(subset_y) / total_samples
        information_gain += subset_weight * abs(total_mean-subset_mean)
        
    return information_gain

def bootStrap(y: pd.Series, x: pd.DataFrame, random_state: int = None): 
    n_samples = len(x)  # Number of rows
    rng = np.random.default_rng(seed=random_state)  # Random generator for reproducibility

    # Generate bootstrapped row numbers
    boot_indices = rng.choice(n_samples, size=n_samples, replace=True)

    # bootstrapped dataset
    x_boot = x.iloc[boot_indices]
    y_boot = y.iloc[boot_indices]

    # Determine out-of-bag (OOB) indices
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
    """Recursively learn the decision tree

    Args:
        X (pd.DataFrame): Table of examples (as DataFrame)
        y (pd.Series): array-like example labels (target values)
        attrs (Sequence[str]): Possible attributes to split examples
        y_parent (pd.Series): array-like example labels for parents (parent target values)

    Returns:
        DecisionNode: Learned decision tree node
    """
    # TODO: Implement recursive tree construction based on pseudo code in class
    # and the assignment
    if len(y.unique()) == 1:
        return DecisionLeaf(y.iloc[0])
    if len(attrs) == 0 or X.empty:
        return DecisionLeaf(y_parent.mode()[0])
    
    
    select_attrs = mtry(attrs)
    best_attr = max(select_attrs, key=lambda attr: information_gain(X, y, attr))
    attrs = [attr for attr in attrs if attr != best_attr]
    
    branches = {}
    for val in X[best_attr].cat.categories:
        subset_X = X[X[best_attr] == val]
        subset_y = y[X[best_attr] == val]
        branches[val] = learn_decision_tree(subset_X, subset_y, attrs, y)
    
    return DecisionBranch(best_attr, branches)


def fit(X: pd.DataFrame, y: pd.Series) -> DecisionNode:
    """Return train decision tree on examples, X, with labels, y"""
    # You can change the implementation of this function, but do not modify the signature
    return learn_decision_tree(X, y, X.columns, y)


def predict(tree: DecisionNode, X: pd.DataFrame):
    """Return array-like predctions for examples, X and Decision Tree, tree"""

    # You can change the implementation of this function, but do not modify the signature

    # Invoke prediction method on every row in dataframe. `lambda` creates an anonymous function
    # with the specified arguments (in this case a row). The axis argument specifies that the function
    # should be applied to all rows.
    return X.apply(lambda row: tree.predict(row), axis=1)


def load_adult(feature_file: str, label_file: str):

    # Load the feature file
    examples = pd.read_table(
        feature_file,
        dtype={
            "age": int,
            "workclass": "category",
            "education": "category",
            "marital-status": "category",
            "occupation": "category",
            "relationship": "category",
            "race": "category",
            "sex": "category",
            "capital-gain": int,
            "capital-loss": int,
            "hours-per-week": int,
            "native-country": "category",
        },
    )
    labels = pd.read_table(label_file).squeeze().rename("label")


    # TODO: Select columns and choose a discretization for any continuous columns. Our decision tree algorithm
    # only supports discretized features and so any continuous columns (those not already "category") will need
    # to be discretized.

    # For example the following discretizes "hours-per-week" into "part-time" [0,40) hours and
    # "full-time" 40+ hours. Then returns a data table with just "education" and "hours-per-week" features.

    examples["hours-per-week"] = pd.cut(
        examples["hours-per-week"],
        bins=[0, 40, sys.maxsize],
        right=False,
        labels=["part-time", "full-time"],
    )

    examples["age"] = pd.cut(
        examples["age"],
        bins=[0, 20, 40, 60, 80, sys.maxsize],
        right=False,
        labels=["0-20", "20-40", "40-60", "60-80", "80+"],
    )

    return examples[["education", "hours-per-week", "sex", "age"]], labels


# You should not need to modify anything below here


def load_examples(
    feature_file: str, label_file: str, **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load example features and labels. Additional arguments are passed to
    the pandas.read_table function.

    Args:
        feature_file (str): Delimited file of categorical features
        label_file (str): Single column binary labels. Column name will be renamed to "label".

    Returns:
        Tuple[pd.DataFrame,pd.Series]: Tuple of features and labels
    """
    return (
        pd.read_table(feature_file, dtype="category", **kwargs),
        pd.read_table(label_file, **kwargs).squeeze().rename("label"),
    )


def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary of metrics in including confusion matrix, accuracy, recall, precision and F1
    """
    return {
        "confusion": metrics.confusion_matrix(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test decision tree learner")
    parser.add_argument(
        "-p",
        "--prefix",
        default="small1",
        help="Prefix for dataset files. Expects <prefix>.[train|test]_[data|label].txt files (except for adult). Allowed values: small1, tennis, hepatitis, adult.",
    )
    parser.add_argument(
        "-k",
        "--k_splits",
        default=10,
        type=int,
        help="Number of splits for stratified k-fold testing",
    )

    args = parser.parse_args()

    if args.prefix != "adult":
        # Derive input files names for test sets
        train_data_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.train_data.txt"
        )
        train_labels_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.train_label.txt"
        )
        test_data_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.test_data.txt"
        )
        test_labels_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.test_label.txt"
        )

        # Load training data and learn decision tree
        train_data, train_labels = load_examples(train_data_file, train_labels_file)
        tree = fit(train_data, train_labels)
        tree.display()

        # Load test data and predict labels with previously learned tree
        test_data, test_labels = load_examples(test_data_file, test_labels_file)
        pred_labels = predict(tree, test_data)

        # Compute and print accuracy metrics
        predict_metrics = compute_metrics(test_labels, pred_labels)
        for met, val in predict_metrics.items():
            print(
                met.capitalize(),
                ": ",
                ("\n" if isinstance(val, np.ndarray) else ""),
                val,
                sep="",
            )
    else:
        # We use a slightly different procedure with "adult". Instead of using a fixed split, we split
        # the data k-ways (preserving the ratio of output classes) and test each split with a Decision
        # Tree trained on the other k-1 splits.
        data_file = os.path.join(os.path.dirname(__file__), "data", "adult.data.txt")
        labels_file = os.path.join(os.path.dirname(__file__), "data", "adult.label.txt")
        data, labels = load_adult(data_file, labels_file)

        scores = []

        kfold = StratifiedKFold(n_splits=args.k_splits)
        for train_index, test_index in kfold.split(data, labels):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            tree = fit(X_train, y_train)
            y_pred = predict(tree, X_test)
            scores.append(metrics.accuracy_score(y_test, y_pred))

            tree.display()

        print(
            f"Mean (std) Accuracy (for k={kfold.n_splits} splits): {np.mean(scores)} ({np.std(scores)})"
        )
