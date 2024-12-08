import unittest
import pandas as pd
from eval7 import DecisionLeaf, DecisionBranch, mtry, information_gain, bootStrap, learn_decision_tree, fit, predict, compute_metrics, compute_acc
from sklearn.model_selection import KFold

# test_eval7.py

class TestDecisionLeaf(unittest.TestCase):

    def test_predict(self):
        leaf = DecisionLeaf(1)
        self.assertEqual(leaf.predict(None), 1)

    def test_predict_with_different_label(self):
        leaf = DecisionLeaf(2)
        self.assertEqual(leaf.predict(None), 2)

if __name__ == '__main__':
    unittest.main()

# 5-fold cross-validation for eval7.py
def cross_validate(data, target, features, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data.iloc[train_index][features], data.iloc[test_index][features]
        y_train, y_test = data.iloc[train_index][target], data.iloc[test_index][target]

        oob_x = []
        oob_y = []
        oob_indices_list = []
        boot_indices_list = []
        unused_attrs = []
        trees = []

        for i in range(100):  # number of trees
            boot_data = bootStrap(y_train, X_train, random_state=i)
            attrs = mtry(list(X_train.columns))

            used_attrs = set(attrs)
            all_attrs = set(X_train.columns)
            unused_attrs.append(all_attrs - used_attrs)

            tree = fit(boot_data[0], boot_data[1], attrs)
            trees.append(tree)

            # Adding all of the data needed for importance calculation/acc
            oob_x.append(boot_data[2])
            oob_y.append(boot_data[3])
            boot_indices_list.append(boot_data[4])
            oob_indices_list.append(boot_data[5])

        # Align the OOB datasets using the indices
        aligned_oob_preds = pd.DataFrame(index=X_train.index)
        for i in range(len(trees)):
            aligned_oob_preds.loc[oob_indices_list[i], i] = predict(trees[i], oob_x[i])

        # Predict the target variable on the test set using majority voting
        final_predictions = aligned_oob_preds.mode(axis=1)[0]

        # Evaluate the model
        metrics = compute_acc(y_train, final_predictions)
        metrics_list.append(metrics)

    # Calculate average metrics
    avg_metrics = {key: sum(d[key] for d in metrics_list) / n_splits for key in metrics_list[0]}
    return avg_metrics

# Load the data
data = pd.read_csv('BridgeDataTrump.csv')

# Define the target variable and features
target = 'avgTricks'
features = data.columns.drop(target)

# Perform 5-fold cross-validation
avg_metrics = cross_validate(data, target, features, n_splits=5)
print("Average metrics from 5-fold cross-validation:")
for metric, value in avg_metrics.items():
    print(f"{metric.capitalize()}: {value}")