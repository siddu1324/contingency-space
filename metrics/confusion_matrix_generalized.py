import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class CMGeneralized:
    """
    Enhanced confusion matrix class for multi-class problems that integrates directly
    with scikit-learn's confusion_matrix function and provides comprehensive matrix manipulations.
    """
    def __init__(self, y_true, y_pred, labels=None):
        """
        Initializes the confusion matrix from true labels and predicted labels.

        :param y_true: array-like of shape (n_samples,) - True labels.
        :param y_pred: array-like of shape (n_samples,) - Predicted labels.
        :param labels: array-like of shape (n_classes,) - List of labels to index the matrix. 
                       This is optional and can be inferred from the data.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels if labels is not None else np.unique(np.concatenate([y_true, y_pred]))
        self.cm = confusion_matrix(y_true, y_pred, labels=self.labels)

    def normalize(self, mode='true'):
        """
        Normalizes the confusion matrix over the specified mode.

        :param mode: str - 'true' (normalize by row), 'pred' (normalize by column), or 'all' (normalize by total).
        :return: The normalized confusion matrix as a pandas DataFrame.
        """
        if mode == 'true':
            self.cm = self.cm.astype(float) / self.cm.sum(axis=1, keepdims=True)
        elif mode == 'pred':
            self.cm = self.cm.astype(float) / self.cm.sum(axis=0, keepdims=True)
        elif mode == 'all':
            self.cm = self.cm.astype(float) / self.cm.sum()
        return pd.DataFrame(self.cm, index=self.labels, columns=self.labels)

    def totals(self):
        """
        Computes the totals for true positives, false positives, and false negatives.

        :return: Dictionary containing total true positives, false positives, and false negatives.
        """
        total_true_positive = np.trace(self.cm)
        total_false_positive = np.sum(self.cm, axis=0) - np.diag(self.cm)
        total_false_negative = np.sum(self.cm, axis=1) - np.diag(self.cm)
        return {
            'total_true_positive': total_true_positive,
            'total_false_positive': total_false_positive,
            'total_false_negative': total_false_negative
        }

    def __repr__(self):
        return pd.DataFrame(self.cm, index=self.labels, columns=self.labels).__str__()

# Example Usage
y_true = [1, 2, 3, 4, 2, 3]
y_pred = [1, 2, 3, 3, 3, 2]
cmg = CMGeneralized(y_true, y_pred)
print("Original Matrix:\n", cmg)
print("Normalized by True Labels:\n", cmg.normalize('true'))
print("Totals:\n", cmg.totals())
