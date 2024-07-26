import numpy as np
import pandas as pd
from typing import Callable

class CM:
    """
    Confusion matrix class for binary problems.
    """
    def __init__(self, tp, fn, tn, fp):
        """
        The class constructor.
        """
        self.tp = tp
        self.fn = fn
        self.tn = tn
        self.fp = fp
        self.n = self.tn + self.fp  # Total negatives
        self.p = self.tp + self.fn  # Total positives

    def normalize(self):
        """
        Normalizes all entries of the confusion matrix.
        """
        return CM(
            tp=self.tp / self.p if self.p != 0 else 0,
            fn=self.fn / self.p if self.p != 0 else 0,
            tn=self.tn / self.n if self.n != 0 else 0,
            fp=self.fp / self.n if self.n != 0 else 0
        )

    def __repr__(self):
        return f"CM(TP: {self.tp}, FN: {self.fn}, TN: {self.tn}, FP: {self.fp})"

class LearningPath:
    """
    Evaluates the learning path of model configurations using provided metrics.
    """
    def __init__(self, cms, metric_func: Callable):
        """
        Initializes the learning path with confusion matrices and a metric function.
        """
        self.cms = [cm.normalize() for cm in cms]  # Normalize all CMs
        self.metric_func = metric_func
        self.metric_values = [self.normalize_metric(metric_func(cm)) for cm in self.cms]
        self.points_2d = [(cm.tn / (cm.tn + cm.fp), cm.tp / (cm.tp + cm.fn)) for cm in self.cms]
        self.points_3d = [(cm.tn / (cm.tn + cm.fp), cm.tp / (cm.tp + cm.fn), self.metric_func(cm)) for cm in self.cms]

    def normalize_metric(self, value):
        """
        Normalizes metric values to the range [0, 1].
        Example normalization: (value + 1) / 2 for metrics in range [-1, 1]
        """
        return (value + 1) / 2 if value < 0 else value

    def compute_2d_path_length(self):
        """
        Computes the 2D Euclidean distance ('length') along the path using TNR and TPR coordinates.
        """
        return sum(np.linalg.norm(np.array(self.points_2d[i+1]) - np.array(self.points_2d[i])) for i in range(len(self.points_2d) - 1))

    def compute_3d_path_length(self):
        """
        Computes the 3D Euclidean distance along the path using TNR, TPR, and metric value as coordinates.
        """
        return sum(np.linalg.norm(np.array(self.points_3d[i+1]) - np.array(self.points_3d[i])) for i in range(len(self.points_3d) - 1))

    def __repr__(self):
        return f"LearningPath with {len(self.cms)} points. 2D Length: {self.compute_2d_path_length()}, 3D Length: {self.compute_3d_path_length()}"

# Example Usage:
def accuracy_metric(cm):
    return (cm.tp + cm.tn) / (cm.tp + cm.fn + cm.tn + cm.fp) if (cm.tp + cm.fn + cm.tn + cm.fp) != 0 else 0

# Sample CMs for testing
cms = [
    CM(50, 50, 30, 70),
    CM(60, 40, 40, 60),
    CM(70, 30, 50, 50),
    CM(80, 30, 40, 60),
]

path = LearningPath(cms, accuracy_metric)
print(path)


def matthews_correlation_coefficient(cm):
    """
    Calculates the Matthews Correlation Coefficient for a given confusion matrix.
    """
    TP = cm.tp
    TN = cm.tn
    FP = cm.fp
    FN = cm.fn
    
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    return numerator / denominator if denominator != 0 else 0

cms2 = [
    CM(50, 10, 80, 10),
    CM(60, 20, 60, 40),
    CM(70, 10, 90, 10),
    CM(85, 15, 80, 20),
]

path = LearningPath(cms2, matthews_correlation_coefficient)
print(path)