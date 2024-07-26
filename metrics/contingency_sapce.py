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

        :param tp: True Positives
        :param fn: False Negatives
        :param tn: True Negatives
        :param fp: False Positives
        """
        self.tp = tp
        self.fn = fn
        self.tn = tn
        self.fp = fp
        self.n = self.tn + self.fp
        self.p = self.tp + self.fn

    def normalize(self):
        """
        Normalizes all entries of the confusion matrix.
        """
        total = self.tp + self.fn + self.tn + self.fp
        if total != 0:
            return CM(self.tp / self.p, self.fn / self.p, self.tn / self.n, self.fp / self.n)
        else:
            return CM(0, 0, 0, 0)

    def __repr__(self):
        return f"TP: {self.tp}, FN: {self.fn}, TN: {self.tn}, FP: {self.fp}"

class CMGenerator:
    def __init__(self, n_p, n_n, n_cm):
        """
        Initializes a generator to create a grid of confusion matrices by varying TPR (True Positive Rate) and TNR (True Negative Rate).
        """
        self.n_p = n_p
        self.n_n = n_n
        self.n_cm = n_cm

    def generate_cms(self):
        """
        Generates confusion matrices across a range of thresholds for TPR and TNR.
        """
        tpr_values = np.linspace(0, 1, self.n_cm)
        tnr_values = np.linspace(0, 1, self.n_cm)
        cms = []
        for tpr in tpr_values:
            for tnr in tnr_values:
                tp = int(self.n_p * tpr)
                fn = self.n_p - tp
                tn = int(self.n_n * tnr)
                fp = self.n_n - tn
                cms.append(np.array([[tn, fp], [fn, tp]]))
        return cms

class ContingencySpace:
    def __init__(self, n_p, n_n, n_cm, metrics):
        """
        Initializes the Contingency Space with a confusion matrix generator and a set of metrics.
        """
        self.cm_generator = CMGenerator(n_p, n_n, n_cm)
        self.metrics = metrics

    def generate_contingency_space(self):
        """
        This function can be used to apply each metric to each confusion matrix generated by CMGenerator in the future.
        """
        pass  # Currently, this function does not perform any actions.

    def compute_metric_values(self, cms, metric: Callable):
        """
        Calculate metric values for all confusion matrices using the given metric function.
        """
        return [metric(cm) for cm in cms]

    def compute_imbalance_sensitivity(self, metric: Callable, imbalance_ratio: float, space_w: int):
        """
        Computes the sensitivity of the specified metric to a single change in class imbalance.
        """
        initial_cms = self.cm_generator.generate_cms()  # Generating initial CMs directly from CMGenerator
        initial_vals = self.compute_metric_values(initial_cms, metric)
        vals_as_mat_0 = np.flip(np.array(initial_vals).reshape((space_w, space_w)), 0)

        adjusted_n_n = int(self.cm_generator.n_p * imbalance_ratio)
        self.cm_generator.n_n = adjusted_n_n
        adjusted_cms = self.cm_generator.generate_cms()  # Re-using CMGenerator for adjusted scenario
        adjusted_vals = self.compute_metric_values(adjusted_cms, metric)
        vals_as_mat = np.flip(np.array(adjusted_vals).reshape((space_w, space_w)), 0)

        diff = np.abs(vals_as_mat - vals_as_mat_0)
        return np.sum(diff)

"""
# Example Usage
n_p = 100
n_n = 100
n_cm = 10
metrics = {
    'Accuracy': lambda cm: np.trace(cm) / np.sum(cm) if np.sum(cm) != 0 else 0
}

contingency_space = ContingencySpace(n_p, n_n, n_cm, metrics)
imbalance_ratio = 4  # Example imbalance ratio
space_w = 10  # Example space width for reshaping
sensitivity_result = contingency_space.compute_imbalance_sensitivity(metrics['Accuracy'], imbalance_ratio, space_w)
print("Imbalance Sensitivity Result for Accuracy:", sensitivity_result) """


    
        


class PerformanceComparison:
    """
    Provides methods to compare two confusion matrices.
    """
    def compare_by_2d_distance(self, cm1, cm2):
        """
        Computes the distance (always positive) between the two given confusion matrices in the 2d contingency space.
        """
        pass

    def compare_by_3d_distance(self, cm1, cm2):
        """
        Computes the distance (always positive) between the two given confusion matrices in the 3d contingency space.
        """
        pass

    def compare_by_3d_triangle(self, cm1, cm2):
        """
        Compares the two given confusion matrices by considering them as two model points in the Contingency Space
        and computing the area of the right triangle they form.
        """
        pass


class LearningPath:
    """
    Evaluates the learning path of model configurations using provided metrics.
    """
    def __init__(self, cms, metric_func):
        """
        Initializes the learning path with confusion matrices and a metric function.
        :param cms: List of confusion matrices directly provided or generated.
        :param metric_func: Function to calculate the metric.
        """
        self.cms = cms
        self.metric_func = metric_func
        self.metric_values = [metric_func(cm) for cm in cms]
        self.changes = self.compute_changes()
        self.distances = self.compute_path_length()

    def compute_changes(self):
        """
        Computes changes in the metric values along the path.
        """
        changes = [self.metric_values[i] - self.metric_values[i - 1] for i in range(1, len(self.metric_values))]
        return changes

    def compute_path_length(self):
        """
        Computes the cumulative Euclidean distance ('length') along the path based on metric values.
        """
        distances = [np.abs(change) for change in self.changes]
        total_distance = np.sum(distances)
        return total_distance

    def __repr__(self):
        return f"LearningPath with {len(self.cms)} points. Total change: {self.compute_path_length()}"





# import numpy as np
# from typing import Callable

# class ContingencySpace:
#     def __init__(self, n_p, n_n, n_cm, metrics):
#         """
#         Initializes the Contingency Space with a confusion matrix generator and a set of metrics.
#         """
#         self.cm_generator = CMGenerator(n_p, n_n, n_cm)
#         self.metrics = metrics

#     def generate_contingency_space(self):
#         """
#         Generates the contingency space by applying each metric to each confusion matrix generated by CMGenerator.
#         """
#         return self.cm_generator.generate_cms()

#     def compute_metric_values(self, cms, metric: Callable):
#         """
#         Calculate metric values for all confusion matrices using the given metric function.
#         """
#         return [metric(cm) for cm in cms]

#     def compute_imbalance_sensitivity(self, metric: Callable, imbalance_ratio: float, space_w: int):
#         """
#         Computes the sensitivity of the specified metric to a single change in class imbalance.

#         Parameters:
#             metric (Callable): A function that computes a metric from a confusion matrix.
#             imbalance_ratio (float): The ratio of negative to positive samples.
#             space_w (int): The width for reshaping metric values for comparison.

#         Example Usage:
#             # Compute imbalance sensitivity for a custom accuracy metric
#             accuracy_metric = lambda cm: np.trace(cm) / np.sum(cm) if np.sum(cm) != 0 else 0
#             sensitivity_result = contingency_space.compute_imbalance_sensitivity(accuracy_metric, 4, 10)
#             print("Imbalance Sensitivity Result for Accuracy:", sensitivity_result)
#         """
        
#         # Generate initial scenario for comparison (balanced case)
#         initial_cms = self.generate_contingency_space()
#         initial_vals = self.compute_metric_values(initial_cms, metric)
#         vals_as_mat_0 = np.flip(np.array(initial_vals).reshape((space_w, space_w)), 0)

#         # Adjust the number of negatives based on the imbalance ratio and recompute the CMs
#         adjusted_n_n = int(self.cm_generator.n_p * imbalance_ratio)
#         self.cm_generator.n_n = adjusted_n_n
#         adjusted_cms = self.generate_contingency_space()
#         adjusted_vals = self.compute_metric_values(adjusted_cms, metric)
#         vals_as_mat = np.flip(np.array(adjusted_vals).reshape((space_w, space_w)), 0)

#         # Compute the volume of difference from the initial configuration
#         diff = np.abs(vals_as_mat - vals_as_mat_0)
#         return np.sum(diff)





# #example - 1
# class LearningPath:
#     """
#     Evaluates the learning path of model configurations using provided metrics.
#     """
#     def __init__(self, cms, metric_func):
#         """
#         Initializes the learning path with confusion matrices and a metric function.
#         :param cms: List of confusion matrices.
#         :param metric_func: Function to calculate the metric.
#         """
#         self.cms = cms
#         self.metric_func = metric_func
#         self.metric_values = self.compute_metric_values(self.cms, self.metric_func)
#         self.distances = []

#     def compute_metric_values(self, cms, metric):
#         """
#         Calculate metric values for all confusion matrices using the given metric function.
#         """
#         return [metric(cm) for cm in cms]

#     def compute_changes(self):
#         """
#         Computes changes in the metric values along the path.
#         """
#         changes = []
#         for i in range(1, len(self.metric_values)):
#             change = self.metric_values[i] - self.metric_values[i - 1]
#             changes.append(change)
#         return changes

#     def compute_path_length(self):
#         """
#         Computes the cumulative Euclidean distance ('length') along the path based on metric values.
#         """
#         self.distances = [np.abs(self.metric_values[i] - self.metric_values[i - 1]) for i in range(1, len(self.metric_values))]
#         return np.sum(self.distances)

#     def __repr__(self):
#         return f"LearningPath with {len(self.cms)} points"