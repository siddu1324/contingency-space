"""# from sklearn.metrics import accuracy_score

# Define a custom metric function
def accuracy_metric(cm):
    total_predictions = cm.tp + cm.fp + cm.fn + cm.tn
    correct_predictions = cm.tp + cm.tn
    return correct_predictions / total_predictions if total_predictions > 0 else 0

# Create a series of confusion matrices
cm1 = CM(tp=50, fn=10, tn=80, fp=5)
cm2 = CM(tp=55, fn=5, tn=85, fp=0)
cm3 = CM(tp=60, fn=0, tn=90, fp=5)
path = [cm1, cm2, cm3]

# Initialize the LearningPath with the path of CMs and the custom accuracy metric
learning_path = LearningPath(path=path, metric=accuracy_metric)

# Compute the impact of changes along the path
learning_path.compute_impact()

# Retrieve results to analyze
results = learning_path.get_results()
print("Evaluation Results:", results)"""


import numpy as np
from sklearn.utils import check_array
from sklearn.base import BaseEstimator
from sklearn.metrics import check_scoring
from copy import deepcopy

class LearningPath(BaseEstimator):
    """
    Analyzes the impact of changes in model configurations over time using a sequence of confusion matrices.

    Parameters
    ----------
    path : list of CM
        A list of confusion matrix instances representing different states of a model.

    metric : function
        A function that takes a confusion matrix and returns a computed metric.

    normalize : bool, default=True
        Whether to normalize the confusion matrices before metric computation.

    Attributes
    ----------
    scores_ : list
        Scores computed for each confusion matrix in the path.

    score_changes_ : list
        Changes in scores from one confusion matrix to the next.

    cm_steps_ : list
        Distances between consecutive confusion matrices in terms of CM metrics.

    cs_steps_ : list
        Distances in a higher-dimensional space that includes the metric as an additional dimension.

    triangle_areas_ : list
        Areas of triangles formed by consecutive confusion matrices and the metric dimension.
    """
    def __init__(self, path, metric, normalize=True):
        if not path:
            raise ValueError("Path cannot be empty.")
        self.path = deepcopy(path)
        self.metric = check_scoring(metric, allow_none=False)
        self.normalize = normalize
        self.scores_ = []
        self.score_changes_ = []
        self.cm_steps_ = []
        self.cs_steps_ = []
        self.triangle_areas_ = []

    def fit(self):
        """
        Computes the metrics and changes along the learning path.

        This method fits the model to the learning path and computes various statistics to understand the evolution of the model's performance.

        Returns
        -------
        self : object
            Returns self.
        """
        scores, score_changes, cm_steps, cs_steps, triangle_areas = [], [], [], [], []
        pc = PerformanceComparison(self.metric, self.normalize)

        for i in range(len(self.path) - 1):
            cm_current, cm_next = self.path[i], self.path[i + 1]
            score_current = self.metric(cm_current)
            score_next = self.metric(cm_next)
            scores.append(score_current)

            # Metric difference
            score_diff = pc.compare_by_metric(cm_current, cm_next)
            score_changes.append(score_diff)

            # CM distance in the metric space
            cm_dist = pc.compare_by_2d_distance(cm_current, cm_next)
            cm_steps.append(cm_dist)

            # CS distance including the metric as a dimension
            cs_dist = pc.compare_by_3d_distance(cm_current, cm_next)
            cs_steps.append(cs_dist)

            # Triangle area in CS space
            triangle_area = pc.compare_by_3d_triangle(cm_current, cm_next)
            triangle_areas.append(triangle_area)

        # Store the last score
        scores.append(self.metric(self.path[-1]))

        # Update attributes
        self.scores_ = deepcopy(scores)
        self.score_changes_ = deepcopy(score_changes)
        self.cm_steps_ = deepcopy(cm_steps)
        self.cs_steps_ = deepcopy(cs_steps)
        self.triangle_areas_ = deepcopy(triangle_areas)
        
        return self

    def get_results(self):
        """
        Retrieves the computed results after the model has been fitted.

        Returns
        -------
        results : dict
            A dictionary containing the computed metrics and changes.
        """
        return {
            "scores": self.scores_,
            "score_changes": self.score_changes_,
            "cm_steps": self.cm_steps_,
            "cs_steps": self.cs_steps_,
            "triangle_areas": self.triangle_areas_
        }
