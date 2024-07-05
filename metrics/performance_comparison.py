import numpy as np
from utils.confusion_matrix import CM
from sklearn.base import BaseEstimator

class PerformanceComparison(BaseEstimator):
    """
    Provides various methods to compare the performance of two models based on their confusion matrices.

    Parameters
    ----------
    metric : callable
        A function that computes a performance metric given a confusion matrix.

    normalize : bool, default=True
        If True, normalizes the confusion matrices before performing comparisons. This affects the calculations
        for 2D and 3D distances.

    Attributes
    ----------
    metric : callable
        The metric function used for comparisons.

    normalize : bool
        Flag to indicate if the confusion matrices should be normalized before comparisons.
    """

    def __init__(self, metric, normalize=True):
        self.metric = metric
        self.normalize = normalize

    def compare_by_metric(self, cm1: CM, cm2: CM):
        """
        Compares two confusion matrices based on the specified metric.

        Parameters
        ----------
        cm1 : CM
            The first confusion matrix.

        cm2 : CM
            The second confusion matrix.

        Returns
        -------
        float
            The difference in the metric values between cm2 and cm1.
        """
        metric1 = self.metric(cm1)
        metric2 = self.metric(cm2)
        return metric2 - metric1

    def compare_by_2d_distance(self, cm1: CM, cm2: CM):
        """
        Computes the Euclidean distance between two confusion matrices in a 2D space.

        Parameters
        ----------
        cm1 : CM
            The first confusion matrix.

        cm2 : CM
            The second confusion matrix.

        Returns
        -------
        float
            The Euclidean distance between the two confusion matrices.
        """
        if self.normalize:
            cm1.normalize()
            cm2.normalize()
        p1 = np.array([cm1.tn, cm1.tp])
        p2 = np.array([cm2.tn, cm2.tp])
        return np.linalg.norm(p1 - p2)

    def compare_by_3d_distance(self, cm1: CM, cm2: CM):
        """
        Computes a 3D distance considering the metric as the third dimension.

        Parameters
        ----------
        cm1 : CM
            The first confusion matrix.

        cm2 : CM
            The second confusion matrix.

        Returns
        -------
        float
            The 3D distance incorporating the metric difference as the third dimension.
        """
        metric_diff = self.compare_by_metric(cm1, cm2)
        spatial_dist = self.compare_by_2d_distance(cm1, cm2)
        return np.sqrt(metric_diff**2 + spatial_dist**2)

    def compare_by_3d_triangle(self, cm1: CM, cm2: CM):
        """
        Computes the area of a triangle formed by the two confusion matrices and the metric difference in a 3D space.

        Parameters
        ----------
        cm1 : CM
            The first confusion matrix.

        cm2 : CM
            The second confusion matrix.

        Returns
        -------
        float
            The area of the triangle formed by the two points and the metric.
        """
        base = self.compare_by_2d_distance(cm1, cm2)
        height = self.compare_by_metric(cm1, cm2)
        return 0.5 * base * height

# Example usage
def main():
    from sklearn.metrics import accuracy_score
    cm1 = CM(tp=50, fn=10, tn=30, fp=5)
    cm2 = CM(tp=45, fn=15, tn=35, fp=10)

    def accuracy_metric(cm):
        total = cm.tp + cm.fn + cm.tn + cm.fp
        correct = cm.tp + cm.tn
        return correct / total if total > 0 else 0

    comparator = PerformanceComparison(metric=accuracy_metric, normalize=False)
    metric_diff = comparator.compare_by_metric(cm1, cm2)
    dist_2d = comparator.compare_by_2d_distance(cm1, cm2)
    dist_3d = comparator.compare_by_3d_distance(cm1, cm2)
    area_triangle = comparator.compare_by_3d_triangle(cm1, cm2)

    print(f"Metric Difference: {metric_diff}")
    print(f"2D Distance: {dist_2d}")
    print(f"3D Distance: {dist_3d}")
    print(f"Area of Triangle: {area_triangle}")

if __name__ == "__main__":
    main()
