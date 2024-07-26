# contingency-space
contingency-space


# Model Performance Space Module

How will it look in Scikit Learn Repo
# Project Structure

<img width="912" alt="Screenshot 2024-07-04 at 1 32 15 PM" src="https://github.com/siddu1324/contingency-space/assets/126925412/61f10247-c744-422c-8136-13969fb08d34">


# Module Descriptions
## model_performance_space.py

CMGenerator: Generates a grid of confusion matrices by varying the true positive rate (TPR) and true negative rate (TNR).

## performance_comparison.py

PerformanceComparison: A class that offers various methods to compare models, such as calculating 2D and 3D distances between confusion matrices and more geometrically complex comparisons.

## learning_path.py

LearningPath: Tracks and evaluates changes in model performance over a sequence of confusion matrices.

# The users can import them as needed as

from sklearn.metrics.confusion_matrix_space import CMGenerator
from sklearn.metrics.performance_comparison import PerformanceComparison
from sklearn.metrics.learning_path import LearningPath


I have started to work on the code and the integration and i am confident that we will some major progress by this weekend. The code in this repo is just the base mimic version of that in the bitbucket with minor changes.

## 1. CMGenerator Class

### Purpose
Generates a series of confusion matrices based on varying thresholds of classification metrics such as True Positive Rate (TPR) and True Negative Rate (TNR).

### User Input
- `n_p`: Number of positive samples.
- `n_n`: Number of negative samples.
- `n_cm`: Number of confusion matrices to generate.

### User Output
A list of `ConfusionMatrix` objects, each representing a point in the performance space.

### Example Usage

`generator = CMGenerator(n_p=1000, n_n=1000, n_cm=10)`
`matrices = generator.generate_matrices()`
`for cm in matrices:`
    `print(cm)`

## 2. PerformanceComparison Class

### Purpose
Compares two models using predefined metrics.

### User Input
- Two instances of `ConfusionMatrix`.
- A metric function that calculates a specific performance metric from a confusion matrix.

### User Output
A numerical value representing the difference in the metric between the two models.

### Example Usage

`from sklearn.metrics import accuracy_score  # Assuming a predefined accuracy calculation`

`def accuracy_metric(cm):`
    `return accuracy_score(cm.tp, cm.fn, cm.tn, cm.fp)`

`comparison = PerformanceComparison(metric=accuracy_metric)`
`difference = comparison.compare_by_metric(cm1, cm2)`
`print(f"Difference in Accuracy: {difference}")`


## LearningPath Class

### Purpose
Evaluates a sequence of model configurations to analyze the impact of incremental changes.

### User Input
- A list of `ConfusionMatrix` objects representing a sequence of model states.
- A metric function to assess model performance.

### User Output
A dictionary containing various statistics and metrics describing the evolution of model performance over the learning path.

### Example Usage

`from sklearn.metrics import accuracy_score  # Assuming a predefined accuracy calculation function`

`def accuracy_metric(cm):`
    `return accuracy_score(cm.tp, cm.fn, cm.tn, cm.fp)  # Hypothetical function usage`

`path = [cm1, cm2, cm3, cm4]  # List of ConfusionMatrix instances`
`learning_path = LearningPath(path=path, metric=accuracy_metric)`
`results = learning_path.evaluate_path()`
`print("Evaluation Results:", results)`


# Compute Imbalance Sensitivity

Evaluates how sensitive a metric is to changes in the imbalance ratio.

### Method: compute_imbalance_sensitivity(metric, imbalance_ratios)

# Input:

metric (function): The metric function to evaluate.
imbalance_ratios (list of float | float): The imbalance ratios to test.

## Output:
float | list of float: Sensitivity scores for each imbalance ratio.

# Generate Contingency Space
Generates a space of metric evaluations across varying conditions.

## Method: `generate_contingency_space(metric, imbalance_ratio=1)`

## Input:

metric (function): The metric function to use for generating the space.

imbalance_ratio (float): The specific imbalance ratio to use.

Output:

np.ndarray: A matrix representing the metric evaluations across different conditions.


import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import CMGenerator, ContingencySpace

# Generate some sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Assume we have a model that outputs probabilities
model = SomeTrainedModel()
y_probs = model.predict_proba(X_test)

# Initialize the CMGenerator
cm_generator = CMGenerator(y_test, y_probs[:, 1])

# Define metrics to evaluate
metrics = {
    'Accuracy': lambda cm: np.trace(cm) / np.sum(cm),
    'Sensitivity': lambda cm: cm[1, 1] / np.sum(cm[1, :])
}

# Initialize the Contingency Space with the generator and metrics
contingency_space = ContingencySpace(cm_generator, metrics)

# Generate and print the contingency space analysis
results = contingency_space.generate_contingency_space(normalize='true')
print("Contingency Space Results:", results)

# Compute imbalance sensitivity
imbalance_sensitivity = contingency_space.compute_imbalance_sensitivity(metrics['Sensitivity'], 0.5)
print("Imbalance Sensitivity:", imbalance_sensitivity)


cms = self.cm_generator.generate_cms()
        results = {metric_name: [] for metric_name in self.metrics}
        for cm in cms:
            for metric_name, metric_func in self.metrics.items():
                results[metric_name].append(metric_func(cm))
        return results


        def compute_metric_values(self, cms, metric):
        """
        Calculate metric values for all confusion matrices using the given metric function.
        """
        return [metric(cm) for cm in cms]