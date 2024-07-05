# contingency-space
contingency-space


# Model Performance Space Module

How will it look in Scikit Learn Repo
# Project Structure

sklearn/
│
├── metrics/
│   ├── __init__.py
│   ├── model_performance_space.py  # Handles the generation and manipulation of confusion matrices i.e CMgenerator
│   ├── performance_comparison.py  # Provides tools to compare models based on confusion matrices 
│   ├── learning_path.py           # Analyzes the progression or learning paths of models
│   └── tests/                     # Unit tests for the new modules
│       ├── test_model_performance_space.py
│       ├── test_performance_comparison.py
│       └── test_learning_path.py
└── 


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
