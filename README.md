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
```python
generator = CMGenerator(n_p=1000, n_n=1000, n_cm=10)
matrices = generator.generate_matrices()
for cm in matrices:
    print(cm)

## 2. PerformanceComparison Class

### Purpose
Compares two models using predefined metrics.

### User Input
- Two instances of `ConfusionMatrix`.
- A metric function that calculates a specific performance metric from a confusion matrix.

### User Output
A numerical value representing the difference in the metric between the two models.

### Example Usage
```python
from sklearn.metrics import accuracy_score  # Assuming a predefined accuracy calculation

def accuracy_metric(cm):
    return accuracy_score(cm.tp, cm.fn, cm.tn, cm.fp)

comparison = PerformanceComparison(metric=accuracy_metric)
difference = comparison.compare_by_metric(cm1, cm2)
print(f"Difference in Accuracy: {difference}")


## LearningPath Class

### Purpose
Evaluates a sequence of model configurations to analyze the impact of incremental changes.

### User Input
- A list of `ConfusionMatrix` objects representing a sequence of model states.
- A metric function to assess model performance.

### User Output
A dictionary containing various statistics and metrics describing the evolution of model performance over the learning path.

### Example Usage
```python
from sklearn.metrics import accuracy_score  # Assuming a predefined accuracy calculation function

def accuracy_metric(cm):
    return accuracy_score(cm.tp, cm.fn, cm.tn, cm.fp)  # Hypothetical function usage

path = [cm1, cm2, cm3, cm4]  # List of ConfusionMatrix instances
learning_path = LearningPath(path=path, metric=accuracy_metric)
results = learning_path.evaluate_path()
print("Evaluation Results:", results)
