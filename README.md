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
