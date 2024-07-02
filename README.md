# contingency-space
contingency-space


# Model Performance Space Module

How will it look in Scikit Learn Repo
# Project Structure

<img width="924" alt="Screenshot 2024-07-01 at 7 59 07 PM" src="https://github.com/siddu1324/contingency-space/assets/126925412/a31c91ac-2f52-452e-b981-f8e5d4764fec">


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
