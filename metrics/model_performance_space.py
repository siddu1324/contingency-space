import numpy as np

class ConfusionMatrix:
    """Class to handle basic operations on confusion matrices."""
    def __init__(self, tp, fn, tn, fp):
        """
        Initialize the ConfusionMatrix with counts for true positives, false negatives, true negatives, and false positives.

        Parameters:
        tp (int): True Positives count.
        fn (int): False Negatives count.
        tn (int): True Negatives count.
        fp (int): False Positives count.
        """
        self.tp = tp
        self.fn = fn
        self.tn = tn
        self.fp = fp

    def __str__(self):
        """Return string representation of the confusion matrix counts."""
        return f"TP: {self.tp}, FN: {self.fn}, TN: {self.tn}, FP: {self.fp}"

class CMGenerator:
    """
    Generates multiple confusion matrices over a range of True Positive Rate (TPR) and True Negative Rate (TNR) values.

    This class is useful for generating a performance space where each point represents a confusion matrix corresponding
    to a specific combination of TPR and TNR.
    """
    def __init__(self, n_p, n_n, n_cm):
        """
        Initializes the CMGenerator with the number of positives, negatives, and the number of matrices to generate.

        Args:
        n_p (int): Total number of positives.
        n_n (int): Total number of negatives.
        n_cm (int): Number of points to generate in the performance space.
        """
        self.n_cm = n_cm
        self.n_p = n_p
        self.n_n = n_n
        self.all_cms = []

    def generate_cms(self):
        """
        Generates confusion matrices based on linear spacing between TPR and TNR.
        This function populates the all_cms list with instances of ConfusionMatrix.
        """
        all_tpr = np.linspace(0, 1.0, self.n_cm)
        all_tnr = np.linspace(0, 1.0, self.n_cm)
        for tpr in all_tpr:
            for tnr in all_tnr:
                tp = int(self.n_p * tpr)
                fn = self.n_p - tp
                tn = int(self.n_n * tnr)
                fp = self.n_n - tn
                self.all_cms.append(ConfusionMatrix(tp, fn, tn, fp))

    def show_all_cms(self):
        """Prints all generated confusion matrices."""
        for idx, cm in enumerate(self.all_cms):
            print(f'--[{idx}]-----------------------------------------')
            print(cm)

if __name__ == "__main__":
    # Example usage demonstrating how to generate and display the confusion matrices
    p, n = 2500, 2500  # Total positives and negatives
    n_cm = 6  # Number of points to generate in the performance space
    generator = CMGenerator(n_p=p, n_n=n, n_cm=n_cm)
    generator.generate_cms()
    generator.show_all_cms()
