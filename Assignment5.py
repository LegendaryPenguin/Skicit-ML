
"""
Program Name: EECS 658 Assignment 5
Description:
    This program loads the imbalanced iris dataset from "imbalanced iris.csv"
    and evaluates a Neural Network classifier with 2-fold cross-validation for:
    1) The original imbalanced dataset
    2) Oversampling methods (RandomOverSampler, SMOTE, ADASYN)
    3) Undersampling methods (RandomUnderSampler, ClusterCentroids, TomekLinks)

    For Part 1, the program prints:
        - Confusion Matrix
        - Accuracy
        - Class Balanced Accuracy
        - Balanced Accuracy
        - scikit-learn balanced_accuracy_score

    For Parts 2 and 3, the program prints:
        - Confusion Matrix
        - Accuracy

Inputs:
    - imbalanced iris.csv (must be in the same folder as this Python file, or the
      DATA_FILE path can be updated below)

Outputs:
    - Printed results for all three assignment parts
    - Confusion matrices and requested scores

Model:
    - Neural Network classifier implemented with scikit-learn's MLPClassifier
    - 2-fold Stratified Cross-Validation

Collaborators:
    - None

Other Sources for Code / Ideas:
    - Course materials / lecture instructions
    - scikit-learn documentation
    - imbalanced-learn documentation
    - ChatGPT

Author:
    - Surender

Creation Date:
    - 2026-03-16
"""

# Import os so we can build a reliable path to the CSV file.
import os

# Import warnings so we can hide convergence warnings from the neural network if they appear.
import warnings

# Import numpy for numerical work.
import numpy as np

# Import pandas to read the CSV file.
import pandas as pd

# Import the ConvergenceWarning class so it can be filtered out cleanly.
from sklearn.exceptions import ConvergenceWarning

# Import confusion_matrix to build the confusion matrix for each experiment.
from sklearn.metrics import confusion_matrix

# Import accuracy_score to compute the standard accuracy.
from sklearn.metrics import accuracy_score

# Import balanced_accuracy_score to compute scikit-learn's balanced accuracy.
from sklearn.metrics import balanced_accuracy_score

# Import StratifiedKFold so the 2 folds preserve the class proportions.
from sklearn.model_selection import StratifiedKFold

# Import MLPClassifier as the neural network model requested in the assignment.
from sklearn.neural_network import MLPClassifier

# Import StandardScaler so the neural network trains on scaled features.
from sklearn.preprocessing import StandardScaler

# Import Pipeline from scikit-learn for the non-resampled case.
from sklearn.pipeline import Pipeline as SklearnPipeline

# Import Pipeline from imbalanced-learn for cases that use resampling.
from imblearn.pipeline import Pipeline as ImbPipeline

# Import the oversampling tools requested in Part 2.
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# Import the undersampling tools requested in Part 3.
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks


# Ignore neural-network convergence warnings so the output stays clean for the assignment printout.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Store the CSV file name in one place so it is easy to update later if needed.
DATA_FILE = "imbalanced iris.csv"


# Create a helper function that computes the Class Balanced Accuracy exactly from a confusion matrix.
def calculate_class_balanced_accuracy(cm):
    """
    Class Balanced Accuracy (CBA):
        For each class i:
            precision_i = TP_i / predicted_as_i
            recall_i    = TP_i / actual_i
            cba_i       = min(precision_i, recall_i)

        Final CBA = average of cba_i over all classes
    """

    # Create an empty list to store the CBA value for each class.
    class_cba_values = []

    # Loop through each class index in the confusion matrix.
    for i in range(len(cm)):
        # True positives for class i are on the diagonal.
        tp = cm[i, i]

        # Predicted positives for class i are the sum of column i.
        predicted_as_i = np.sum(cm[:, i])

        # Actual positives for class i are the sum of row i.
        actual_i = np.sum(cm[i, :])

        # Compute precision carefully to avoid division by zero.
        precision_i = tp / predicted_as_i if predicted_as_i != 0 else 0.0

        # Compute recall carefully to avoid division by zero.
        recall_i = tp / actual_i if actual_i != 0 else 0.0

        # Compute the class CBA as the minimum of precision and recall.
        cba_i = min(precision_i, recall_i)

        # Save the class-level CBA.
        class_cba_values.append(cba_i)

    # Return the average CBA across all classes.
    return float(np.mean(class_cba_values))


# Create a helper function that computes Balanced Accuracy from a confusion matrix.
def calculate_balanced_accuracy(cm):
    """
    Balanced Accuracy (Urbanowicz-style multi-class one-vs-rest version):
        For each class i:
            sensitivity_i = TP_i / (TP_i + FN_i)
            specificity_i = TN_i / (TN_i + FP_i)
            ba_i          = (sensitivity_i + specificity_i) / 2

        Final BA = average of ba_i over all classes
    """

    # Compute the total number of samples in the confusion matrix.
    total = np.sum(cm)

    # Create an empty list to store the balanced accuracy value for each class.
    class_ba_values = []

    # Loop through each class index in the confusion matrix.
    for i in range(len(cm)):
        # True positives are the diagonal entry for class i.
        tp = cm[i, i]

        # False negatives are the rest of row i besides the diagonal.
        fn = np.sum(cm[i, :]) - tp

        # False positives are the rest of column i besides the diagonal.
        fp = np.sum(cm[:, i]) - tp

        # True negatives are everything not in tp, fn, or fp.
        tn = total - tp - fn - fp

        # Compute sensitivity (same as recall) carefully.
        sensitivity_i = tp / (tp + fn) if (tp + fn) != 0 else 0.0

        # Compute specificity carefully.
        specificity_i = tn / (tn + fp) if (tn + fp) != 0 else 0.0

        # Compute the balanced accuracy contribution for this class.
        ba_i = (sensitivity_i + specificity_i) / 2.0

        # Save the class-level balanced accuracy.
        class_ba_values.append(ba_i)

    # Return the average across classes.
    return float(np.mean(class_ba_values))


# Create a helper function that builds the neural network model.
def create_neural_network():
    """
    Return a neural network classifier with a fixed random_state for reproducibility.
    """

    # Create and return an MLP classifier.
    return MLPClassifier(
        hidden_layer_sizes=(10,),   # Use one hidden layer with 10 neurons.
        activation="relu",          # Use ReLU activation.
        solver="adam",              # Use the Adam optimizer.
        max_iter=2000,              # Allow enough iterations for training.
        random_state=42             # Fix the random seed so results are repeatable.
    )


# Create a helper function that evaluates a pipeline with 2-fold cross-validation.
def evaluate_with_2fold_cv(X, y, pipeline, labels):
    """
    Train and test the supplied pipeline using 2-fold StratifiedKFold.
    Return:
        - overall confusion matrix across both test folds
        - overall accuracy
        - all true labels
        - all predicted labels
    """

    # Create the 2-fold stratified cross-validator requested in the assignment.
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # Make empty lists to store true labels and predicted labels from both folds.
    all_true = []
    all_pred = []

    # Loop through each train/test split produced by the 2-fold cross-validation.
    for train_index, test_index in skf.split(X, y):
        # Build the training feature set for this fold.
        X_train = X.iloc[train_index]

        # Build the testing feature set for this fold.
        X_test = X.iloc[test_index]

        # Build the training labels for this fold.
        y_train = y.iloc[train_index]

        # Build the testing labels for this fold.
        y_test = y.iloc[test_index]

        # Fit the pipeline using only the training data for this fold.
        pipeline.fit(X_train, y_train)

        # Predict the labels for the test data in this fold.
        y_pred = pipeline.predict(X_test)

        # Save the true labels from this fold.
        all_true.extend(y_test.tolist())

        # Save the predicted labels from this fold.
        all_pred.extend(y_pred.tolist())

    # Convert the combined true labels into a numpy array.
    all_true = np.array(all_true)

    # Convert the combined predicted labels into a numpy array.
    all_pred = np.array(all_pred)

    # Build a single confusion matrix across both folds.
    cm = confusion_matrix(all_true, all_pred, labels=labels)

    # Compute the standard overall accuracy across both folds.
    acc = accuracy_score(all_true, all_pred)

    # Return all requested results.
    return cm, acc, all_true, all_pred


# Create a helper function that prints a section title in a consistent format.
def print_section_title(title):
    # Print a blank line for readability.
    print()

    # Print a separator line.
    print("=" * 70)

    # Print the supplied title.
    print(title)

    # Print another separator line.
    print("=" * 70)


# Create a helper function that prints a confusion matrix neatly with class labels.
def print_confusion_matrix(cm, labels):
    # Print a header line.
    print("Confusion Matrix:")

    # Print the class order so the matrix is easy to interpret.
    print("Class Order:", list(labels))

    # Print the matrix itself.
    print(cm)


# Build the full path to the CSV file using the folder containing this script.
script_folder = os.path.dirname(os.path.abspath(__file__))

# Combine the script folder and the CSV file name.
csv_path = os.path.join(script_folder, DATA_FILE)

# Read the imbalanced iris CSV file into a pandas DataFrame.
df = pd.read_csv(csv_path)

# Separate the feature columns from the target column.
X = df.drop(columns=["class"])

# Store the target labels in y.
y = df["class"]

# Store the class labels in sorted order for consistent confusion matrices.
labels = sorted(y.unique())


# ----------------------------- Part 1 ---------------------------------
# Print the part number exactly as requested in the assignment.
print_section_title("Part 1: Imbalanced Data Set")

# Build a pipeline for scaling followed by the neural network.
part1_pipeline = SklearnPipeline([
    ("scaler", StandardScaler()),
    ("nn", create_neural_network())
])

# Evaluate the original imbalanced dataset with 2-fold cross-validation.
cm1, acc1, y_true1, y_pred1 = evaluate_with_2fold_cv(X, y, part1_pipeline, labels)

# Compute the manual Class Balanced Accuracy from the confusion matrix.
class_bal_acc1 = calculate_class_balanced_accuracy(cm1)

# Compute the manual Balanced Accuracy from the confusion matrix.
balanced_acc1 = calculate_balanced_accuracy(cm1)

# Compute scikit-learn's balanced accuracy from the predictions.
sklearn_bal_acc1 = balanced_accuracy_score(y_true1, y_pred1)

# Print the confusion matrix.
print_confusion_matrix(cm1, labels)

# Print the standard accuracy with a clear label.
print(f"Accuracy: {acc1:.6f}")

# Print the Class Balanced Accuracy with a clear label.
print(f"Class Balanced Accuracy: {class_bal_acc1:.6f}")

# Print the manual Balanced Accuracy with a clear label.
print(f"Balanced Accuracy: {balanced_acc1:.6f}")

# Print scikit-learn's balanced accuracy with a clear label.
print(f"balanced_accuracy_score (scikit-learn): {sklearn_bal_acc1:.6f}")


# ----------------------------- Part 2 ---------------------------------
# Print the part number exactly as requested in the assignment.
print_section_title("Part 2: Oversampling")

# Create a dictionary of the oversampling methods requested in the assignment.
oversamplers = {
    "Random Oversampling": RandomOverSampler(random_state=42),
    "SMOTE Oversampling": SMOTE(random_state=42),
    "ADASYN Oversampling": ADASYN(sampling_strategy="minority", random_state=42)
}

# Loop through each oversampling method.
for method_name, sampler in oversamplers.items():
    # Print a subheading for the current method.
    print_section_title(method_name)

    # Build an imbalanced-learn pipeline: sampler -> scaler -> neural network.
    pipeline = ImbPipeline([
        ("sampler", sampler),
        ("scaler", StandardScaler()),
        ("nn", create_neural_network())
    ])

    # Evaluate the current oversampling method with 2-fold cross-validation.
    cm, acc, _, _ = evaluate_with_2fold_cv(X, y, pipeline, labels)

    # Print the confusion matrix for this method.
    print_confusion_matrix(cm, labels)

    # Print the accuracy for this method.
    print(f"Accuracy: {acc:.6f}")


# ----------------------------- Part 3 ---------------------------------
# Print the part number exactly as requested in the assignment.
print_section_title("Part 3: Undersampling")

# Create a dictionary of the undersampling methods requested in the assignment.
undersamplers = {
    "Random Undersampling": RandomUnderSampler(random_state=42),
    "Cluster Centroids Undersampling": ClusterCentroids(random_state=42),
    "Tomek Links Undersampling": TomekLinks()
}

# Loop through each undersampling method.
for method_name, sampler in undersamplers.items():
    # Print a subheading for the current method.
    print_section_title(method_name)

    # Build an imbalanced-learn pipeline: sampler -> scaler -> neural network.
    pipeline = ImbPipeline([
        ("sampler", sampler),
        ("scaler", StandardScaler()),
        ("nn", create_neural_network())
    ])

    # Evaluate the current undersampling method with 2-fold cross-validation.
    cm, acc, _, _ = evaluate_with_2fold_cv(X, y, pipeline, labels)

    # Print the confusion matrix for this method.
    print_confusion_matrix(cm, labels)

    # Print the accuracy for this method.
    print(f"Accuracy: {acc:.6f}")
