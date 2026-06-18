from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# First existing path wins (same folder as the scripts).
DEFAULT_DATA_CANDIDATES = [
    PROJECT_ROOT / "Data.xlsx",
    PROJECT_ROOT / "data" / "Data.xlsx",
    Path(r"C:\Users\alrhe\Downloads\Telegram Desktop\Data.xlsx"),
]

TARGET_COLUMN = "Ece mean"
ID_COLUMNS = ("ID1", "Profile_ID")

LABEL_MODES = {
    "threshold": "Numeric: value ≥ threshold → class 1",
    "direct": "Column is already 0/1 (or two unique values)",
}

MULTI_TARGET_COMBINE = {
    "any": "Class 1 if ANY target column meets the rule",
    "all": "Class 1 if ALL target columns meet the rule",
}

RANK_METRICS = {
    "recall_mean": "Recall (detect positives)",
    "f1_mean": "F1 score",
    "precision_mean": "Precision",
    "balanced_accuracy_mean": "Balanced accuracy",
    "roc_auc_mean": "ROC-AUC",
    "accuracy_mean": "Accuracy",
}

BEST_RESULT_TITLES = {
    "recall_mean": "Best recall (saline)",
    "f1_mean": "Best F1 score",
    "precision_mean": "Best precision",
    "balanced_accuracy_mean": "Best balanced accuracy",
    "roc_auc_mean": "Best ROC-AUC",
    "accuracy_mean": "Best accuracy",
}

DEFAULT_SALINITY_THRESHOLD = 1.5
DEFAULT_TEST_SIZE = 0.25
DEFAULT_RUNS = 30
DEFAULT_DECISION_THRESHOLD = 0.5

ALGORITHM_CHOICES = {
    "random_forest": "Random Forest (balanced, good default)",
    "extra_trees": "Extra Trees (fast ensemble)",
    "gradient_boosting": "Gradient Boosting (strong but can overfit small data)",
    "logistic_regression": "Logistic Regression (linear baseline)",
    "svm_rbf": "SVM RBF (works on few samples with scaling)",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Gaussian Naive Bayes (very small data)",
}

SAMPLING_CHOICES = {
    "none": "No resampling (use class weights where supported)",
    "smote": "SMOTE oversampling",
    "adasyn": "ADASYN (focuses on harder minority points)",
}

EVALUATION_MODES = {
    "holdout": "Repeated random hold-out validation",
    "stratified_cv": "Stratified k-fold cross-validation each run",
}

TUNE_METRICS = ("recall", "f1")
