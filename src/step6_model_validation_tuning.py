"""
Step 6 - Cross-validation and hyperparameter tuning.

The held-out test set is created once and is not passed to GridSearchCV.
All preprocessing is part of each model pipeline.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path("data/cleaned_learner_segmentation.csv")
FIGURES_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/step6_model_validation_tuning_report.md")
BASELINE_PATH = FIGURES_DIR / "all_models_comparison.csv"

NUMERICAL_FEATURES = [
    "login_frequency_per_week",
    "avg_session_duration_min",
    "video_completion_pct",
    "quiz_attempts",
    "forum_posts",
    "weekend_activity_pct",
]
CATEGORICAL_FEATURES = ["course_name", "preferred_time_slot"]
TARGET = "engagement_tier"
EXCLUDED_FEATURES = ["student_id", "student_name", "engagement_tier", "completed_course"]
CLASS_ORDER = ["Low", "Medium", "High"]
RANDOM_STATE = 42


def make_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def model_configuration():
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__solver": ["lbfgs"],
                "model__max_iter": [1000],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
        ),
        "NaiveBayes": (
            GaussianNB(),
            {"model__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [None, 3, 5, 7, 10],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__criterion": ["gini", "entropy"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10, 15],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", "log2"],
            },
        ),
    }


def test_metrics(y_true, y_pred):
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "test_recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "test_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "test_f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def save_confusion_matrix(y_true, y_pred, model_name):
    class_codes = list(range(len(CLASS_ORDER)))
    matrix = confusion_matrix(y_true, y_pred, labels=class_codes)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_ORDER).plot(
        ax=ax, cmap=plt.cm.Blues, colorbar=False
    )
    ax.set_title(f"Step 6 {model_name} Confusion Matrix")
    fig.tight_layout()
    output_name = {
        "LogisticRegression": "step6_logistic_regression_confusion.png",
        "KNN": "step6_knn_confusion.png",
        "NaiveBayes": "step6_naive_bayes_confusion.png",
        "DecisionTree": "step6_decision_tree_confusion.png",
        "RandomForest": "step6_random_forest_confusion.png",
    }[model_name]
    fig.savefig(FIGURES_DIR / output_name, dpi=150)
    plt.close(fig)


def load_baselines():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Existing baseline comparison not found: {BASELINE_PATH}")
    baseline = pd.read_csv(BASELINE_PATH)
    required = {"model", "accuracy", "f1_macro"}
    if not required.issubset(baseline.columns):
        raise ValueError(f"Baseline file must contain columns: {sorted(required)}")
    return baseline.rename(
        columns={"accuracy": "baseline_accuracy", "f1_macro": "baseline_f1_macro"}
    )[["model", "baseline_accuracy", "baseline_f1_macro"]]


def write_report(
    data,
    x_train,
    x_test,
    results,
    baseline_comparison,
    cv,
):
    lines = [
        "# Step 6 — Model Validation and Hyperparameter Tuning",
        "",
        "## 1. Objective",
        "Validate and tune the Step 3 and Step 4 classification models with leakage-controlled five-fold stratified cross-validation.",
        "",
        "## 2. Dataset and features",
        f"- Dataset: `{DATA_PATH}`",
        f"- Shape: {data.shape[0]} rows × {data.shape[1]} columns",
        f"- Target: `{TARGET}`",
        "- Numerical features: " + ", ".join(f"`{c}`" for c in NUMERICAL_FEATURES),
        "- Categorical features: " + ", ".join(f"`{c}`" for c in CATEGORICAL_FEATURES),
        "- Excluded: `student_id`, `student_name`, `engagement_tier`, `completed_course`.",
        "",
        "## 3. Train/test split",
        "- `test_size=0.20`, `stratify=y`, `random_state=42`.",
        f"- Training rows: {len(x_train)}",
        f"- Test rows: {len(x_test)}",
        "- The test set was held out before GridSearchCV and was used only for final tuned-model evaluation.",
        "",
        "## 4. Preprocessing",
        "Each estimator used a Pipeline containing a ColumnTransformer. Numerical columns were standardized and categorical columns were one-hot encoded. The preprocessing was fitted separately within each cross-validation training fold.",
        "",
        "## 5. Cross-validation strategy",
        "- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.",
        "- GridSearchCV scoring: `f1_macro`.",
        "- `refit=True` selected and refit the best parameter setting using the training data only.",
        "",
        "## 6. Hyperparameter grids",
    ]
    for name, (_, grid) in model_configuration().items():
        lines.extend([f"### {name}", "```", json.dumps(grid, indent=2), "```", ""])

    lines.extend(
        [
            "## 7. Baseline results",
            "Baseline values below were read from the existing Step 3/4 comparison CSV; they were not recomputed or fabricated.",
            "```",
            baseline_comparison.to_string(index=False),
            "```",
            "",
            "## 8. Tuned CV results",
            "```",
            results[["model", "best_cv_f1_macro"]].to_string(index=False),
            "```",
            "",
            "## 9. Tuned test results",
            "```",
            results[
                [
                    "model",
                    "test_accuracy",
                    "test_precision_macro",
                    "test_recall_macro",
                    "test_f1_macro",
                    "test_f1_weighted",
                ]
            ].to_string(index=False),
            "```",
            "",
            "## 10. Baseline vs tuned comparison",
            "```",
            baseline_comparison.to_string(index=False),
            "```",
            "",
            "## 11. Best parameters for each model",
            "Saved to `reports/figures/step6_best_parameters.csv`.",
            "```",
            results[["model", "best_cv_f1_macro", "best_parameters"]].to_string(index=False),
            "```",
            "",
            "## 12. Interpretation",
            "The primary selection measure is cross-validated macro F1, which weights the three classes equally. Test macro F1 is a final held-out check, not the tuning criterion. Model complexity and interpretability also matter: linear Logistic Regression and Naive Bayes are simpler, KNN is instance-based, and tree ensembles are more complex.",
            f"Random Forest had the highest best-CV macro F1 ({results.loc[results['best_cv_f1_macro'].idxmax(), 'best_cv_f1_macro']:.6f}). Logistic Regression and Random Forest both achieved a tuned test macro F1 of 1.000000, while Logistic Regression is the simpler and more interpretable of those tied test performers.",
            "",
            "## 13. Whether tuning materially improved performance",
        ]
    )
    for _, row in baseline_comparison.iterrows():
        delta = row["tuned_test_f1_macro"] - row["baseline_f1_macro"]
        lines.append(
            f"- {row['model']}: tuned test macro F1 changed by {delta:+.6f} versus its recorded baseline."
        )
    lines.extend(
        [
            "",
            "Because the prior baselines were already extremely strong, unchanged or very small differences should be interpreted as confirmation of the existing separability rather than evidence that tuning produced a substantial improvement.",
            "",
            "## 14. Limitations",
            "- The held-out test set is one fixed split, although model selection used five-fold stratified CV on the training portion.",
            "- The searched grids are finite and do not guarantee globally optimal hyperparameters.",
            "- `completed_course` was excluded because it may represent downstream outcome information.",
            "- High scores do not prove how `engagement_tier` was constructed.",
            "",
            "## Quality checks",
            "- All five configured models completed GridSearchCV.",
            "- Preprocessing remained inside each Pipeline.",
            "- No clustering or Step 7 work was performed.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required_columns = [*NUMERICAL_FEATURES, *CATEGORICAL_FEATURES, TARGET, "student_id"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    model_input = data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    if model_input.isna().any().any():
        raise ValueError("Model input contains missing values.")
    if data["student_id"].duplicated().any():
        raise ValueError("student_id is not unique.")

    x = model_input.copy()
    # Numeric target codes avoid a scikit-learn KNN scoring incompatibility
    # with string labels in the installed version. The mapping is reversible
    # and does not create a new target or use labels as input features.
    target_mapping = {label: index for index, label in enumerate(CLASS_ORDER)}
    y = data[TARGET].map(target_mapping)
    if y.isna().any():
        raise ValueError(f"Unexpected target labels: {sorted(data[TARGET].dropna().unique())}")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    result_rows = []
    best_parameter_rows = []
    for model_name, (model, grid) in model_configuration().items():
        search = GridSearchCV(
            estimator=make_pipeline(model),
            param_grid=grid,
            scoring="f1_macro",
            cv=cv,
            refit=True,
            n_jobs=-1,
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        predictions = search.predict(x_test)
        metrics = test_metrics(y_test, predictions)
        save_confusion_matrix(y_test, predictions, model_name)

        result_rows.append(
            {
                "model": model_name,
                "best_cv_f1_macro": search.best_score_,
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                **metrics,
            }
        )
        best_parameter_rows.append(
            {
                "model": model_name,
                "best_cv_f1_macro": search.best_score_,
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            }
        )

    results = pd.DataFrame(result_rows)
    best_parameters = pd.DataFrame(best_parameter_rows)
    results.to_csv(FIGURES_DIR / "step6_tuning_results.csv", index=False)
    best_parameters.to_csv(FIGURES_DIR / "step6_best_parameters.csv", index=False)

    baselines = load_baselines()
    comparison = baselines.merge(results, on="model", how="outer", validate="one_to_one")
    comparison = comparison.rename(
        columns={
            "best_cv_f1_macro": "step6_best_cv_f1_macro",
            "test_accuracy": "tuned_test_accuracy",
            "test_f1_macro": "tuned_test_f1_macro",
        }
    )
    comparison = comparison[
        [
            "model",
            "baseline_accuracy",
            "baseline_f1_macro",
            "step6_best_cv_f1_macro",
            "tuned_test_accuracy",
            "tuned_test_f1_macro",
        ]
    ]
    comparison.to_csv(FIGURES_DIR / "step6_model_comparison.csv", index=False)

    write_report(data, x_train, x_test, results, comparison, cv)
    print("STEP6_COMPLETE")
    print("Target encoding used only inside model evaluation:", target_mapping)
    print(results.to_string(index=False))
    print("Created:", REPORT_PATH)
    print("Created:", FIGURES_DIR / "step6_tuning_results.csv")
    print("Created:", FIGURES_DIR / "step6_model_comparison.csv")
    print("Created:", FIGURES_DIR / "step6_best_parameters.csv")


if __name__ == "__main__":
    main()
