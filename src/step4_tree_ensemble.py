"""
Step 4 - Decision Tree and Random Forest baseline script
Reads: data/cleaned_learner_segmentation.csv
Writes: reports/step4_tree_ensemble_report.md
Produces:
  - reports/figures/decision_tree_confusion.png
  - reports/figures/random_forest_confusion.png
  - reports/figures/all_models_comparison.csv
  - reports/figures/random_forest_feature_importance.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


CLEANED_DATA_PATH = Path("data/cleaned_learner_segmentation.csv")
STEP3_REPORT_PATH = Path("reports/step3_baseline_classification_report.md")
STEP4_REPORT_PATH = Path("reports/step4_tree_ensemble_report.md")
FIGURES_DIR = Path("reports/figures")

TARGET_COLUMN = "engagement_tier"
EXCLUDED_FEATURES = ["student_id", "student_name", "engagement_tier", "completed_course"]
CLASS_ORDER = ["Low", "Medium", "High"]
RANDOM_STATE = 42


def parse_step3_metrics(report_path: Path) -> pd.DataFrame:
    if not report_path.exists():
        raise FileNotFoundError(f"Step 3 report not found at: {report_path}")

    lines = report_path.read_text(encoding="utf-8").splitlines()
    header_index = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("model") and "accuracy" in line and "f1_macro" in line:
            header_index = idx
            break

    if header_index is None:
        raise ValueError("Could not locate Step 3 metrics table in report.")

    step3_rows = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("##"):
            break
        parts = stripped.split()
        if len(parts) < 6:
            continue
        model_name = parts[0]
        step3_rows.append(
            {
                "model": model_name,
                "accuracy": float(parts[1]),
                "precision_macro": float(parts[2]),
                "recall_macro": float(parts[3]),
                "f1_macro": float(parts[4]),
                "f1_weighted": float(parts[5]),
            }
        )

    if not step3_rows:
        raise ValueError("Step 3 metrics rows were not parsed from report.")

    return pd.DataFrame(step3_rows)


def make_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def evaluate_model(name, pipeline, x_train, x_test, y_train, y_test):
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    return metrics, cm, report_dict, report_text


def save_confusion_matrix(cm, title, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_random_forest_feature_importance(fitted_pipeline, numeric_cols):
    pre = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]

    feature_names = pre.get_feature_names_out()
    importance = model.feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )
    fi.reset_index(drop=True, inplace=True)
    return fi


def write_step4_report(
    out_path: Path,
    dataset_shape,
    target_distribution,
    feature_columns,
    numeric_cols,
    categorical_cols,
    train_size,
    test_size,
    dt_metrics,
    rf_metrics,
    dt_report_text,
    rf_report_text,
    all_models_df,
    rf_importance_df,
):
    lines = []
    lines.append("# Step 4 — Trees and Ensembles Report\n")
    lines.append("## 1. Objective")
    lines.append(
        "Build and evaluate Decision Tree and Random Forest classifiers for engagement tier prediction, "
        "then compare them with Step 3 baseline models.\n"
    )

    lines.append("## 2. Dataset used")
    lines.append("- Source: `data/cleaned_learner_segmentation.csv`")
    lines.append(f"- Shape: {dataset_shape[0]} rows × {dataset_shape[1]} columns")
    lines.append("- Target: `engagement_tier`")
    lines.append("- Target distribution:")
    for label, count in target_distribution.items():
        lines.append(f"  - {label}: {count}")
    lines.append("")

    lines.append("## 3. Features used")
    lines.append("- Numerical:")
    for c in numeric_cols:
        lines.append(f"  - {c}")
    lines.append("- Categorical:")
    for c in categorical_cols:
        lines.append(f"  - {c}")
    lines.append("")

    lines.append("## 4. Features excluded and why")
    lines.append("- `student_id`: identifier, not a behavior feature")
    lines.append("- `student_name`: identifier, not a behavior feature")
    lines.append("- `engagement_tier`: target variable")
    lines.append("- `completed_course`: downstream outcome; excluded to avoid leakage risk\n")

    lines.append("## 5. Train/test split")
    lines.append("- `test_size=0.20`")
    lines.append("- `stratify=y`")
    lines.append("- `random_state=42`")
    lines.append(f"- Train size: {train_size}")
    lines.append(f"- Test size: {test_size}\n")

    lines.append("## 6. Preprocessing approach")
    lines.append(
        "Used `ColumnTransformer` inside sklearn `Pipeline` for each model. "
        "Numerical features were scaled with `StandardScaler`; categorical features encoded with `OneHotEncoder`.\n"
    )

    lines.append("## 7. Decision Tree methodology")
    lines.append("- Model: `DecisionTreeClassifier(random_state=42)`\n")

    lines.append("## 8. Random Forest methodology")
    lines.append("- Model: `RandomForestClassifier(random_state=42)`\n")

    lines.append("## 9. Evaluation metrics")
    lines.append("### Decision Tree")
    for k, v in dt_metrics.items():
        if k == "model":
            continue
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("### Random Forest")
    for k, v in rf_metrics.items():
        if k == "model":
            continue
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")

    lines.append("## 10. Confusion matrices")
    lines.append("- Decision Tree: `reports/figures/decision_tree_confusion.png`")
    lines.append("- Random Forest: `reports/figures/random_forest_confusion.png`\n")

    lines.append("## 11. Comparison of all five models")
    lines.append("Saved to: `reports/figures/all_models_comparison.csv`\n")
    lines.append("```")
    lines.append(all_models_df.to_string(index=False))
    lines.append("```\n")

    lines.append("## 12. Random Forest feature importance")
    lines.append("Saved to: `reports/figures/random_forest_feature_importance.csv`\n")
    lines.append("Top 10 features:")
    lines.append("```")
    lines.append(rf_importance_df.head(10).to_string(index=False))
    lines.append("```\n")

    lines.append("## 13. Interpretation of results")
    lines.append(
        "If tree-based models are also very high-performing, that is consistent with strong separability in the current feature set. "
        "This does not by itself prove how the target labels were generated.\n"
    )

    lines.append("## 14. Limitations / warnings")
    lines.append("- Evaluation is based on one train/test split for Step 4 baseline comparison.")
    lines.append("- No hyperparameter tuning was performed in this step by design.")
    lines.append("- Results should be interpreted with prior robustness audit context.\n")

    lines.append("## 15. Conclusion")
    lines.append(
        "Decision Tree and Random Forest baselines were evaluated with the same data split and feature policy as Step 3, "
        "and compared directly with Logistic Regression, KNN, and Naive Bayes."
    )
    lines.append("")
    lines.append("### Class-level report — Decision Tree")
    lines.append("```")
    lines.append(dt_report_text)
    lines.append("```")
    lines.append("")
    lines.append("### Class-level report — Random Forest")
    lines.append("```")
    lines.append(rf_report_text)
    lines.append("```")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {CLEANED_DATA_PATH}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEANED_DATA_PATH)

    x = df.drop(columns=EXCLUDED_FEATURES)
    y = df[TARGET_COLUMN]

    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)

    dt_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )

    dt_metrics, dt_cm, dt_report_dict, dt_report_text = evaluate_model(
        "DecisionTree", dt_pipeline, x_train, x_test, y_train, y_test
    )
    rf_metrics, rf_cm, rf_report_dict, rf_report_text = evaluate_model(
        "RandomForest", rf_pipeline, x_train, x_test, y_train, y_test
    )

    save_confusion_matrix(
        dt_cm, "Decision Tree Confusion Matrix", FIGURES_DIR / "decision_tree_confusion.png"
    )
    save_confusion_matrix(
        rf_cm, "Random Forest Confusion Matrix", FIGURES_DIR / "random_forest_confusion.png"
    )

    step3_metrics_df = parse_step3_metrics(STEP3_REPORT_PATH)
    step4_metrics_df = pd.DataFrame([dt_metrics, rf_metrics])
    all_models_df = pd.concat([step3_metrics_df, step4_metrics_df], ignore_index=True)
    all_models_df = all_models_df[
        ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    ]
    all_models_df.to_csv(FIGURES_DIR / "all_models_comparison.csv", index=False)

    rf_importance_df = compute_random_forest_feature_importance(rf_pipeline, numeric_cols)
    rf_importance_df.to_csv(FIGURES_DIR / "random_forest_feature_importance.csv", index=False)

    write_step4_report(
        out_path=STEP4_REPORT_PATH,
        dataset_shape=df.shape,
        target_distribution=y.value_counts().to_dict(),
        feature_columns=x.columns.tolist(),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        train_size=len(x_train),
        test_size=len(x_test),
        dt_metrics=dt_metrics,
        rf_metrics=rf_metrics,
        dt_report_text=dt_report_text,
        rf_report_text=rf_report_text,
        all_models_df=all_models_df,
        rf_importance_df=rf_importance_df,
    )

    print("STEP4_COMPLETE")
    print("DecisionTree metrics:", dt_metrics)
    print("RandomForest metrics:", rf_metrics)
    print("Created:", str(STEP4_REPORT_PATH))
    print("Created:", str(FIGURES_DIR / "decision_tree_confusion.png"))
    print("Created:", str(FIGURES_DIR / "random_forest_confusion.png"))
    print("Created:", str(FIGURES_DIR / "all_models_comparison.csv"))
    print("Created:", str(FIGURES_DIR / "random_forest_feature_importance.csv"))


if __name__ == "__main__":
    main()
