"""
Step 5 - Unsupervised learner segmentation with K-Means.

The six numerical behavioral features are scaled before clustering.
engagement_tier is retained only for post-clustering interpretation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/cleaned_learner_segmentation.csv")
FIGURES_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/step5_clustering_report.md")

BEHAVIOR_FEATURES = [
    "login_frequency_per_week",
    "avg_session_duration_min",
    "video_completion_pct",
    "quiz_attempts",
    "forum_posts",
    "weekend_activity_pct",
]
TARGET_COLUMN = "engagement_tier"
ID_COLUMN = "student_id"
CANDIDATE_K = [2, 3, 4, 5, 6]
RANDOM_STATE = 42


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_table(frame):
    return frame.to_string(index=True, float_format=lambda value: f"{value:.3f}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required dataset does not exist: {DATA_PATH}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)

    missing_columns = [
        column
        for column in [ID_COLUMN, TARGET_COLUMN, *BEHAVIOR_FEATURES]
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if len(data) != 400:
        raise ValueError(f"Expected exactly 400 cleaned learners, found {len(data)}.")
    if data[ID_COLUMN].duplicated().any():
        raise ValueError("student_id is not unique in the cleaned dataset.")
    if data[BEHAVIOR_FEATURES].isna().any().any():
        raise ValueError("Clustering features contain missing values.")

    behavior_data = data[BEHAVIOR_FEATURES].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(behavior_data)

    evaluation_rows = []
    fitted_models = {}
    for k in CANDIDATE_K:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(scaled_features)
        evaluation_rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(scaled_features, labels),
            }
        )
        fitted_models[k] = model

    evaluation = pd.DataFrame(evaluation_rows)
    # Silhouette is the primary quantitative criterion; the smaller K breaks ties
    # because it is easier to interpret and avoids unnecessary fragmentation.
    selected_row = evaluation.sort_values(
        ["silhouette_score", "k"], ascending=[False, True]
    ).iloc[0]
    selected_k = int(selected_row["k"])
    final_model = fitted_models[selected_k]
    final_labels = final_model.labels_

    assignments = data[[ID_COLUMN, TARGET_COLUMN, *BEHAVIOR_FEATURES]].copy()
    assignments.insert(1, "cluster", final_labels)
    assignments.to_csv(FIGURES_DIR / "cluster_assignments.csv", index=False)

    cluster_sizes = (
        assignments.groupby("cluster")
        .size()
        .rename("learner_count")
        .to_frame()
        .sort_index()
    )
    cluster_sizes["percentage"] = cluster_sizes["learner_count"] / len(assignments) * 100

    profiles = (
        assignments.groupby("cluster")[BEHAVIOR_FEATURES]
        .mean()
        .sort_index()
    )
    profiles.to_csv(FIGURES_DIR / "cluster_profiles.csv")

    crosstab = pd.crosstab(assignments["cluster"], assignments[TARGET_COLUMN]).reindex(
        columns=["Low", "Medium", "High"], fill_value=0
    )
    crosstab.to_csv(FIGURES_DIR / "cluster_engagement_crosstab.csv")
    crosstab_percentage = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

    evaluation.to_csv(FIGURES_DIR / "kmeans_evaluation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(evaluation["k"], evaluation["inertia"], marker="o")
    ax.set_title("K-Means Inertia by Number of Clusters")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(CANDIDATE_K)
    save_plot(fig, FIGURES_DIR / "kmeans_elbow.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(evaluation["k"], evaluation["silhouette_score"], marker="o")
    ax.set_title("K-Means Silhouette Score by K")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(CANDIDATE_K)
    save_plot(fig, FIGURES_DIR / "kmeans_silhouette_scores.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(cluster_sizes.index.astype(str), cluster_sizes["learner_count"])
    ax.set_title(f"Final Cluster Sizes (K={selected_k})")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of learners")
    save_plot(fig, FIGURES_DIR / "cluster_sizes.png")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coordinates = pca.fit_transform(scaled_features)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        pca_coordinates[:, 0],
        pca_coordinates[:, 1],
        c=final_labels,
        cmap="tab10",
        alpha=0.75,
        edgecolors="none",
    )
    ax.set_title(f"Final K-Means Clusters in 2D PCA Space (K={selected_k})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    save_plot(fig, FIGURES_DIR / "cluster_visualization_pca.png")

    profile_relative = profiles / profiles.mean(axis=0)
    profile_descriptions = {}
    for cluster, row in profiles.iterrows():
        highest = row.nlargest(2).index.tolist()
        lowest = row.nsmallest(2).index.tolist()
        profile_descriptions[cluster] = (
            f"Cluster {cluster} has relatively higher values for "
            f"{highest[0]} and {highest[1]}, and relatively lower values for "
            f"{lowest[0]} and {lowest[1]}."
        )

    report_lines = [
        "# Step 5 — Learner Segmentation Using K-Means",
        "",
        "## 1. Objective",
        "Discover natural groups of learners from behavioral activity using unsupervised K-Means clustering.",
        "",
        "## 2. Why clustering is needed",
        "Classification predicts an existing engagement label. Clustering instead explores whether learners form behavioral groups without using that label.",
        "",
        "## 3. Supervised classification versus unsupervised clustering",
        "- Supervised classification uses `engagement_tier` as a target during training.",
        "- Unsupervised clustering uses only behavioral inputs and discovers group structure without target labels.",
        "",
        "## 4. Dataset used",
        f"- Source: `{DATA_PATH}`",
        f"- Learners used: {len(data)}",
        f"- Clustering feature missing values: {int(data[BEHAVIOR_FEATURES].isna().sum().sum())}",
        f"- Unique student IDs: {data[ID_COLUMN].nunique()}",
        "",
        "## 5. Features used",
        *[f"- `{feature}`" for feature in BEHAVIOR_FEATURES],
        "",
        "## 6. Features excluded and why",
        "- `student_id`, `student_name`: identifiers.",
        "- `engagement_tier`: supervised target, excluded from clustering.",
        "- `completed_course`: outcome/potential leakage variable.",
        "- `course_name`, `preferred_time_slot`: categorical fields excluded from this numerical behavioral analysis.",
        "",
        "## 7. Scaling/preprocessing",
        "The six numerical features were standardized with `StandardScaler` before K-Means. No PCA representation was used for clustering; PCA was used only for the final visualization.",
        "",
        "## 8. Candidate K values",
        f"Candidate values were K = {', '.join(map(str, CANDIDATE_K))}. Each model used `random_state=42` and `n_init=10`.",
        "",
        "## 9. Inertia results",
        "```",
        evaluation[["k", "inertia"]].to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "```",
        "",
        "## 10. Silhouette results",
        "```",
        evaluation[["k", "silhouette_score"]].to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "```",
        "",
        "## 11. Selected K and justification",
        f"K={selected_k} was selected because it produced the highest silhouette score ({selected_row['silhouette_score']:.6f}) among the evaluated candidates. Inertia decreases as K increases, so it was used as an elbow diagnostic rather than as a standalone selection rule.",
        "",
        "## 12. Cluster sizes",
        "```",
        format_table(cluster_sizes),
        "```",
        "",
        "## 13. Cluster profiles",
        "The following are mean behavioral values within each cluster:",
        "```",
        format_table(profiles),
        "```",
        "",
    ]
    for cluster, description in profile_descriptions.items():
        report_lines.append(f"- {description}")
    report_lines.extend(
        [
            "",
            "## 14. PCA visualization interpretation",
            f"The PCA plot is a two-dimensional visualization of the six-dimensional standardized feature space. The first two components explain {(pca.explained_variance_ratio_.sum() * 100):.1f}% of variance; apparent separation in this projection should not be treated as proof that all six-dimensional clusters are objectively correct.",
            "",
            "## 15. Comparison with engagement_tier",
            "The following table was created only after clustering and was not used to fit K-Means:",
            "```",
            crosstab.to_string(),
            "```",
            "",
            "Within-cluster engagement_tier percentages:",
            "```",
            crosstab_percentage.to_string(float_format=lambda value: f"{value:.2f}%"),
            "```",
            "",
            "These tables describe alignment/association between discovered behavioral groups and the existing labels; they do not establish causation or make the clusters equivalent to Low, Medium, or High engagement.",
            "",
            "## 16. Business/learning interpretation for personalized course delivery",
            "Course-delivery strategies should be based on the measured behavioral profiles rather than on preassigned names. Relatively higher activity clusters may be candidates for enrichment, pacing flexibility, and advanced practice. Relatively lower activity clusters may benefit from shorter milestones, reminders, low-friction practice, and additional support. These are hypotheses for later validation, not causal conclusions.",
            "",
            "## 17. Limitations",
            "- K-Means assumes distance-based, reasonably compact groups and can be sensitive to feature choices and the selected K.",
            "- The candidate K range was limited to 2–6.",
            "- Categorical behavior context was excluded from this initial numerical analysis.",
            "- Engagement labels were used only for post-clustering interpretation.",
            "",
            "## 18. Conclusion",
            f"K-Means segmentation was completed on all {len(data)} cleaned learners using only the six specified standardized behavioral features. K={selected_k} was selected using silhouette performance plus the inertia diagnostic. The resulting profiles and their association with `engagement_tier` are reported without treating either the clusters or labels as objectively correct categories.",
            "",
            "### Generated artifacts",
            "- `reports/figures/kmeans_elbow.png`",
            "- `reports/figures/kmeans_silhouette_scores.png`",
            "- `reports/figures/cluster_sizes.png`",
            "- `reports/figures/cluster_visualization_pca.png`",
            "- `reports/figures/cluster_assignments.csv`",
            "- `reports/figures/cluster_profiles.csv`",
            "- `reports/figures/cluster_engagement_crosstab.csv`",
            "- `reports/figures/kmeans_evaluation.csv`",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("STEP5_COMPLETE")
    print(f"Selected K: {selected_k}")
    print("K evaluation:")
    print(evaluation.to_string(index=False))
    print("Cluster sizes:")
    print(cluster_sizes.to_string())
    print("Cluster profiles:")
    print(profiles.to_string())
    print("Cluster versus engagement_tier:")
    print(crosstab.to_string())
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
