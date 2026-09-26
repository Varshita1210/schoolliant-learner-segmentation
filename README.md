# Learner Segmentation for Personalized Course Delivery

## Project Information

- **Organization:** Schoolliant
- **Track:** Machine Learning Internship
- **Project Type:** Machine Learning / Learner Analytics / Educational Data Mining
- **Status:** Steps 1–7 completed

## Project Overview

This project analyzes learner behavioral data, predicts engagement tiers, discovers behavioral segments using unsupervised clustering, and combines classification and clustering evidence to support personalized course-delivery strategies.

## Project Objectives

1. What distinct behavioral patterns exist?
2. Can engagement tier be predicted from early learner activity?
3. Which features matter most?
4. How could Schoolliant tailor course delivery for different learner segments?

## Dataset

- **Raw dataset:** `data/schoolliant_learner_segmentation.csv`
- **Raw size:** 405 rows × 12 columns
- **Cleaned dataset:** `data/cleaned_learner_segmentation.csv`
- **Cleaned size:** 400 rows × 12 columns
- **Target:** `engagement_tier`
- **Target classes:** Low, Medium, High
- **Cleaned target distribution:** Low 129, Medium 171, High 100

Important behavioral variables are:

- `login_frequency_per_week`
- `avg_session_duration_min`
- `video_completion_pct`
- `quiz_attempts`
- `forum_posts`
- `weekend_activity_pct`
- `course_name`
- `preferred_time_slot`

`student_id` and `student_name` were excluded from modeling because they are identifiers. `engagement_tier` was excluded from predictive inputs because it is the target. `completed_course` was excluded because it is an outcome/potential leakage variable.

## Project Workflow

### Step 1 — Data Understanding

The raw CSV was inspected for structure, column meanings, data types, missing values, duplicate rows, repeated student IDs, target distribution, categorical values, numerical statistics, and suspicious ranges. The six numerical activity variables and two categorical learner-context variables were identified for later supervised modeling.

The detailed Step 1 feature-separability report is not currently preserved, so this README does not reproduce historical separability statistics that are unavailable in the repository.

### Step 2 — Data Cleaning

The current cleaning script and report document the following procedure:

- Remove 3 exact duplicate rows.
- Fill missing `forum_posts` values with 0 as a count-field treatment; 12 missing values were recorded before cleaning.
- Group repeated `student_id` records into one learner row.
- Aggregate numeric columns by mean and categorical columns by mode.
- Set `completed_course` to `Yes` if any record in a repeated-ID group has `Yes`; otherwise use `No`.
- Round count-like `quiz_attempts` and `forum_posts` values to integers.

The raw dataset was preserved.

### Step 3 — Baseline Classification

Baseline classifiers were trained with the same 80/20 stratified train/test split and `random_state=42`:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

Evaluation included accuracy, macro precision, macro recall, macro F1, weighted F1, confusion matrices, and class-level classification reports. Preprocessing used numerical scaling and categorical one-hot encoding inside a pipeline.

### Robustness Investigation

The preserved evidence includes the Step 6 five-fold validation and tuning artifacts described below. The historical Step 3 robustness report and `reports/figures/step3_cross_validation_results.csv` are not currently preserved, so no specific historical Step 3 robustness metrics are claimed here.

### Step 4 — Decision Tree and Random Forest

Decision Tree and Random Forest baselines were added using the same feature policy and 80/20 stratified split. Their results were compared with the Step 3 models using the same test metrics. Random Forest feature importance was persisted to identify which encoded behavioral and categorical inputs contributed most to its predictions.

### Step 5 — Learner Segmentation

K-Means was applied using only the six numerical behavioral features after `StandardScaler`. `engagement_tier`, identifiers, `completed_course`, and categorical fields were excluded from clustering. Candidate values K=2, 3, 4, 5, and 6 were evaluated using inertia and silhouette score. K=2 was selected because it had the highest tested silhouette score, 0.3815228485.

The resulting solution contains two behavioral activity segments. PCA was used only to visualize the final clusters, not as the K-Means input.

### Step 6 — Cross-Validation and Hyperparameter Tuning

The five classification models were tuned with:

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- `GridSearchCV`
- `f1_macro` scoring
- preprocessing inside each model pipeline
- final evaluation on the held-out test set kept separate from tuning

### Step 7 — Final Capstone Report

The final report profiles the two behavioral segments, presents classification and clustering together, proposes data-informed course-delivery strategies, and documents limitations and evidence boundaries.

## Key Results

### Engagement distribution

- Low: **129**
- Medium: **171**
- High: **100**

### Baseline classification

These are held-out test-set results from the original Step 3 baseline:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.00000 | 1.000000 |
| KNN | 0.9750 | 0.97536 | 0.974958 |
| Naive Bayes | 1.0000 | 1.00000 | 1.000000 |

### Tree and ensemble results

These are held-out test-set results from Step 4:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Decision Tree | 0.9625 | 0.964140 | 0.962250 |
| Random Forest | 1.0000 | 1.000000 | 1.000000 |

### Tuned model results

The best CV Macro F1 values are five-fold cross-validation results from the training portion. Tuned test metrics are from the held-out 80-row test set.

| Model | Best CV Macro F1 | Tuned Test Accuracy | Tuned Test Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.987224 | 1.0000 | 1.000000 |
| KNN | 0.983881 | 0.9375 | 0.933199 |
| Naive Bayes | 0.987331 | 0.9875 | 0.986622 |
| Decision Tree | 0.960833 | 0.9000 | 0.905419 |
| Random Forest | 0.990479 | 1.0000 | 1.000000 |

These values describe different evaluation views; they are not used here to declare a universal best model or to rank models for every deployment purpose.

## Feature Importance

Persisted Random Forest feature importance values:

| Feature | Importance |
|---|---:|
| `video_completion_pct` | 0.2949761034 |
| `quiz_attempts` | 0.2110298928 |
| `login_frequency_per_week` | 0.1671969190 |
| `avg_session_duration_min` | 0.1535010390 |
| `forum_posts` | 0.1145756722 |
| `weekend_activity_pct` | 0.0354813412 |
| `preferred_time_slot_Afternoon` | 0.0035698183 |
| `preferred_time_slot_Evening` | 0.0024659095 |
| `course_name_Deep Learning Fundamentals` | 0.0022937078 |
| `course_name_SQL Mastery` | 0.0021361985 |
| `preferred_time_slot_Night` | 0.0021352301 |
| `course_name_Business Analytics` | 0.0020324894 |
| `course_name_Python for Beginners` | 0.0019143324 |
| `course_name_Machine Learning` | 0.0018826536 |
| `preferred_time_slot_Morning` | 0.0018418838 |
| `course_name_Data Science` | 0.0016519548 |
| `course_name_Power BI & Tableau` | 0.0013148539 |

Feature importance is model-derived and does not establish causality.

## Learner Segments

| Segment | Learners | Percentage | Login frequency | Session duration (min) | Video completion (%) | Quiz attempts | Forum posts | Weekend activity (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Relatively Lower-Activity Behavioral Segment | 236 | 59.0% | 2.074153 | 22.884746 | 42.312288 | 2.737288 | 1.322034 | 20.441949 |
| Relatively Higher-Activity Behavioral Segment | 164 | 41.0% | 4.615244 | 47.674390 | 78.740244 | 7.353659 | 6.329268 | 39.245732 |

Cluster 0 is the Relatively Lower-Activity Behavioral Segment. Cluster 1 is the Relatively Higher-Activity Behavioral Segment.

The cluster-versus-engagement counts are:

| Cluster | Low | Medium | High |
|---:|---:|---:|---:|
| 0 | 129 | 107 | 0 |
| 1 | 0 | 64 | 100 |

Within-cluster distributions are Cluster 0: Low 54.66%, Medium 45.34%, High 0.00%; and Cluster 1: Low 0.00%, Medium 39.02%, High 60.98%. This shows strong alignment with the existing labels in this dataset, but clusters are not equivalent to engagement tiers and the alignment is not causal evidence.

## Personalization Strategy

| Segment | Observed Characteristics | Possible Course-Delivery Strategy |
|---|---|---|
| Relatively Lower-Activity Behavioral Segment | Lower means across all six behavioral measures | Shorter modules, progress reminders, targeted practice prompts, easier re-entry into unfinished content, and prompts to complete videos and quizzes |
| Relatively Higher-Activity Behavioral Segment | Higher means across all six behavioral measures | Optional advanced material, additional practice, deeper discussion activities, extension content, and flexible learning opportunities |

These are data-informed product recommendations based on observed behavior, not experimentally validated interventions.

## Limitations

- The detailed historical Step 1 separability report is not currently preserved.
- The historical Step 3 robustness report and cross-validation CSV are not currently preserved.
- Step 5 used the six original numerical behavioral features.
- A consistency score was not implemented.
- Time-of-day preference was not used as a clustering feature.
- A weekend-versus-weekday activity ratio was not engineered.
- Cluster stability across multiple runs was not evaluated.
- High classification performance does not prove causality or prove how the target label was originally constructed.
- Feature importance does not establish causality.
- Cluster-label alignment does not establish causality.
- The K=2 solution represents two behavioral activity groups, not three clusters corresponding directly to Low, Medium, and High.

## Project Structure

```text
schoolliant-learner-segmentation/
├── data/
│   ├── schoolliant_learner_segmentation.csv
│   └── cleaned_learner_segmentation.csv
├── src/
│   ├── step1_data_understanding.py
│   ├── step2_data_cleaning.py
│   ├── step3_baseline_classification.py
│   ├── step3_baseline_robustness_audit.py
│   ├── step4_tree_ensemble.py
│   ├── step5_clustering.py
│   └── step6_model_validation_tuning.py
├── reports/
│   ├── step2_data_cleaning_report.md
│   ├── step3_baseline_classification_report.md
│   ├── step4_tree_ensemble_report.md
│   ├── step5_clustering_report.md
│   ├── step6_model_validation_tuning_report.md
│   ├── step7_final_capstone_report.md
│   └── figures/
├── README.md
├── requirements.txt
└── .gitignore
```

The `reports/figures/` directory contains confusion matrices, model comparison tables, tuning outputs, K-Means evaluation tables, cluster profiles, cluster assignments, and feature-importance results.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Git/GitHub

## How to Run

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Run the completed Python workflows from the repository root:

```bash
python src/step1_data_understanding.py
python src/step2_data_cleaning.py
python src/step3_baseline_classification.py
python src/step3_baseline_robustness_audit.py
python src/step4_tree_ensemble.py
python src/step5_clustering.py
python src/step6_model_validation_tuning.py
```

The scripts read from the repository's `data/` directory and write reports and generated tables/figures under `reports/`. The raw dataset should not be overwritten. Jupyter Notebook is not required to run the completed Python workflows.

## Final Outcome

The completed project combines data cleaning → classification → validation → tree and ensemble analysis → behavioral clustering → hyperparameter tuning → personalization strategy. Classification predicts the existing engagement tier, while clustering discovers behavioral groups independently of that target; together they provide a cautious, data-backed framework for differentiated learner support.
