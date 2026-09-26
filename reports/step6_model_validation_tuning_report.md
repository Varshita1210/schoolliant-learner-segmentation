# Step 6 — Model Validation and Hyperparameter Tuning

## 1. Objective
Validate and tune the Step 3 and Step 4 classification models with leakage-controlled five-fold stratified cross-validation.

## 2. Dataset and features
- Dataset: `data\cleaned_learner_segmentation.csv`
- Shape: 400 rows × 12 columns
- Target: `engagement_tier`
- Numerical features: `login_frequency_per_week`, `avg_session_duration_min`, `video_completion_pct`, `quiz_attempts`, `forum_posts`, `weekend_activity_pct`
- Categorical features: `course_name`, `preferred_time_slot`
- Excluded: `student_id`, `student_name`, `engagement_tier`, `completed_course`.

## 3. Train/test split
- `test_size=0.20`, `stratify=y`, `random_state=42`.
- Training rows: 320
- Test rows: 80
- The test set was held out before GridSearchCV and was used only for final tuned-model evaluation.

## 4. Preprocessing
Each estimator used a Pipeline containing a ColumnTransformer. Numerical columns were standardized and categorical columns were one-hot encoded. The preprocessing was fitted separately within each cross-validation training fold.

## 5. Cross-validation strategy
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- GridSearchCV scoring: `f1_macro`.
- `refit=True` selected and refit the best parameter setting using the training data only.

## 6. Hyperparameter grids
### LogisticRegression
```
{
  "model__C": [
    0.01,
    0.1,
    1,
    10,
    100
  ],
  "model__solver": [
    "lbfgs"
  ],
  "model__max_iter": [
    1000
  ]
}
```

### KNN
```
{
  "model__n_neighbors": [
    3,
    5,
    7,
    9,
    11
  ],
  "model__weights": [
    "uniform",
    "distance"
  ],
  "model__metric": [
    "euclidean",
    "manhattan"
  ]
}
```

### NaiveBayes
```
{
  "model__var_smoothing": [
    1e-11,
    1e-10,
    1e-09,
    1e-08,
    1e-07
  ]
}
```

### DecisionTree
```
{
  "model__max_depth": [
    null,
    3,
    5,
    7,
    10
  ],
  "model__min_samples_split": [
    2,
    5,
    10
  ],
  "model__min_samples_leaf": [
    1,
    2,
    4
  ],
  "model__criterion": [
    "gini",
    "entropy"
  ]
}
```

### RandomForest
```
{
  "model__n_estimators": [
    100,
    200
  ],
  "model__max_depth": [
    null,
    5,
    10,
    15
  ],
  "model__min_samples_split": [
    2,
    5
  ],
  "model__min_samples_leaf": [
    1,
    2
  ],
  "model__max_features": [
    "sqrt",
    "log2"
  ]
}
```

## 7. Baseline results
Baseline values below were read from the existing Step 3/4 comparison CSV; they were not recomputed or fabricated.
```
             model  baseline_accuracy  baseline_f1_macro  step6_best_cv_f1_macro  tuned_test_accuracy  tuned_test_f1_macro
      DecisionTree             0.9625            0.96414                0.960833               0.9000             0.905419
               KNN             0.9750            0.97536                0.983881               0.9375             0.933199
LogisticRegression             1.0000            1.00000                0.987224               1.0000             1.000000
        NaiveBayes             1.0000            1.00000                0.987331               0.9875             0.986622
      RandomForest             1.0000            1.00000                0.990479               1.0000             1.000000
```

## 8. Tuned CV results
```
             model  best_cv_f1_macro
LogisticRegression          0.987224
               KNN          0.983881
        NaiveBayes          0.987331
      DecisionTree          0.960833
      RandomForest          0.990479
```

## 9. Tuned test results
```
             model  test_accuracy  test_precision_macro  test_recall_macro  test_f1_macro  test_f1_weighted
LogisticRegression         1.0000              1.000000           1.000000       1.000000          1.000000
               KNN         0.9375              0.951618           0.923529       0.933199          0.936161
        NaiveBayes         0.9875              0.990476           0.983333       0.986622          0.987430
      DecisionTree         0.9000              0.919591           0.895777       0.905419          0.900620
      RandomForest         1.0000              1.000000           1.000000       1.000000          1.000000
```

## 10. Baseline vs tuned comparison
```
             model  baseline_accuracy  baseline_f1_macro  step6_best_cv_f1_macro  tuned_test_accuracy  tuned_test_f1_macro
      DecisionTree             0.9625            0.96414                0.960833               0.9000             0.905419
               KNN             0.9750            0.97536                0.983881               0.9375             0.933199
LogisticRegression             1.0000            1.00000                0.987224               1.0000             1.000000
        NaiveBayes             1.0000            1.00000                0.987331               0.9875             0.986622
      RandomForest             1.0000            1.00000                0.990479               1.0000             1.000000
```

## 11. Best parameters for each model
Saved to `reports/figures/step6_best_parameters.csv`.
```
             model  best_cv_f1_macro                                                                                                                                    best_parameters
LogisticRegression          0.987224                                                                                 {"model__C": 1, "model__max_iter": 1000, "model__solver": "lbfgs"}
               KNN          0.983881                                                              {"model__metric": "euclidean", "model__n_neighbors": 11, "model__weights": "uniform"}
        NaiveBayes          0.987331                                                                                                                    {"model__var_smoothing": 1e-11}
      DecisionTree          0.960833                             {"model__criterion": "entropy", "model__max_depth": null, "model__min_samples_leaf": 1, "model__min_samples_split": 5}
      RandomForest          0.990479 {"model__max_depth": null, "model__max_features": "sqrt", "model__min_samples_leaf": 1, "model__min_samples_split": 2, "model__n_estimators": 100}
```

## 12. Interpretation
The primary selection measure is cross-validated macro F1, which weights the three classes equally. Test macro F1 is a final held-out check, not the tuning criterion. Model complexity and interpretability also matter: linear Logistic Regression and Naive Bayes are simpler, KNN is instance-based, and tree ensembles are more complex.
Random Forest had the highest best-CV macro F1 (0.990479). Logistic Regression and Random Forest both achieved a tuned test macro F1 of 1.000000, while Logistic Regression is the simpler and more interpretable of those tied test performers.

## 13. Whether tuning materially improved performance
- DecisionTree: tuned test macro F1 changed by -0.058721 versus its recorded baseline.
- KNN: tuned test macro F1 changed by -0.042161 versus its recorded baseline.
- LogisticRegression: tuned test macro F1 changed by +0.000000 versus its recorded baseline.
- NaiveBayes: tuned test macro F1 changed by -0.013378 versus its recorded baseline.
- RandomForest: tuned test macro F1 changed by +0.000000 versus its recorded baseline.

Because the prior baselines were already extremely strong, unchanged or very small differences should be interpreted as confirmation of the existing separability rather than evidence that tuning produced a substantial improvement.

## 14. Limitations
- The held-out test set is one fixed split, although model selection used five-fold stratified CV on the training portion.
- The searched grids are finite and do not guarantee globally optimal hyperparameters.
- `completed_course` was excluded because it may represent downstream outcome information.
- High scores do not prove how `engagement_tier` was constructed.

## Quality checks
- All five configured models completed GridSearchCV.
- Preprocessing remained inside each Pipeline.
- No clustering or Step 7 work was performed.
