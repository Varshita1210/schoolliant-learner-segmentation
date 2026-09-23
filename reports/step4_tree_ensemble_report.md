# Step 4 — Trees and Ensembles Report

## 1. Objective
Build and evaluate Decision Tree and Random Forest classifiers for engagement tier prediction, then compare them with Step 3 baseline models.

## 2. Dataset used
- Source: `data/cleaned_learner_segmentation.csv`
- Shape: 400 rows × 12 columns
- Target: `engagement_tier`
- Target distribution:
  - Medium: 171
  - Low: 129
  - High: 100

## 3. Features used
- Numerical:
  - login_frequency_per_week
  - avg_session_duration_min
  - video_completion_pct
  - quiz_attempts
  - forum_posts
  - weekend_activity_pct
- Categorical:
  - course_name
  - preferred_time_slot

## 4. Features excluded and why
- `student_id`: identifier, not a behavior feature
- `student_name`: identifier, not a behavior feature
- `engagement_tier`: target variable
- `completed_course`: downstream outcome; excluded to avoid leakage risk

## 5. Train/test split
- `test_size=0.20`
- `stratify=y`
- `random_state=42`
- Train size: 320
- Test size: 80

## 6. Preprocessing approach
Used `ColumnTransformer` inside sklearn `Pipeline` for each model. Numerical features were scaled with `StandardScaler`; categorical features encoded with `OneHotEncoder`.

## 7. Decision Tree methodology
- Model: `DecisionTreeClassifier(random_state=42)`

## 8. Random Forest methodology
- Model: `RandomForestClassifier(random_state=42)`

## 9. Evaluation metrics
### Decision Tree
- accuracy: 0.962500
- precision_macro: 0.960317
- recall_macro: 0.970588
- f1_macro: 0.964140
- f1_weighted: 0.962250

### Random Forest
- accuracy: 1.000000
- precision_macro: 1.000000
- recall_macro: 1.000000
- f1_macro: 1.000000
- f1_weighted: 1.000000

## 10. Confusion matrices
- Decision Tree: `reports/figures/decision_tree_confusion.png`
- Random Forest: `reports/figures/random_forest_confusion.png`

## 11. Comparison of all five models
Saved to: `reports/figures/all_models_comparison.csv`

```
             model  accuracy  precision_macro  recall_macro  f1_macro  f1_weighted
LogisticRegression    1.0000         1.000000      1.000000   1.00000     1.000000
               KNN    0.9750         0.977850      0.973529   0.97536     0.974958
        NaiveBayes    1.0000         1.000000      1.000000   1.00000     1.000000
      DecisionTree    0.9625         0.960317      0.970588   0.96414     0.962250
      RandomForest    1.0000         1.000000      1.000000   1.00000     1.000000
```

## 12. Random Forest feature importance
Saved to: `reports/figures/random_forest_feature_importance.csv`

Top 10 features:
```
                                    feature  importance
                  num__video_completion_pct    0.294976
                         num__quiz_attempts    0.211030
              num__login_frequency_per_week    0.167197
              num__avg_session_duration_min    0.153501
                           num__forum_posts    0.114576
                  num__weekend_activity_pct    0.035481
         cat__preferred_time_slot_Afternoon    0.003570
           cat__preferred_time_slot_Evening    0.002466
cat__course_name_Deep Learning Fundamentals    0.002294
               cat__course_name_SQL Mastery    0.002136
```

## 13. Interpretation of results
If tree-based models are also very high-performing, that is consistent with strong separability in the current feature set. This does not by itself prove how the target labels were generated.

## 14. Limitations / warnings
- Evaluation is based on one train/test split for Step 4 baseline comparison.
- No hyperparameter tuning was performed in this step by design.
- Results should be interpreted with prior robustness audit context.

## 15. Conclusion
Decision Tree and Random Forest baselines were evaluated with the same data split and feature policy as Step 3, and compared directly with Logistic Regression, KNN, and Naive Bayes.

### Class-level report — Decision Tree
```
              precision    recall  f1-score   support

        High       0.95      1.00      0.98        20
         Low       0.93      1.00      0.96        26
      Medium       1.00      0.91      0.95        34

    accuracy                           0.96        80
   macro avg       0.96      0.97      0.96        80
weighted avg       0.96      0.96      0.96        80

```

### Class-level report — Random Forest
```
              precision    recall  f1-score   support

        High       1.00      1.00      1.00        20
         Low       1.00      1.00      1.00        26
      Medium       1.00      1.00      1.00        34

    accuracy                           1.00        80
   macro avg       1.00      1.00      1.00        80
weighted avg       1.00      1.00      1.00        80

```
