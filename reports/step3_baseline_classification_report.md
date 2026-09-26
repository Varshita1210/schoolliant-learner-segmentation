# Step 3 — Baseline Classification Report

Dataset: data/cleaned_learner_segmentation.csv

Train/test split: 80/20 stratified by engagement_tier, random_state=42

Excluded features from modeling: student_id, student_name, engagement_tier, completed_course

## Models evaluated
- LogisticRegression
- KNN
- NaiveBayes

## Metrics
             model  accuracy  precision_macro  recall_macro  f1_macro  f1_weighted
LogisticRegression     1.000          1.00000      1.000000   1.00000     1.000000
               KNN     0.975          0.97785      0.973529   0.97536     0.974958
        NaiveBayes     1.000          1.00000      1.000000   1.00000     1.000000

## Confusion matrices saved in reports/figures/
- LogisticRegression: reports/figures/logisticregression_confusion.png
- KNN: reports/figures/knn_confusion.png
- NaiveBayes: reports/figures/naivebayes_confusion.png


## Detailed classification reports
### LogisticRegression

```
              precision    recall  f1-score   support

        High       1.00      1.00      1.00        20
         Low       1.00      1.00      1.00        26
      Medium       1.00      1.00      1.00        34

    accuracy                           1.00        80
   macro avg       1.00      1.00      1.00        80
weighted avg       1.00      1.00      1.00        80

```
### KNN

```
              precision    recall  f1-score   support

        High       1.00      0.95      0.97        20
         Low       0.96      1.00      0.98        26
      Medium       0.97      0.97      0.97        34

    accuracy                           0.97        80
   macro avg       0.98      0.97      0.98        80
weighted avg       0.98      0.97      0.97        80

```
### NaiveBayes

```
              precision    recall  f1-score   support

        High       1.00      1.00      1.00        20
         Low       1.00      1.00      1.00        26
      Medium       1.00      1.00      1.00        34

    accuracy                           1.00        80
   macro avg       1.00      1.00      1.00        80
weighted avg       1.00      1.00      1.00        80

```
