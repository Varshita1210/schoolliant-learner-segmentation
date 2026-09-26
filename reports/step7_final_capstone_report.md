# Step 7: Final Capstone Report

## 1. Project Objective

Schoolliant's objective is to understand learner behavioral patterns, predict the existing `engagement_tier`, discover behavioral learner segments without using that target as a clustering input, and use the findings to support differentiated course delivery.

## 2. Dataset and Data Preparation

The raw dataset was `data/schoolliant_learner_segmentation.csv` with **405 rows and 12 columns**. The final documented modeling dataset, `data/cleaned_learner_segmentation.csv`, contains **400 learners and 12 columns**.

The current Step 2 script and report document this procedure:

- Remove **3 exact duplicate rows**.
- Fill missing `forum_posts` values with **0** as a count-field treatment. The report records **12 missing values before this step**.
- Group repeated `student_id` records into one row per learner.
- Aggregate numeric columns by mean and categorical columns by mode.
- Set `completed_course` to `Yes` if any record in a repeated-ID group has `Yes`; otherwise use `No`.
- Round count-like `quiz_attempts` and `forum_posts` values to integers.

The raw CSV was preserved. The modeling inputs excluded `student_id` and `student_name` as identifiers, `engagement_tier` as the target, and `completed_course` as an outcome/potential leakage field.

The cleaned target distribution is:

| Engagement tier | Learners |
|---|---:|
| Low | 129 |
| Medium | 171 |
| High | 100 |

## 3. Behavioral Patterns

The detailed Step 1 feature-separability report is not currently preserved, so its historical numerical separability statistics are not reproduced here. Persisted model and clustering artifacts provide the following descriptive evidence:

- The six behavioral variables used throughout modeling were login frequency, average session duration, video completion, quiz attempts, forum posts, and weekend activity.
- The Random Forest importance values place video completion, quiz attempts, login frequency, and session duration above forum posts and weekend activity.
- Step 5 profiles show a lower-mean and higher-mean activity grouping across all six clustering variables.

These are descriptive/model-based findings. They do not establish that any feature causes engagement tier.

## 4. Engagement Classification

The baseline results below are held-out test-set results from the 80/20 stratified split. Step 6's best CV Macro F1 is a five-fold training-set cross-validation result, while its tuned test values are from the untouched test split.

| Model | Baseline Test Accuracy | Tuned Test Accuracy | Tuned Test Macro F1 | Best CV Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.000000 | 0.987224 |
| KNN | 0.9750 | 0.9375 | 0.933199 | 0.983881 |
| Naive Bayes | 1.0000 | 0.9875 | 0.986622 | 0.987331 |
| Decision Tree | 0.9625 | 0.9000 | 0.905419 | 0.960833 |
| Random Forest | 1.0000 | 1.0000 | 1.000000 | 0.990479 |

The values show that several models have very high test performance on this dataset, while tuned KNN and Decision Tree test results are lower than their earlier baseline results on the same split. These are observations from the recorded evaluations, not a universal model ranking.

## 5. Robustness and Validation

Steps 3 and 4 used an **80/20 stratified train/test split** with `random_state=42`; the recorded split sizes were **320 training rows and 80 test rows**.

Step 6 used `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` inside `GridSearchCV`, with `f1_macro` as the primary scoring metric. Numerical scaling and categorical encoding were inside each model's `Pipeline` and `ColumnTransformer`. The held-out test set was separated before tuning and used only for final tuned evaluation.

The historical Step 3 robustness report and `reports/figures/step3_cross_validation_results.csv` are not currently preserved. No Step 3 robustness values are claimed here.

## 6. Feature Importance

The persisted Random Forest feature importance values are:

| Encoded feature | Importance |
|---|---:|
| `num__video_completion_pct` | 0.2949761034 |
| `num__quiz_attempts` | 0.2110298928 |
| `num__login_frequency_per_week` | 0.1671969190 |
| `num__avg_session_duration_min` | 0.1535010390 |
| `num__forum_posts` | 0.1145756722 |
| `num__weekend_activity_pct` | 0.0354813412 |
| `cat__preferred_time_slot_Afternoon` | 0.0035698183 |
| `cat__preferred_time_slot_Evening` | 0.0024659095 |
| `cat__course_name_Deep Learning Fundamentals` | 0.0022937078 |
| `cat__course_name_SQL Mastery` | 0.0021361985 |
| `cat__preferred_time_slot_Night` | 0.0021352301 |
| `cat__course_name_Business Analytics` | 0.0020324894 |
| `cat__course_name_Python for Beginners` | 0.0019143324 |
| `cat__course_name_Machine Learning` | 0.0018826536 |
| `cat__preferred_time_slot_Morning` | 0.0018418838 |
| `cat__course_name_Data Science` | 0.0016519548 |
| `cat__course_name_Power BI & Tableau` | 0.0013148539 |

Feature importance is model-derived importance, not a causal effect or independent proof of predictive necessity.

## 7. Learner Segmentation

Step 5 used K-Means on only the six numerical behavioral features after `StandardScaler`. `engagement_tier`, identifiers, `completed_course`, `course_name`, and `preferred_time_slot` were excluded from clustering. K-Means used `random_state=42` and `n_init=10`. PCA was used only for visualization after clustering.

Candidate K values were **2, 3, 4, 5, and 6**. The recorded evaluation was:

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1242.9209738457 | 0.3815228485 |
| 3 | 820.8726531176 | 0.3705147775 |
| 4 | 723.1145801252 | 0.3102212055 |
| 5 | 640.2971555122 | 0.3012389899 |
| 6 | 602.4853260302 | 0.2138151899 |

K=2 was selected because it had the highest tested silhouette score, **0.3815228485**. Inertia was used as an elbow diagnostic, not as the sole selection rule.

## 8. Cluster Profiles

| Segment | Learners | Percentage | Login frequency | Session duration (min) | Video completion (%) | Quiz attempts | Forum posts | Weekend activity (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cluster 0 — Relatively Lower-Activity Behavioral Segment | 236 | 59.0 | 2.074153 | 22.884746 | 42.312288 | 2.737288 | 1.322034 | 20.441949 |
| Cluster 1 — Relatively Higher-Activity Behavioral Segment | 164 | 41.0 | 4.615244 | 47.674390 | 78.740244 | 7.353659 | 6.329268 | 39.245732 |

Cluster 0 has relatively lower mean values across all six behavioral measures. Cluster 1 has relatively higher mean values across all six behavioral measures. These are neutral behavioral descriptions, not causal categories.

## 9. Cluster and Engagement Relationship

The persisted post-clustering crosstab is:

| Cluster | Low | Medium | High |
|---:|---:|---:|---:|
| 0 | 129 | 107 | 0 |
| 1 | 0 | 64 | 100 |

Within-cluster percentages:

- Cluster 0: Low **54.66%**, Medium **45.34%**, High **0.00%**.
- Cluster 1: Low **0.00%**, Medium **39.02%**, High **60.98%**.

The clustering shows strong alignment with the existing engagement labels in this dataset. The clusters are not equivalent to the labels, and the relationship does not establish causality.

## 10. Personalization Strategy

| Segment | Observed Characteristics | Possible Course-Delivery Strategy | Rationale |
|---|---|---|---|
| Relatively Lower-Activity Behavioral Segment | Lower means across login frequency, session duration, video completion, quiz attempts, forum posts, and weekend activity | Shorter learning modules; progress reminders; targeted practice prompts; easier re-entry into unfinished content; encouragement to complete videos and quizzes | Reduce friction and provide structured prompts around the behaviors that are lower in this segment |
| Relatively Higher-Activity Behavioral Segment | Higher means across all six behavioral measures | Optional advanced material; additional practice; deeper discussion activities; extension content; flexible learning opportunities | Provide enrichment and flexibility that matches the segment's observed higher activity profile |

These are proposed product strategies based on observed behavior. They have not been experimentally shown in this project to improve learner outcomes.

## 11. Answers to the Four Schoolliant Questions

### Question 1: What distinct behavioral patterns exist?

The persisted K-Means solution identifies two broad behavioral activity profiles: a relatively lower-activity segment of **236 learners (59.0%)** and a relatively higher-activity segment of **164 learners (41.0%)**. The higher-activity segment has larger means for all six clustering variables.

### Question 2: Can engagement tier be predicted from learner activity?

On the recorded held-out test split, Logistic Regression and Random Forest achieved **1.0000 accuracy and 1.000000 macro F1** after tuning. Naive Bayes achieved **0.9875 accuracy and 0.986622 macro F1**, while tuned KNN and Decision Tree achieved **0.9375/0.933199** and **0.9000/0.905419**, respectively. Five-fold CV best Macro F1 values ranged from **0.960833** to **0.990479**. These results indicate strong predictive separability in this dataset, not guaranteed performance on future populations.

### Question 3: Which features matter most?

The persisted Random Forest importance values are highest for `video_completion_pct` (**0.2949761034**), `quiz_attempts` (**0.2110298928**), `login_frequency_per_week` (**0.1671969190**), and `avg_session_duration_min` (**0.1535010390**). This is model-derived evidence and should not be interpreted causally.

### Question 4: How could Schoolliant tailor course delivery for each segment?

For the relatively lower-activity segment, Schoolliant could test shorter modules, reminders, targeted practice, re-entry support, and prompts to complete videos and quizzes. For the relatively higher-activity segment, it could test optional advanced material, extra practice, deeper discussion, extension content, and flexible pacing. These are hypotheses for product experimentation, not proven interventions.

## 12. Limitations

1. The detailed Step 1 feature-separability report is not currently preserved.
2. The Step 3 robustness report and cross-validation CSV are not currently preserved.
3. The current Step 2 report reflects the current cleaning script and should be treated as the documented cleaning methodology.
4. Step 5 used the six original numerical behavioral variables rather than engineered consistency/time-of-day/weekend-weekday features.
5. Cluster stability across multiple runs was not evaluated.
6. Classification results rely on the specified 80/20 test split, while Step 6 also provides five-fold cross-validation for tuning.
7. Very high classification scores indicate strong separability in this dataset but do not prove causation or prove how the target label was originally constructed.
8. Feature importance and cluster-label alignment are descriptive/model-based evidence rather than causal effects.
9. The K=2 clustering solution represents two behavioral activity groups, not three clusters corresponding directly to Low/Medium/High engagement.

## 13. Final Conclusion

The project progressed from data understanding and documented cleaning to baseline classification, tree and ensemble modeling, learner clustering, and Step 6 cross-validation and tuning. Classification predicts the existing `engagement_tier`; clustering discovers behavioral groups without using `engagement_tier` as an input. Together, these provide complementary evidence for a data-backed personalization framework: model-derived engagement prediction can coexist with behaviorally defined segments and proposed differentiated delivery strategies.

The evidence supports cautious product hypotheses, especially around lower-activity and higher-activity behavioral profiles. It does not establish causality, prove the origin of the target label, or establish that the two-cluster solution is objectively correct.
