# Step 5 — Learner Segmentation Using K-Means

## 1. Objective
Discover natural groups of learners from behavioral activity using unsupervised K-Means clustering.

## 2. Why clustering is needed
Classification predicts an existing engagement label. Clustering instead explores whether learners form behavioral groups without using that label.

## 3. Supervised classification versus unsupervised clustering
- Supervised classification uses `engagement_tier` as a target during training.
- Unsupervised clustering uses only behavioral inputs and discovers group structure without target labels.

## 4. Dataset used
- Source: `data\cleaned_learner_segmentation.csv`
- Learners used: 400
- Clustering feature missing values: 0
- Unique student IDs: 400

## 5. Features used
- `login_frequency_per_week`
- `avg_session_duration_min`
- `video_completion_pct`
- `quiz_attempts`
- `forum_posts`
- `weekend_activity_pct`

## 6. Features excluded and why
- `student_id`, `student_name`: identifiers.
- `engagement_tier`: supervised target, excluded from clustering.
- `completed_course`: outcome/potential leakage variable.
- `course_name`, `preferred_time_slot`: categorical fields excluded from this numerical behavioral analysis.

## 7. Scaling/preprocessing
The six numerical features were standardized with `StandardScaler` before K-Means. No PCA representation was used for clustering; PCA was used only for the final visualization.

## 8. Candidate K values
Candidate values were K = 2, 3, 4, 5, 6. Each model used `random_state=42` and `n_init=10`.

## 9. Inertia results
```
 k     inertia
 2 1242.920974
 3  820.872653
 4  723.114580
 5  640.297156
 6  602.485326
```

## 10. Silhouette results
```
 k  silhouette_score
 2          0.381523
 3          0.370515
 4          0.310221
 5          0.301239
 6          0.213815
```

## 11. Selected K and justification
K=2 was selected because it produced the highest silhouette score (0.381523) among the evaluated candidates. Inertia decreases as K increases, so it was used as an elbow diagnostic rather than as a standalone selection rule.

## 12. Cluster sizes
```
         learner_count  percentage
cluster                           
0                  236      59.000
1                  164      41.000
```

## 13. Cluster profiles
The following are mean behavioral values within each cluster:
```
         login_frequency_per_week  avg_session_duration_min  video_completion_pct  quiz_attempts  forum_posts  weekend_activity_pct
cluster                                                                                                                            
0                           2.074                    22.885                42.312          2.737        1.322                20.442
1                           4.615                    47.674                78.740          7.354        6.329                39.246
```

- Cluster 0 has relatively lower mean values across all six behavioral measures.
- Cluster 1 has relatively higher mean values across all six behavioral measures.

## 14. PCA visualization interpretation
The PCA plot is a two-dimensional visualization of the six-dimensional standardized feature space. The first two components explain 81.2% of variance; apparent separation in this projection should not be treated as proof that all six-dimensional clusters are objectively correct.

## 15. Comparison with engagement_tier
The following table was created only after clustering and was not used to fit K-Means:
```
engagement_tier  Low  Medium  High
cluster                           
0                129     107     0
1                  0      64   100
```

Within-cluster engagement_tier percentages:
```
engagement_tier    Low  Medium   High
cluster                              
0               54.66%  45.34%  0.00%
1                0.00%  39.02% 60.98%
```

These tables describe alignment/association between discovered behavioral groups and the existing labels; they do not establish causation or make the clusters equivalent to Low, Medium, or High engagement.

## 16. Business/learning interpretation for personalized course delivery
Course-delivery strategies should be based on the measured behavioral profiles rather than on preassigned names. Relatively higher activity clusters may be candidates for enrichment, pacing flexibility, and advanced practice. Relatively lower activity clusters may benefit from shorter milestones, reminders, low-friction practice, and additional support. These are hypotheses for later validation, not causal conclusions.

## 17. Limitations
- K-Means assumes distance-based, reasonably compact groups and can be sensitive to feature choices and the selected K.
- The candidate K range was limited to 2–6.
- Categorical behavior context was excluded from this initial numerical analysis.
- Engagement labels were used only for post-clustering interpretation.

## 18. Conclusion
K-Means segmentation was completed on all 400 cleaned learners using only the six specified standardized behavioral features. K=2 was selected using silhouette performance plus the inertia diagnostic. The resulting profiles and their association with `engagement_tier` are reported without treating either the clusters or labels as objectively correct categories.

### Generated artifacts
- `reports/figures/kmeans_elbow.png`
- `reports/figures/kmeans_silhouette_scores.png`
- `reports/figures/cluster_sizes.png`
- `reports/figures/cluster_visualization_pca.png`
- `reports/figures/cluster_assignments.csv`
- `reports/figures/cluster_profiles.csv`
- `reports/figures/cluster_engagement_crosstab.csv`
- `reports/figures/kmeans_evaluation.csv`
