# Learner Segmentation for Personalized Course Delivery

## Project Information
- **Organization:** Schoolliant
- **Track:** Machine Learning Internship
- **Project Type:** Machine Learning / Learner Analytics / Educational Data Mining

## Project Objective
This project analyzes learner engagement behavior to predict engagement tiers, identify meaningful behavioral learner segments, and support personalized course-delivery strategy recommendations based on the official project brief.

## Current Status
**Step 1 — Dataset Understanding and Project Setup**

The dataset has been inspected for structure and quality indicators. Modeling, clustering, and recommendation development have not started.

## Official 7-Step Plan
1. **Step 1: Understand and Collect the Data**
2. **Step 2: Clean the Data**
3. **Step 3: Build Baseline Classification Models**  
   Logistic Regression, KNN, Naive Bayes
4. **Step 4: Improve with Trees and Ensembles**  
   Decision Tree, Random Forest
5. **Step 5: Segment Learners Using Clustering**  
   K-Means or Hierarchical Clustering, elbow method, and behavioral features such as consistency, time-of-day preference, and weekend-vs-weekday activity
6. **Step 6: Cross-Validation and Hyperparameter Tuning**  
   Cross-validation, GridSearch or similar hyperparameter tuning, and clustering stability checks
7. **Step 7: Final Capstone Report**  
   Learner segment profiling, personalized course-delivery recommendations, and combined classification + clustering interpretation

## Dataset
- **Filename:** `schoolliant_learner_segmentation.csv`
- **Rows:** 405
- **Columns:** 12
- **Unique student IDs:** 400
- **Exact duplicate rows:** 3
- **Missing values:** 12 in `forum_posts`
- **Target (`engagement_tier`) classes present:** Low, Medium, High
- **Target distribution:**
  - Medium: 172
  - Low: 130
  - High: 103

### Columns
- `student_id`
- `student_name`
- `course_name`
- `login_frequency_per_week`
- `avg_session_duration_min`
- `video_completion_pct`
- `quiz_attempts`
- `forum_posts`
- `weekend_activity_pct`
- `preferred_time_slot`
- `engagement_tier`
- `completed_course`

## Data Governance Notes
- The original dataset is preserved.
- Data cleaning and transformation will be performed in **Step 2**.
- Learner segments are **not** assumed in advance; they will be identified from clustering analysis in **Step 5**.
