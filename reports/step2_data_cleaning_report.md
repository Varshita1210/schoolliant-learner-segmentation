# Step 2 — Data Cleaning Report

Source file: data/schoolliant_learner_segmentation.csv

Original shape: 405 rows, 12 columns

Cleaning actions applied:
- Removed exact duplicate rows (3)
- Imputed missing forum_posts with 0 (forum_posts missing before: 12)
- Resolved repeated student_id records by aggregating: numeric->mean, categorical->mode, completed_course->Yes if any

Results:
- Cleaned shape: 400 rows, 12 columns

Conflicting student_id records inspected (showing rows before cleaning):

--- S2329 ---
student_id,student_name,course_name,login_frequency_per_week,avg_session_duration_min,video_completion_pct,quiz_attempts,forum_posts,weekend_activity_pct,preferred_time_slot,engagement_tier,completed_course
S2329,Adeel Chaudhry,Deep Learning Fundamentals,0.1,9.7,12.3,0,1.0,16.0,Morning,Low,No
S2329,Adeel Chaudhry,Deep Learning Fundamentals,0.1,9.7,12.3,0,1.0,16.0,Morning,Low,No

--- S2126 ---
student_id,student_name,course_name,login_frequency_per_week,avg_session_duration_min,video_completion_pct,quiz_attempts,forum_posts,weekend_activity_pct,preferred_time_slot,engagement_tier,completed_course
S2126,Zain Warsi,SQL Mastery,5.3,49.8,87.9,9,5.0,57.6,Night,High,Yes
S2126,Zain Warsi,SQL Mastery,5.3,49.8,87.9,9,5.0,57.6,Night,High,Yes

--- S2173 ---
student_id,student_name,course_name,login_frequency_per_week,avg_session_duration_min,video_completion_pct,quiz_attempts,forum_posts,weekend_activity_pct,preferred_time_slot,engagement_tier,completed_course
S2173,Hina Baig,Power BI & Tableau,6.4,58.4,82.5,8,12.0,27.5,Morning,High,Yes
S2173,Hina Baig,Power BI & Tableau,6.4,58.4,82.5,8,,27.5,Morning,High,Yes

--- S2340 ---
student_id,student_name,course_name,login_frequency_per_week,avg_session_duration_min,video_completion_pct,quiz_attempts,forum_posts,weekend_activity_pct,preferred_time_slot,engagement_tier,completed_course
S2340,Iqra Abbasi,Machine Learning,5.3,69.7,64.9,9,4.0,21.3,Evening,High,Yes
S2340,Iqra Abbasi,Machine Learning,5.3,69.7,64.9,9,,21.3,Evening,High,Yes

--- S2399 ---
student_id,student_name,course_name,login_frequency_per_week,avg_session_duration_min,video_completion_pct,quiz_attempts,forum_posts,weekend_activity_pct,preferred_time_slot,engagement_tier,completed_course
S2399,Imran Chaudhry,Data Science,2.3,29.0,48.3,5,3.0,29.2,Morning,Medium,Yes
S2399,Imran Chaudhry,Data Science,2.3,29.0,48.3,5,3.0,29.2,Morning,Medium,Yes


Note: Aggregation rules are documented above. If any aggregation decision is not acceptable, request manual review before using the cleaned dataset.