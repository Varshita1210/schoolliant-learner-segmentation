import pandas as pd

DATA_PATH = 'data/schoolliant_learner_segmentation.csv'

print('# Step 1 — Data Understanding')

# 1. Import Libraries
print('\n## 1. Import Libraries')
print('Imported pandas successfully.')

# 2. Load Dataset
print('\n## 2. Load Dataset')
df = pd.read_csv(DATA_PATH)
print(f'Dataset loaded from: {DATA_PATH}')

# 3. Initial Inspection
print('\n## 3. Initial Inspection')
print('\nFirst 5 rows:')
print(df.head().to_string(index=False))
print('\nLast 5 rows:')
print(df.tail().to_string(index=False))

# 4. Dataset Structure
print('\n## 4. Dataset Structure')
rows, cols = df.shape
print(f'Shape: ({rows}, {cols})')
print('\nColumn names:')
for col in df.columns:
    print(f'- {col}')

# 5. Data Types
print('\n## 5. Data Types')
print(df.dtypes.to_string())

# 6. Missing Values
print('\n## 6. Missing Values')
missing_by_column = df.isna().sum()
total_missing = int(missing_by_column.sum())
print('Missing values per column:')
print(missing_by_column.to_string())
print(f'\nTotal missing values: {total_missing}')

# 7. Duplicate Records
print('\n## 7. Duplicate Records')
exact_duplicate_rows = int(df.duplicated().sum())
print(f'Exact duplicate rows: {exact_duplicate_rows}')

# 8. Student ID Analysis
print('\n## 8. Student ID Analysis')
unique_student_ids = int(df['student_id'].nunique())
student_id_is_unique = bool(df['student_id'].is_unique)
duplicate_student_id_counts = df['student_id'].value_counts()
duplicate_student_id_counts = duplicate_student_id_counts[duplicate_student_id_counts > 1]
print(f'Unique student_id values: {unique_student_ids}')
print(f'Is student_id unique? {student_id_is_unique}')
print('Duplicate student_id counts:')
if duplicate_student_id_counts.empty:
    print('None')
else:
    print(duplicate_student_id_counts.to_string())

print('\nNote: Exact duplicate rows and repeated student_id values are different checks. Repeated student_id values may represent multiple records per learner and are only identified in Step 1, not altered.')

# 9. Engagement Tier Distribution
print('\n## 9. Engagement Tier Distribution')
engagement_counts = df['engagement_tier'].value_counts(dropna=False)
print(engagement_counts.to_string())

# 10. Categorical Variables
print('\n## 10. Categorical Variables')
categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print('Categorical columns:')
for col in categorical_columns:
    print(f'- {col}')
print('\nUnique values for each categorical column:')
for col in categorical_columns:
    unique_vals = sorted(df[col].dropna().astype(str).unique().tolist())
    print(f'\n{col} ({len(unique_vals)} unique):')
    print(unique_vals)

# 11. Numerical Variables
print('\n## 11. Numerical Variables')
numerical_columns = df.select_dtypes(include='number').columns.tolist()
print('Numerical columns:')
for col in numerical_columns:
    print(f'- {col}')
print('\nDescriptive statistics:')
print(df[numerical_columns].describe().to_string())

# 12. Data Validation Checks
print('\n## 12. Data Validation Checks')
percentage_columns = [c for c in df.columns if ('pct' in c.lower() or 'percent' in c.lower() or 'percentage' in c.lower()) and pd.api.types.is_numeric_dtype(df[c])]
print('Percentage-range validation (0 to 100):')
for col in percentage_columns:
    out_of_range = int(((df[col] < 0) | (df[col] > 100)).sum())
    print(f'- {col}: {out_of_range} values outside 0-100')

count_columns = [c for c in ['login_frequency_per_week', 'quiz_attempts', 'forum_posts'] if c in df.columns]
print('\nNegative-value validation for count columns:')
for col in count_columns:
    negatives = int((df[col] < 0).sum())
    print(f'- {col}: {negatives} negative values')

session_duration_col = 'avg_session_duration_min'
session_duration_negatives = int((df[session_duration_col] < 0).sum())
print('\nNegative session-duration validation:')
print(f'- {session_duration_col}: {session_duration_negatives} negative values')

# 13. Data Dictionary
print('\n## 13. Data Dictionary')
print('| Column | Data Type | Role | Description |')
print('|---|---|---|---|')

data_dictionary = [
    ('student_id', str(df['student_id'].dtype), 'Identifier', 'Unique learner identifier field; repeated IDs may indicate multiple records for the same learner.'),
    ('student_name', str(df['student_name'].dtype), 'Identifier', 'Learner name field; personally identifying attribute and not a modeling feature.'),
    ('course_name', str(df['course_name'].dtype), 'Categorical feature', 'Course enrolled by the learner.'),
    ('login_frequency_per_week', str(df['login_frequency_per_week'].dtype), 'Numerical behavioral feature', 'Weekly login frequency as a learner activity measure.'),
    ('avg_session_duration_min', str(df['avg_session_duration_min'].dtype), 'Numerical behavioral feature', 'Average session duration in minutes as an engagement behavior indicator.'),
    ('video_completion_pct', str(df['video_completion_pct'].dtype), 'Numerical behavioral feature', 'Percentage of video content completed by the learner.'),
    ('quiz_attempts', str(df['quiz_attempts'].dtype), 'Numerical behavioral feature', 'Number of quiz attempts as assessment-engagement behavior.'),
    ('forum_posts', str(df['forum_posts'].dtype), 'Numerical behavioral feature', 'Number of forum posts as collaborative engagement behavior.'),
    ('weekend_activity_pct', str(df['weekend_activity_pct'].dtype), 'Numerical behavioral feature', 'Percentage of learner activity performed on weekends.'),
    ('preferred_time_slot', str(df['preferred_time_slot'].dtype), 'Categorical feature', 'Preferred time-of-day activity slot.'),
    ('engagement_tier', str(df['engagement_tier'].dtype), 'Target', 'Engagement class label (Low/Medium/High) for future classification.'),
    ('completed_course', str(df['completed_course'].dtype), 'Outcome / potential leakage feature', 'Course completion outcome that may occur after observed activity and may introduce leakage depending on prediction timing.')
]

for col, dtype, role, desc in data_dictionary:
    print(f'| {col} | {dtype} | {role} | {desc} |')

# 14. Potential Target and Leakage Considerations
print('\n## 14. Potential Target and Leakage Considerations')
print('- engagement_tier is the classification target and must not be used as an input feature when predicting engagement_tier.')
print('- completed_course is treated as an outcome/potential leakage feature for now; it is not removed in Step 1 and its modeling role will be decided later based on prediction scenario and timing.')

# 15. Step 1 Findings
print('\n## 15. Step 1 Findings')
print('STEP 1 FINDINGS')
print(f'- Dataset size: {rows} rows and {cols} columns.')
print(f'- Unique learners (student_id): {unique_student_ids}.')
print(f'- Missing-value issue: total missing values = {total_missing}; forum_posts missing = {int(missing_by_column.get("forum_posts", 0))}.')
print(f'- Exact duplicate rows: {exact_duplicate_rows}.')
print(f'- Repeated student IDs: {len(duplicate_student_id_counts)} student_id values appear more than once.')
print('- Engagement-tier distribution:')
for tier, count in engagement_counts.items():
    print(f'  - {tier}: {count}')
print(f'- Main numerical behavioral variables: {", ".join(numerical_columns)}.')
print(f'- Main categorical variables: {", ".join(categorical_columns)}.')
print('- Target/leakage considerations: engagement_tier is target-only; completed_course is a potential leakage/outcome field pending scenario-based decision.')
print('- Step 2 attention items: handle missing forum_posts values, define duplicate-row strategy, define repeated-student record handling policy, and verify final feature eligibility under leakage constraints.')
