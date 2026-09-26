"""
Step 2 - Data cleaning script
Reads: data/schoolliant_learner_segmentation.csv
Writes: data/cleaned_learner_segmentation.csv
Writes: reports/step2_data_cleaning_report.md

Cleaning rules (documented):
- Remove exact duplicate rows
- Impute missing forum_posts with 0 (defensible for a count field)
- Resolve repeated student_id by aggregation:
  - numeric features: mean
  - categorical features: mode
  - completed_course: 'Yes' if any record is Yes, else 'No'

This script is non-destructive to the raw CSV.
"""

import pandas as pd
from pathlib import Path

RAW = Path('data/schoolliant_learner_segmentation.csv')
OUT = Path('data/cleaned_learner_segmentation.csv')
REPORT = Path('reports/step2_data_cleaning_report.md')

if __name__ == '__main__':
    df = pd.read_csv(RAW)
    orig_shape = df.shape

    # Initial checks
    exact_duplicates = int(df.duplicated().sum())
    dup_ids = df['student_id'][df['student_id'].duplicated(keep=False)].unique()
    conflicting = {sid: df[df['student_id'] == sid] for sid in dup_ids}

    # Remove exact duplicate rows
    df_nodup = df.drop_duplicates()

    # Impute forum_posts missing values with 0 (count)
    forum_missing_before = int(df_nodup['forum_posts'].isna().sum()) if 'forum_posts' in df_nodup.columns else 0
    if 'forum_posts' in df_nodup.columns:
        df_nodup['forum_posts'] = df_nodup['forum_posts'].fillna(0)

    # Aggregation rules for repeated student_id
    num_cols = [c for c in df_nodup.select_dtypes(include='number').columns]
    cat_cols = [c for c in df_nodup.select_dtypes(include=['object', 'category', 'bool']).columns if c != 'student_id']

    agg_funcs = {}
    for c in num_cols:
        agg_funcs[c] = 'mean'
    for c in cat_cols:
        if c == 'completed_course':
            agg_funcs[c] = lambda x: 'Yes' if (x.astype(str).str.strip().eq('Yes').any()) else 'No'
        else:
            agg_funcs[c] = lambda x: x.mode().iat[0] if not x.mode().empty else (x.dropna().iat[0] if not x.dropna().empty else x.iat[0])

    cleaned = df_nodup.groupby('student_id', as_index=False).agg(agg_funcs)

    # Round count-like columns
    for c in ['quiz_attempts', 'forum_posts']:
        if c in cleaned.columns:
            cleaned[c] = cleaned[c].round().astype(int)

    cleaned_shape = cleaned.shape
    cleaned.to_csv(OUT, index=False)

    # Write report
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('w', encoding='utf-8') as f:
        f.write('# Step 2 — Data Cleaning Report\n\n')
        f.write('Source file: data/schoolliant_learner_segmentation.csv\n\n')
        f.write(f'Original shape: {orig_shape[0]} rows, {orig_shape[1]} columns\n\n')
        f.write('Cleaning actions applied:\n')
        f.write(f'- Removed exact duplicate rows ({exact_duplicates})\n')
        f.write(f'- Imputed missing forum_posts with 0 (forum_posts missing before: {forum_missing_before})\n')
        f.write('- Resolved repeated student_id records by aggregating: numeric->mean, categorical->mode, completed_course->Yes if any\n\n')
        f.write('Results:\n')
        f.write(f'- Cleaned shape: {cleaned_shape[0]} rows, {cleaned_shape[1]} columns\n\n')
        f.write('Conflicting student_id records inspected (showing rows before cleaning):\n\n')
        for sid, rows in conflicting.items():
            f.write(f'--- {sid} ---\n')
            f.write(rows.to_csv(index=False))
            f.write('\n')
        f.write('\nNote: Aggregation rules are documented above. If any aggregation decision is not acceptable, request manual review before using the cleaned dataset.')

    print('CLEANING_COMPLETE')
    print('Original shape:', orig_shape)
    print('Exact duplicate rows removed:', exact_duplicates)
    print('Forum posts missing imputed:', forum_missing_before)
    print('Cleaned shape:', cleaned_shape)
