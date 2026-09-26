"""
Step 3 - Baseline classification script
Reads: data/cleaned_learner_segmentation.csv
Writes: reports/step3_baseline_classification_report.md
Produces: reports/figures/*.png and metrics_comparison.csv

Models: Logistic Regression, KNN, Naive Bayes
Train/test: 80/20 stratified by engagement_tier, random_state=42
Excluded features: student_id, student_name, engagement_tier, completed_course
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

CLEANED = Path('data/cleaned_learner_segmentation.csv')
FIG_DIR = Path('reports/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT = Path('reports/step3_baseline_classification_report.md')

if __name__ == '__main__':
    df = pd.read_csv(CLEANED)
    exclude = ['student_id','student_name','engagement_tier','completed_course']
    X = df.drop(columns=exclude)
    y = df['engagement_tier']

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', cat_transformer, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(),
        'NaiveBayes': GaussianNB()
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[('pre', preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=['Low','Medium','High'])
        cls_report = classification_report(y_test, y_pred, zero_division=0)
        results[name] = {
            'accuracy': acc,
            'precision_macro': prec_macro,
            'recall_macro': rec_macro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'classification_report': cls_report
        }
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low','Medium','High'])
        fig, ax = plt.subplots(figsize=(5,4))
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        ax.set_title(f'{name} Confusion Matrix')
        fig_path = FIG_DIR / f'{name.lower()}_confusion.png'
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

    metrics_df = pd.DataFrame([{ 'model': name,
                                  'accuracy': r['accuracy'],
                                  'precision_macro': r['precision_macro'],
                                  'recall_macro': r['recall_macro'],
                                  'f1_macro': r['f1_macro'],
                                  'f1_weighted': r['f1_weighted'] } for name,r in results.items()])
    metrics_df.to_csv(FIG_DIR / 'metrics_comparison.csv', index=False)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('w', encoding='utf-8') as f:
        f.write('# Step 3 — Baseline Classification Report\n\n')
        f.write('Dataset: data/cleaned_learner_segmentation.csv\n\n')
        f.write('Train/test split: 80/20 stratified by engagement_tier, random_state=42\n\n')
        f.write('Excluded features from modeling: student_id, student_name, engagement_tier, completed_course\n\n')
        f.write('## Models evaluated\n')
        for name in results:
            f.write(f'- {name}\n')
        f.write('\n## Metrics\n')
        f.write(metrics_df.to_string(index=False))
        f.write('\n\n## Confusion matrices saved in reports/figures/\n')
        for name in results:
            f.write(f'- {name}: reports/figures/{name.lower()}_confusion.png\n')
        f.write('\n\n## Detailed classification reports\n')
        for name,r in results.items():
            f.write(f'### {name}\n\n')
            f.write('```\n')
            f.write(r['classification_report'])
            f.write('\n```\n')

    print('STEP3_COMPLETE')
    print('Train size:', X_train.shape)
    print('Test size:', X_test.shape)
    print('Metrics saved to', FIG_DIR / 'metrics_comparison.csv')
