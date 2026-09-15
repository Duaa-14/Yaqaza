from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUTPUTS_DIR = ROOT / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / 'invoices.csv')

df['is_cash'] = (df['payment'] == 'cash').astype(int)
df['off_hours'] = ((df['hour'] < 10) | (df['hour'] > 21)).astype(int)

emp_mean = df.groupby('employee')['amount'].transform('mean')
emp_std = df.groupby('employee')['amount'].transform('std')
df['amount_zscore'] = (df['amount'] - emp_mean) / emp_std

features = ['amount', 'hour', 'discount_pct', 'is_cash', 'off_hours', 'amount_zscore']
X = StandardScaler().fit_transform(df[features])
y = df['is_fraud']

rate = 0.05

models = {
    'Isolation Forest': IsolationForest(contamination=rate, random_state=42),
    'Local Outlier Factor': LocalOutlierFactor(contamination=rate),
    'One-Class SVM': OneClassSVM(nu=rate, kernel='rbf', gamma='scale'),
    'Elliptic Envelope': EllipticEnvelope(contamination=rate, random_state=42),
}

rows = []

for name, model in models.items():
    if name == 'Local Outlier Factor':
        pred = model.fit_predict(X)
    else:
        pred = model.fit(X).predict(X)

    pred = (pred == -1).astype(int)
    pred[df['amount_zscore'] < 0] = 0

    rows.append({
        'Model': name,
        'Caught': int(((pred == 1) & (y == 1)).sum()),
        'False Alarms': int(((pred == 1) & (y == 0)).sum()),
        'Precision': round(precision_score(y, pred), 3),
        'Recall': round(recall_score(y, pred), 3),
        'F1': round(f1_score(y, pred), 3),
    })

results = pd.DataFrame(rows).sort_values('F1', ascending=False)
print(results.to_string(index=False))
results.to_csv(OUTPUTS_DIR / 'model_comparison.csv', index=False)