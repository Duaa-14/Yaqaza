from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
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

train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['is_fraud'])

scaler = StandardScaler().fit(train[features])
X_train = scaler.transform(train[features])
X_test = scaler.transform(test[features])
y_test = test['is_fraud'].values

rate = 0.05

models = {
    'Elliptic Envelope': EllipticEnvelope(contamination=rate, random_state=42),
    'Isolation Forest': IsolationForest(contamination=rate, random_state=42),
    'One-Class SVM': OneClassSVM(nu=rate, kernel='rbf', gamma='scale'),
}

rows = []

for name, model in models.items():
    model.fit(X_train)
    pred = (model.predict(X_test) == -1).astype(int)
    pred[test['amount_zscore'].values < 0] = 0

    rows.append({
        'Model': name,
        'Caught': int(((pred == 1) & (y_test == 1)).sum()),
        'Actual frauds': int(y_test.sum()),
        'False Alarms': int(((pred == 1) & (y_test == 0)).sum()),
        'Precision': round(precision_score(y_test, pred, zero_division=0), 3),
        'Recall': round(recall_score(y_test, pred), 3),
        'F1': round(f1_score(y_test, pred), 3),
    })

results = pd.DataFrame(rows).sort_values('F1', ascending=False)
print(results.to_string(index=False))
results.to_csv(OUTPUTS_DIR / 'evaluation_holdout.csv', index=False)