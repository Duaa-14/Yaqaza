from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

df = pd.read_csv(DATA_DIR / 'invoices.csv')

df['is_cash'] = (df['payment'] == 'cash').astype(int)
df['off_hours'] = ((df['hour'] < 10) | (df['hour'] > 21)).astype(int)

emp_mean = df.groupby('employee')['amount'].transform('mean')
emp_std = df.groupby('employee')['amount'].transform('std')
df['amount_zscore'] = (df['amount'] - emp_mean) / emp_std

features = ['amount', 'hour', 'discount_pct', 'is_cash', 'off_hours', 'amount_zscore']
X = df[features]

X_scaled = StandardScaler().fit_transform(X)

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_score'] = model.fit_predict(X_scaled)
df['flagged'] = (df['anomaly_score'] == -1).astype(int)
df.loc[df['amount_zscore'] < 0, 'flagged'] = 0

print(pd.crosstab(df['fraud_level'], df['flagged']))

caught = df[(df['is_fraud'] == 1) & (df['flagged'] == 1)].shape[0]
print(f"\ncaught: {caught} / 40")

df.to_csv(DATA_DIR / 'results.csv', index=False)