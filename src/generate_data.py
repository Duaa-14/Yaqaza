from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)  #

n = 800  #

# ------
data = pd.DataFrame({
    'invoice_id': range(1000, 1000 + n),
    'date': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.random.randint(0, 240, n), unit='D'),
    'hour': np.random.choice(range(10, 22), n),        # 
    'amount': np.round(np.random.normal(250, 90, n), 2),  #
    'employee': np.random.choice(['E1', 'E2', 'E3'], n),
    'payment': np.random.choice(['cash', 'card'], n, p=[0.4, 0.6]),
    'discount_pct': np.random.choice([0, 5, 10], n, p=[0.7, 0.2, 0.1]),
})
data['is_fraud'] = 0   # 
fraud_hard = pd.DataFrame({
    'invoice_id': range(9000, 9015),
    'date': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.random.randint(0, 240, 15), unit='D'),
    'hour': np.random.choice([2, 3, 4, 23], 15),
    'amount': np.round(np.random.uniform(900, 2000, 15), 2),
    'employee': np.random.choice(['E2', 'E3'], 15),
    'payment': 'cash',
    'discount_pct': np.random.choice([40, 50, 60], 15),
})

fraud_medium = pd.DataFrame({
    'invoice_id': range(9015, 9030),
    'date': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.random.randint(0, 240, 15), unit='D'),
    'hour': np.random.choice(range(10, 22), 15),
    'amount': np.round(np.random.uniform(500, 800, 15), 2),
    'employee': 'E2',
    'payment': 'cash',
    'discount_pct': np.random.choice([20, 25, 30], 15),
})

fraud_soft = pd.DataFrame({
    'invoice_id': range(9030, 9040),
    'date': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.random.randint(0, 240, 10), unit='D'),
    'hour': np.random.choice(range(10, 22), 10),
    'amount': np.round(np.random.uniform(350, 500, 10), 2),
    'employee': 'E2',
    'payment': 'cash',
    'discount_pct': np.random.choice([15, 20], 10),
})

fraud = pd.concat([fraud_hard, fraud_medium, fraud_soft], ignore_index=True)
fraud['is_fraud'] = 1
fraud['fraud_level'] = ['hard'] * 15 + ['medium'] * 15 + ['soft'] * 10
data['fraud_level'] = 'normal'

df = pd.concat([data, fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(DATA_DIR / 'invoices.csv', index=False)
print(df.shape)
print(df['is_fraud'].value_counts())
print(df.head()) 