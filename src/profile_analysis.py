from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

df = pd.read_csv(DATA_DIR / 'results.csv')
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M').astype(str)

def build_profiles(df, group_col):
    p = df.groupby(group_col).agg(
        invoice_count=('amount', 'count'),
        avg_amount=('amount', 'mean'),
        std_amount=('amount', 'std'),
        avg_discount=('discount_pct', 'mean'),
        max_discount=('discount_pct', 'max'),
        cash_ratio=('is_cash', 'mean'),
        off_hours_ratio=('off_hours', 'mean'),
        high_discount_ratio=('discount_pct', lambda x: (x >= 20).mean()),
    ).round(3)
    return p

def detect(profiles):
    cols = ['avg_amount', 'avg_discount', 'cash_ratio', 'off_hours_ratio', 'high_discount_ratio']
    z = (profiles[cols] - profiles[cols].mean()) / profiles[cols].std()
    out = profiles.copy()
    out['risk_score'] = z.clip(lower=0).sum(axis=1).round(2)
    out['flagged'] = (out['risk_score'] > 2).astype(int)
    return out.sort_values('risk_score', ascending=False)

for col in ['employee', 'month']:
    print(f"\n===== {col.upper()} =====")
    prof = build_profiles(df, col)
    result = detect(prof)
    print(result[['invoice_count', 'avg_amount', 'avg_discount', 'cash_ratio',
                  'off_hours_ratio', 'high_discount_ratio', 'risk_score', 'flagged']].to_string())