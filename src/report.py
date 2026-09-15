from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUTPUTS_DIR = ROOT / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / 'results.csv')

def reasons(row):
    r = []
    if row['off_hours'] == 1:
        r.append('outside working hours')
    if abs(row['amount_zscore']) > 2:
        r.append('amount far from employee average')
    if row['discount_pct'] >= 20:
        r.append('unusually high discount')
    if row['is_cash'] == 1 and row['amount'] > 500:
        r.append('large cash payment')
    return ' | '.join(r) if r else 'unusual combination of values'

flagged = df[df['flagged'] == 1].copy()
flagged['reason'] = flagged.apply(reasons, axis=1)
flagged = flagged.sort_values('amount', ascending=False)

cols = ['invoice_id', 'date', 'hour', 'amount', 'employee', 'payment', 'discount_pct', 'reason']
print(flagged[cols].to_string(index=False))

flagged[cols].to_csv(OUTPUTS_DIR / 'flagged_invoices.csv', index=False)

summary = pd.crosstab(df['fraud_level'], df['flagged'], normalize='index') * 100
summary = summary.reindex(['normal', 'soft', 'medium', 'hard'])
summary.plot(kind='bar', figsize=(8, 5), color=['#cccccc', '#d94f4f'])
plt.title('Detection rate by fraud level')
plt.xlabel('Fraud level')
plt.ylabel('Percentage (%)')
plt.legend(['Not flagged', 'Flagged'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'detection_results.png')
plt.show()