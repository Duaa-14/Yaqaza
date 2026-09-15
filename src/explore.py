from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUTPUTS_DIR = ROOT / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / 'invoices.csv')

print(df.describe())
print(df.groupby('employee')['amount'].agg(['count', 'mean', 'max']))
print(df.groupby('payment')['discount_pct'].mean())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df['amount'], bins=40)
axes[0].set_title('Amount')
axes[1].hist(df['hour'], bins=24)
axes[1].set_title('Hour')
axes[2].hist(df['discount_pct'], bins=20)
axes[2].set_title('Discount %')
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'exploration.png')
plt.show()