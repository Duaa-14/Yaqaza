"""Shared fraud-detection logic, mirrors the original app.py exactly."""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Canonical schema used internally once a file's columns are mapped, and the schema manual entries already use.
REQUIRED_COLUMNS = ['invoice_id', 'date', 'hour', 'amount', 'employee', 'payment', 'discount_pct']

# Fields the user maps an uploaded file's own columns onto. invoice_id is never mapped -- it's
# always assigned as a running sequence, since the user is never asked to identify an ID column.
MAPPING_FIELDS = [
    {'key': 'amount', 'label': 'المبلغ', 'required': True},
    {'key': 'date', 'label': 'التاريخ', 'required': True},
    {'key': 'hour', 'label': 'الساعة', 'required': True},
    {'key': 'employee', 'label': 'الموظف', 'required': False},
    {'key': 'payment', 'label': 'طريقة الدفع', 'required': False},
    {'key': 'discount_pct', 'label': 'نسبة الخصم', 'required': False},
]

# Literal column-name matches only (English name or a known Arabic synonym) -- never based on
# column content. Used to pre-fill the mapping screen; the user can always change any selection.
COLUMN_NAME_SYNONYMS = {
    'amount': ['amount', 'المبلغ', 'الإجمالي', 'المبلغ الإجمالي', 'القيمة', 'السعر', 'اجمالي الفاتورة'],
    'date': ['date', 'التاريخ', 'تاريخ الفاتورة', 'اليوم'],
    'hour': ['hour', 'الساعة', 'الوقت', 'وقت الفاتورة'],
    'employee': ['employee', 'الموظف', 'البائع', 'اسم الموظف', 'الكاشير', 'المستخدم'],
    'payment': ['payment', 'طريقة الدفع', 'الدفع', 'نوع الدفع', 'وسيلة الدفع'],
    'discount_pct': ['discount_pct', 'الخصم', 'نسبة الخصم', 'الخصومات'],
}


def _normalize_column_name(name):
    return ' '.join(str(name).strip().lower().split())


def auto_match_columns(columns):
    """Pre-fill the mapping screen using exact (case/whitespace-insensitive) name matches
    against the expected English field name or a known Arabic synonym -- not a content-based
    guess. Each source column is used for at most one field; unmatched fields are left None
    for the user to pick themselves.
    """
    by_normalized_name = {}
    for col in columns:
        key = _normalize_column_name(col)
        if key not in by_normalized_name:
            by_normalized_name[key] = col

    used = set()
    matches = {}
    for field in MAPPING_FIELDS:
        matched = None
        for synonym in COLUMN_NAME_SYNONYMS.get(field['key'], []):
            candidate = by_normalized_name.get(_normalize_column_name(synonym))
            if candidate and candidate not in used:
                matched = candidate
                break
        matches[field['key']] = matched
        if matched:
            used.add(matched)
    return matches


def apply_mapping(raw_df, mapping, start_id=1):
    """Build a standardized invoice dataframe from a raw upload using a user-chosen column mapping.

    mapping: dict of canonical field key -> source column name (or '' / None if left unmapped).
    Returns (standardized_df, flags) where flags says which optional fields were actually provided.
    Raises ValueError listing missing required fields.
    """
    missing = [f['label'] for f in MAPPING_FIELDS if f['required'] and not mapping.get(f['key'])]
    if missing:
        raise ValueError('لم يتم تحديد الأعمدة الإلزامية التالية: ' + '، '.join(missing))

    out = pd.DataFrame(index=raw_df.index)
    out['invoice_id'] = range(start_id, start_id + len(raw_df))
    out['amount'] = pd.to_numeric(raw_df[mapping['amount']], errors='coerce')
    out['date'] = raw_df[mapping['date']]
    out['hour'] = pd.to_numeric(raw_df[mapping['hour']], errors='coerce')

    flags = {}

    if mapping.get('employee'):
        out['employee'] = raw_df[mapping['employee']].astype(str)
        flags['has_employee'] = True
    else:
        out['employee'] = 'الكل'
        flags['has_employee'] = False

    if mapping.get('payment'):
        out['payment'] = raw_df[mapping['payment']].astype(str).str.strip().str.lower()
        flags['has_payment'] = True
    else:
        out['payment'] = 'card'
        flags['has_payment'] = False

    if mapping.get('discount_pct'):
        out['discount_pct'] = pd.to_numeric(raw_df[mapping['discount_pct']], errors='coerce').fillna(0)
        flags['has_discount'] = True
    else:
        out['discount_pct'] = 0
        flags['has_discount'] = False

    return out[REQUIRED_COLUMNS], flags


def add_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['is_cash'] = (df['payment'] == 'cash').astype(int)
    df['off_hours'] = ((df['hour'] < 10) | (df['hour'] > 21)).astype(int)

    emp_mean = df.groupby('employee')['amount'].transform('mean')
    emp_std = df.groupby('employee')['amount'].transform('std')
    df['amount_zscore'] = ((df['amount'] - emp_mean) / emp_std).fillna(0)
    return df


def _reasons(row):
    r = []
    if row['off_hours'] == 1:
        r.append('خارج ساعات العمل')
    if row['amount_zscore'] > 2:
        r.append('المبلغ أعلى بكثير من متوسط الموظف')
    if row['discount_pct'] >= 20:
        r.append('خصم مرتفع بشكل غير معتاد')
    if row['is_cash'] == 1 and row['amount'] > 500:
        r.append('دفعة نقدية كبيرة')
    return ' | '.join(r) if r else 'تركيبة قيم غير معتادة'


def detect_invoice_fraud(df, rate):
    """rate: contamination fraction, e.g. 0.05 for 5%."""
    df = df.copy()
    features = ['amount', 'hour', 'discount_pct', 'is_cash', 'off_hours', 'amount_zscore']
    X_scaled = StandardScaler().fit_transform(df[features])

    model = IsolationForest(contamination=rate, random_state=42)
    df['flagged'] = (model.fit_predict(X_scaled) == -1).astype(int)
    df.loc[df['amount_zscore'] < 0, 'flagged'] = 0

    flagged = df[df['flagged'] == 1].copy()
    flagged['reason'] = flagged.apply(_reasons, axis=1)
    flagged = flagged.sort_values('amount', ascending=False)
    return df, flagged


def build_profile(df, group_col):
    prof = df.groupby(group_col).agg(
        invoice_count=('amount', 'count'),
        avg_amount=('amount', 'mean'),
        avg_discount=('discount_pct', 'mean'),
        cash_ratio=('is_cash', 'mean'),
        off_hours_ratio=('off_hours', 'mean'),
        high_discount_ratio=('discount_pct', lambda x: (x >= 20).mean()),
    ).round(3)

    zcols = ['avg_amount', 'avg_discount', 'cash_ratio', 'off_hours_ratio', 'high_discount_ratio']
    z = ((prof[zcols] - prof[zcols].mean()) / prof[zcols].std()).fillna(0)
    prof['risk_score'] = z.clip(lower=0).sum(axis=1).round(2)
    prof['flagged'] = (prof['risk_score'] > 2).astype(int)
    prof = prof.sort_values('risk_score', ascending=False)
    return prof
