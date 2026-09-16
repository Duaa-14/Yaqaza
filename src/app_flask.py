import io
import json
import os
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, session, send_file, redirect, url_for

from analysis import (
    REQUIRED_COLUMNS, MAPPING_FIELDS, apply_mapping, auto_match_columns,
    add_features, detect_invoice_fraud, build_profile,
)

# templates/ and static/ live at the project root, as siblings of src/, not inside it.
ROOT_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(ROOT_DIR / 'templates'),
    static_folder=str(ROOT_DIR / 'static'),
)
app.secret_key = 'invoice-fraud-detector-local-dev-key'

# In-memory per-session result store (single-user local tool, so this is fine).
STORE = {}


def get_store():
    sid = session.get('sid')
    if not sid or sid not in STORE:
        sid = str(uuid.uuid4())
        session['sid'] = sid
        STORE[sid] = {}
    return STORE[sid]


def read_uploaded_file(file_storage):
    filename = file_storage.filename or ''
    if filename.lower().endswith('.xlsx'):
        return pd.read_excel(file_storage)
    return pd.read_csv(file_storage)


def _sync_settings_from_form(store):
    manual_json = request.form.get('manual_rows_json', '[]')
    try:
        store['manual_rows'] = json.loads(manual_json) if manual_json else []
    except (ValueError, TypeError):
        store['manual_rows'] = []

    rate = int(request.form.get('rate', 5) or 5)
    store['rate'] = min(max(rate, 1), 15)

    group_col = request.form.get('group_col', 'employee')
    store['group_col'] = group_col if group_col in ('employee', 'month') else 'employee'


@app.route('/')
def index():
    store = get_store()
    return render_template(
        'index.html',
        manual_rows=store.get('manual_rows', []),
        rate=store.get('rate', 5),
        group_col=store.get('group_col', 'employee'),
        results=store.get('results'),
        profile=store.get('profile'),
        error=store.get('error'),
        active_tab=request.args.get('tab', 'invoice'),
        has_data=bool(store.get('combined_csv')),
        pending_columns=store.get('pending_columns'),
        pending_filename=store.get('pending_filename'),
        mapping_fields=MAPPING_FIELDS,
        mapping_selection=store.get('mapping_selection', {}),
        has_employee=store.get('has_employee', True),
    )


@app.route('/template')
def download_template():
    template = pd.DataFrame(columns=REQUIRED_COLUMNS)
    buf = io.BytesIO(template.to_csv(index=False).encode('utf-8-sig'))
    return send_file(buf, mimetype='text/csv', as_attachment=True, download_name='invoice_template.csv')


@app.route('/upload/preview', methods=['POST'])
def upload_preview():
    store = get_store()
    _sync_settings_from_form(store)

    uploaded = request.files.get('file')
    if not uploaded or not uploaded.filename:
        store['error'] = 'الرجاء اختيار ملف أولاً.'
        return redirect(url_for('index'))

    try:
        raw_df = read_uploaded_file(uploaded)
    except Exception as exc:
        store['error'] = f'تعذّرت قراءة الملف: {exc}'
        return redirect(url_for('index'))

    if len(raw_df.columns) == 0:
        store['error'] = 'الملف المرفوع لا يحتوي على أعمدة يمكن مطابقتها.'
        return redirect(url_for('index'))

    columns = [str(c) for c in raw_df.columns]
    store['pending_raw_csv'] = raw_df.to_csv(index=False)
    store['pending_columns'] = columns
    store['pending_filename'] = uploaded.filename
    store['mapping_selection'] = auto_match_columns(columns)
    store['error'] = None

    return redirect(url_for('index'))


@app.route('/analyze', methods=['POST'])
def analyze():
    store = get_store()
    _sync_settings_from_form(store)
    manual_rows = store['manual_rows']
    rate = store['rate']
    group_col = store['group_col']

    frames = []
    has_employee_flag = bool(manual_rows)  # manual entries always carry a real employee value

    pending_csv = store.get('pending_raw_csv')
    file_selected_but_not_mapped = not pending_csv and request.files.get('file') and request.files['file'].filename

    if pending_csv:
        mapping = {f['key']: request.form.get(f'map_{f["key"]}', '').strip() for f in MAPPING_FIELDS}
        store['mapping_selection'] = mapping

        try:
            raw_df = pd.read_csv(io.StringIO(pending_csv))
            mapped_df, flags = apply_mapping(raw_df, mapping)
        except ValueError as exc:
            store['error'] = str(exc)
            store['results'] = None
            store['profile'] = None
            return redirect(url_for('index'))
        except Exception as exc:
            store['error'] = f'تعذّر تطبيق المطابقة: {exc}'
            store['results'] = None
            store['profile'] = None
            return redirect(url_for('index'))

        frames.append(mapped_df)
        has_employee_flag = has_employee_flag or flags['has_employee']
        store.pop('pending_raw_csv', None)
        store.pop('pending_columns', None)
        store.pop('pending_filename', None)
        store.pop('mapping_selection', None)
    elif file_selected_but_not_mapped:
        store['error'] = 'الرجاء الضغط على "رفع ومطابقة الأعمدة" أولاً لتحديد أعمدة الملف قبل التحليل.'
        store['results'] = None
        store['profile'] = None
        return redirect(url_for('index'))

    if manual_rows:
        frames.append(pd.DataFrame(manual_rows)[REQUIRED_COLUMNS])

    if not frames:
        store['error'] = 'الرجاء رفع ملف أو إضافة فواتير يدويًا قبل التحليل.'
        store['results'] = None
        store['profile'] = None
        return redirect(url_for('index'))

    if not has_employee_flag and group_col == 'employee':
        group_col = 'month'
    store['group_col'] = group_col
    store['has_employee'] = has_employee_flag

    df = pd.concat(frames, ignore_index=True)
    store['error'] = None
    store['combined_csv'] = df.to_csv(index=False)

    _run_invoice_level(store, df, rate)
    _run_profile_level(store, df, group_col)

    return redirect(url_for('index'))


def _run_invoice_level(store, raw_df, rate):
    df = add_features(raw_df)
    _, flagged = detect_invoice_fraud(df, rate / 100)

    cols = ['invoice_id', 'date', 'hour', 'amount', 'employee', 'payment', 'discount_pct', 'reason']
    flagged_display = flagged[cols].copy()
    flagged_display['date'] = flagged_display['date'].dt.strftime('%Y-%m-%d')

    payment_ar = {'cash': 'نقدي', 'card': 'بطاقة'}
    rows = flagged_display.to_dict(orient='records')
    for row in rows:
        row['payment_ar'] = payment_ar.get(row['payment'], row['payment'])
        row['reason_tags'] = row['reason'].split(' | ')

    results = {
        'total': len(df),
        'flagged_count': len(flagged),
        'flag_rate': round(len(flagged) / len(df) * 100, 1) if len(df) else 0,
        'low_data_warning': len(df) < 30,
        'rows': rows,
    }
    store['results'] = results
    store['flagged_csv'] = flagged_display.to_csv(index=False)


def _run_profile_level(store, raw_df, group_col):
    df = add_features(raw_df)
    prof = build_profile(df, group_col)
    profile = {
        'group_col': group_col,
        'columns': ['invoice_count', 'avg_amount', 'avg_discount', 'cash_ratio',
                    'off_hours_ratio', 'high_discount_ratio', 'risk_score', 'flagged'],
        'rows': [{'name': idx, **row} for idx, row in prof.to_dict(orient='index').items()],
        'risky': prof[prof['flagged'] == 1].index.astype(str).tolist(),
    }
    store['profile'] = profile
    store['profile_csv'] = prof.to_csv()


@app.route('/reanalyze/<int:rate>')
def reanalyze(rate):
    store = get_store()
    raw_csv = store.get('combined_csv')
    if not raw_csv:
        return redirect(url_for('index'))
    rate = min(max(rate, 1), 15)
    store['rate'] = rate
    df = pd.read_csv(io.StringIO(raw_csv))
    _run_invoice_level(store, df, rate)
    return redirect(url_for('index', tab='invoice'))


@app.route('/profile/<group_col>')
def switch_profile(group_col):
    store = get_store()
    if group_col not in ('employee', 'month'):
        group_col = 'employee'
    if group_col == 'employee' and not store.get('has_employee', True):
        group_col = 'month'
    store['group_col'] = group_col
    raw_csv = store.get('combined_csv')
    if not raw_csv:
        return redirect(url_for('index'))
    df = pd.read_csv(io.StringIO(raw_csv))
    _run_profile_level(store, df, group_col)
    return redirect(url_for('index', tab='profile'))


@app.route('/download/flagged')
def download_flagged():
    store = get_store()
    csv_data = store.get('flagged_csv')
    if not csv_data:
        return redirect(url_for('index'))
    buf = io.BytesIO(csv_data.encode('utf-8-sig'))
    return send_file(buf, mimetype='text/csv', as_attachment=True, download_name='flagged_invoices.csv')


@app.route('/download/profile')
def download_profile():
    store = get_store()
    csv_data = store.get('profile_csv')
    if not csv_data:
        return redirect(url_for('index'))
    buf = io.BytesIO(csv_data.encode('utf-8-sig'))
    return send_file(buf, mimetype='text/csv', as_attachment=True, download_name='profile_analysis.csv')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
