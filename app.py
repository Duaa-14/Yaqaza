import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Invoice Fraud Detector", layout="wide")
st.title("AI Invoice Fraud Detector")

REQUIRED = ['invoice_id', 'date', 'hour', 'amount', 'employee', 'payment', 'discount_pct']

if 'manual' not in st.session_state:
    st.session_state.manual = []

src1, src2 = st.tabs(["Upload file", "Manual entry"])

df = None

with src1:
    template = pd.DataFrame(columns=REQUIRED)
    st.download_button("Download empty template", template.to_csv(index=False), "invoice_template.csv")
    st.caption("date: YYYY-MM-DD | hour: 0-23 | payment: cash or card | discount_pct: 0-100")

    uploaded = st.file_uploader("Upload invoices (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded:
        if uploaded.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)

        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            df = None

with src2:
    c1, c2, c3 = st.columns(3)
    inv_id = c1.number_input("Invoice ID", min_value=1, step=1)
    inv_date = c2.date_input("Date")
    inv_hour = c3.number_input("Hour", min_value=0, max_value=23, step=1)

    c4, c5, c6 = st.columns(3)
    inv_amount = c4.number_input("Amount", min_value=0.0, step=10.0)
    inv_emp = c5.text_input("Employee")
    inv_pay = c6.selectbox("Payment", ["cash", "card"])

    inv_disc = st.number_input("Discount %", min_value=0, max_value=100, step=5)

    if st.button("Add invoice"):
        st.session_state.manual.append({
            'invoice_id': inv_id,
            'date': inv_date.strftime('%Y-%m-%d'),
            'hour': inv_hour,
            'amount': inv_amount,
            'employee': inv_emp,
            'payment': inv_pay,
            'discount_pct': inv_disc,
        })

    if st.session_state.manual:
        manual_df = pd.DataFrame(st.session_state.manual)
        st.dataframe(manual_df, width='stretch')

        c7, c8 = st.columns(2)
        c7.download_button("Download as CSV", manual_df.to_csv(index=False), "manual_invoices.csv")
        if c8.button("Clear all"):
            st.session_state.manual = []
            st.rerun()

        if df is None:
            df = manual_df.copy()

if df is not None and len(df) > 0:
    st.divider()

    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['is_cash'] = (df['payment'] == 'cash').astype(int)
    df['off_hours'] = ((df['hour'] < 10) | (df['hour'] > 21)).astype(int)

    emp_mean = df.groupby('employee')['amount'].transform('mean')
    emp_std = df.groupby('employee')['amount'].transform('std')
    df['amount_zscore'] = ((df['amount'] - emp_mean) / emp_std).fillna(0)

    if len(df) < 30:
        st.info(f"Only {len(df)} invoices. Results are more reliable with 30 or more.")

    tab1, tab2 = st.tabs(["Invoice level", "Profile level"])

    with tab1:
        rate = st.slider("Sensitivity (%)", 1, 15, 5) / 100

        features = ['amount', 'hour', 'discount_pct', 'is_cash', 'off_hours', 'amount_zscore']
        X_scaled = StandardScaler().fit_transform(df[features])

        model = IsolationForest(contamination=rate, random_state=42)
        df['flagged'] = (model.fit_predict(X_scaled) == -1).astype(int)
        df.loc[df['amount_zscore'] < 0, 'flagged'] = 0

        def reasons(row):
            r = []
            if row['off_hours'] == 1:
                r.append('outside working hours')
            if row['amount_zscore'] > 2:
                r.append('amount far above employee average')
            if row['discount_pct'] >= 20:
                r.append('unusually high discount')
            if row['is_cash'] == 1 and row['amount'] > 500:
                r.append('large cash payment')
            return ' | '.join(r) if r else 'unusual combination of values'

        flagged = df[df['flagged'] == 1].copy()
        flagged['reason'] = flagged.apply(reasons, axis=1)
        flagged = flagged.sort_values('amount', ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total invoices", len(df))
        c2.metric("Flagged", len(flagged))
        c3.metric("Flag rate", f"{len(flagged)/len(df)*100:.1f}%")

        cols = ['invoice_id', 'date', 'hour', 'amount', 'employee', 'payment', 'discount_pct', 'reason']
        st.dataframe(flagged[cols], width='stretch')

        st.download_button("Download results", flagged[cols].to_csv(index=False), "flagged_invoices.csv")

    with tab2:
        group_col = st.selectbox("Analyse by", ['employee', 'month'])

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

        st.dataframe(prof, width='stretch')
        st.bar_chart(prof['risk_score'])

        risky = prof[prof['flagged'] == 1]
        if len(risky) > 0:
            st.warning(f"High risk: {', '.join(risky.index.astype(str))}")
        else:
            st.success("No high-risk profiles detected")