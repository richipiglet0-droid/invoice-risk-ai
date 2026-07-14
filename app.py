import random
from datetime import date

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Invoice Risk AI",
    page_icon="📊",
    layout="wide",
)


REQUIRED_COLUMNS = [
    "invoice_amount",
    "payment_terms_days",
    "previous_late_payments",
    "total_previous_invoices",
    "paid_late",
]

FEATURE_COLUMNS = [
    "invoice_amount",
    "payment_terms_days",
    "previous_late_payments",
    "total_previous_invoices",
]


def create_demo_data(rows: int = 500) -> pd.DataFrame:
    random.seed(42)
    records = []

    for invoice_id in range(1, rows + 1):
        amount = random.randint(500, 100000)
        terms = random.choice([15, 30, 45, 60])
        previous_total = random.randint(1, 50)
        previous_late = random.randint(0, previous_total)

        risk = previous_late / previous_total

        if amount > 50000:
            risk += 0.15

        if terms >= 45:
            risk += 0.10

        risk += random.uniform(-0.20, 0.20)

        records.append(
            {
                "invoice_id": f"INV-{invoice_id:04d}",
                "customer_name": f"Customer {invoice_id:03d}",
                "invoice_amount": amount,
                "payment_terms_days": terms,
                "previous_late_payments": previous_late,
                "total_previous_invoices": previous_total,
                "paid_late": 1 if risk >= 0.50 else 0,
            }
        )

    return pd.DataFrame(records)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    cleaned = data.copy()

    for column in REQUIRED_COLUMNS:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)

    cleaned = cleaned[
        (cleaned["invoice_amount"] > 0)
        & (cleaned["payment_terms_days"] > 0)
        & (cleaned["total_previous_invoices"] > 0)
        & (cleaned["previous_late_payments"] >= 0)
        & (
            cleaned["previous_late_payments"]
            <= cleaned["total_previous_invoices"]
        )
    ]

    cleaned["paid_late"] = cleaned["paid_late"].astype(int)

    if len(cleaned) < 20:
        raise ValueError("At least 20 valid invoices are required.")

    if cleaned["paid_late"].nunique() < 2:
        raise ValueError(
            "The paid_late column must contain both 0 and 1."
        )

    return cleaned


def train_model(data: pd.DataFrame):
    X = data[FEATURE_COLUMNS]
    y = data["paid_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test),
    )

    return model, accuracy


def risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "HIGH"
    if probability >= 0.40:
        return "MEDIUM"
    return "LOW"


def recommended_action(probability: float) -> str:
    if probability >= 0.70:
        return (
            "Contact the customer today, confirm receipt of the invoice, "
            "ask whether payment is scheduled, and check for disputes."
        )

    if probability >= 0.40:
        return (
            "Send a professional reminder and monitor the invoice "
            "for the next 48 hours."
        )

    return "Continue normal monitoring."


def create_reminder(
    customer_name: str,
    invoice_number: str,
    amount: float,
    due_date: date,
    risk: str,
) -> str:
    days_overdue = max((date.today() - due_date).days, 0)

    urgency = {
        "LOW": "This is a friendly reminder",
        "MEDIUM": "This is a follow-up reminder",
        "HIGH": "This is an important payment follow-up",
    }[risk]

    return f"""Subject: Payment reminder for {invoice_number}

Dear {customer_name},

{urgency} regarding invoice {invoice_number} for ${amount:,.2f}, due on {due_date.strftime("%B %d, %Y")}.

The invoice is currently {days_overdue} day(s) overdue. Could you please confirm whether payment has already been processed?

If there is a dispute, missing document, or other issue preventing payment, please let us know so it can be resolved promptly.

Thank you."""


for key, value in {
    "invoice_data": None,
    "model": None,
    "accuracy": None,
    "ranked": None,
    "prediction": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.title("Invoice Risk AI")

st.write(
    "Identify invoices at risk of late payment, prioritize collection work, "
    "and generate professional payment reminders."
)

st.warning(
    "Prototype decision-support tool. Predictions must be reviewed by a human."
)


with st.sidebar:
    st.header("Controls")

    if st.button("Load demo data", use_container_width=True):
        st.session_state.invoice_data = create_demo_data()
        st.session_state.model = None
        st.session_state.ranked = None

    if st.button("Reset application", use_container_width=True):
        for key in [
            "invoice_data",
            "model",
            "accuracy",
            "ranked",
            "prediction",
        ]:
            st.session_state[key] = None


st.divider()
st.header("1. Load historical invoice data")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        st.session_state.invoice_data = clean_data(
            pd.read_csv(uploaded_file)
        )
        st.session_state.model = None
        st.session_state.ranked = None
        st.success("CSV loaded successfully.")
    except Exception as error:
        st.error(str(error))


data = st.session_state.invoice_data

if data is not None:
    st.divider()
    st.header("2. Invoice portfolio dashboard")

    total_value = data["invoice_amount"].sum()
    late_value = data.loc[
        data["paid_late"] == 1,
        "invoice_amount",
    ].sum()
    late_rate = data["paid_late"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Invoices", f"{len(data):,}")
    col2.metric("Total value", f"${total_value:,.2f}")
    col3.metric("Historically late value", f"${late_value:,.2f}")
    col4.metric("Historical late rate", f"{late_rate:.1%}")

    with st.expander("View historical data"):
        st.dataframe(data, use_container_width=True)

    if st.button("Train payment-risk model", type="primary"):
        try:
            model, accuracy = train_model(data)

            ranked = data.copy()
            ranked["late_payment_probability"] = model.predict_proba(
                ranked[FEATURE_COLUMNS]
            )[:, 1]

            ranked["risk_level"] = ranked[
                "late_payment_probability"
            ].apply(risk_level)

            ranked["recommended_action"] = ranked[
                "late_payment_probability"
            ].apply(recommended_action)

            ranked = ranked.sort_values(
                "late_payment_probability",
                ascending=False,
            )

            st.session_state.model = model
            st.session_state.accuracy = accuracy
            st.session_state.ranked = ranked

            st.success("Model trained successfully.")

        except Exception as error:
            st.error(f"Training failed: {error}")


ranked = st.session_state.ranked

if ranked is not None:
    st.divider()
    st.header("3. Prioritized invoice-risk list")

    high_risk = ranked[ranked["risk_level"] == "HIGH"]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Test accuracy",
        f"{st.session_state.accuracy:.1%}",
    )
    c2.metric("High-risk invoices", f"{len(high_risk):,}")
    c3.metric(
        "High-risk value",
        f"${high_risk['invoice_amount'].sum():,.2f}",
    )

    st.dataframe(
        ranked,
        use_container_width=True,
    )

    st.download_button(
        "Download prioritized list",
        data=ranked.to_csv(index=False).encode("utf-8"),
        file_name="prioritized_invoice_risk.csv",
        mime="text/csv",
    )


model = st.session_state.model

if model is not None:
    st.divider()
    st.header("4. Analyze a new invoice")

    with st.form("prediction_form"):
        amount = st.number_input(
            "Invoice amount",
            min_value=1.0,
            value=10000.0,
        )
        terms = st.number_input(
            "Payment terms in days",
            min_value=1,
            value=30,
        )
        previous_late = st.number_input(
            "Previous late payments",
            min_value=0,
            value=2,
        )
        previous_total = st.number_input(
            "Total previous invoices",
            min_value=1,
            value=10,
        )

        submitted = st.form_submit_button(
            "Calculate payment risk"
        )

    if submitted:
        if previous_late > previous_total:
            st.error(
                "Previous late payments cannot exceed total invoices."
            )
        else:
            invoice = pd.DataFrame(
                [
                    {
                        "invoice_amount": amount,
                        "payment_terms_days": terms,
                        "previous_late_payments": previous_late,
                        "total_previous_invoices": previous_total,
                    }
                ]
            )

            probability = model.predict_proba(invoice)[0][1]

            st.session_state.prediction = {
                "probability": probability,
                "risk": risk_level(probability),
                "action": recommended_action(probability),
            }

    prediction = st.session_state.prediction

    if prediction:
        a, b = st.columns(2)

        a.metric(
            "Late-payment probability",
            f"{prediction['probability']:.1%}",
        )
        b.metric("Risk level", prediction["risk"])

        st.info(prediction["action"])


st.divider()
st.header("5. Generate a payment reminder")

with st.form("reminder_form"):
    customer_name = st.text_input(
        "Customer name",
        value="Example Customer",
    )
    invoice_number = st.text_input(
        "Invoice number",
        value="INV-1001",
    )
    reminder_amount = st.number_input(
        "Invoice amount",
        min_value=0.0,
        value=10000.0,
    )
    due_date = st.date_input("Invoice due date")
    reminder_risk = st.selectbox(
        "Risk level",
        ["LOW", "MEDIUM", "HIGH"],
        index=1,
    )

    create_message = st.form_submit_button(
        "Generate reminder"
    )

if create_message:
    st.text_area(
        "Review and edit before sending",
        value=create_reminder(
            customer_name,
            invoice_number,
            reminder_amount,
            due_date,
            reminder_risk,
        ),
        height=260,
    )


st.divider()

st.caption(
    "Use demo or anonymized data only. Do not upload sensitive customer "
    "information until authentication and security controls are added."
)
