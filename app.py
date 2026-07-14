import random
from datetime import date

import ollama
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# =========================================================
# APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Invoice Risk AI",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# CONSTANTS
# =========================================================

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


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def create_demo_data(number_of_rows: int = 500) -> pd.DataFrame:
    """
    Create synthetic invoice data for testing only.

    This data must not be presented as real business evidence.
    """

    random.seed(42)

    rows = []

    for invoice_id in range(1, number_of_rows + 1):
        invoice_amount = random.randint(500, 100000)
        payment_terms_days = random.choice([15, 30, 45, 60])

        total_previous_invoices = random.randint(1, 50)

        previous_late_payments = random.randint(
            0,
            total_previous_invoices,
        )

        historical_late_rate = (
            previous_late_payments
            / total_previous_invoices
        )

        risk_score = historical_late_rate

        if invoice_amount > 50000:
            risk_score += 0.15

        if payment_terms_days >= 45:
            risk_score += 0.10

        risk_score += random.uniform(-0.20, 0.20)

        paid_late = 1 if risk_score >= 0.50 else 0

        rows.append(
            {
                "invoice_id": f"INV-{invoice_id:04d}",
                "customer_name": f"Customer {invoice_id:03d}",
                "invoice_amount": invoice_amount,
                "payment_terms_days": payment_terms_days,
                "previous_late_payments": previous_late_payments,
                "total_previous_invoices": total_previous_invoices,
                "paid_late": paid_late,
            }
        )

    return pd.DataFrame(rows)


def validate_and_clean_data(
    uploaded_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate required columns and clean numeric values.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in uploaded_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    cleaned_data = uploaded_data.copy()

    numeric_columns = [
        "invoice_amount",
        "payment_terms_days",
        "previous_late_payments",
        "total_previous_invoices",
        "paid_late",
    ]

    for column in numeric_columns:
        cleaned_data[column] = pd.to_numeric(
            cleaned_data[column],
            errors="coerce",
        )

    cleaned_data = cleaned_data.dropna(
        subset=numeric_columns
    )

    cleaned_data = cleaned_data[
        cleaned_data["invoice_amount"] > 0
    ]

    cleaned_data = cleaned_data[
        cleaned_data["payment_terms_days"] > 0
    ]

    cleaned_data = cleaned_data[
        cleaned_data["total_previous_invoices"] > 0
    ]

    cleaned_data = cleaned_data[
        cleaned_data["previous_late_payments"] >= 0
    ]

    cleaned_data = cleaned_data[
        cleaned_data["previous_late_payments"]
        <= cleaned_data["total_previous_invoices"]
    ]

    cleaned_data["paid_late"] = (
        cleaned_data["paid_late"]
        .round()
        .astype(int)
    )

    cleaned_data = cleaned_data[
        cleaned_data["paid_late"].isin([0, 1])
    ]

    if len(cleaned_data) < 20:
        raise ValueError(
            "The dataset needs at least 20 valid invoices."
        )

    if cleaned_data["paid_late"].nunique() < 2:
        raise ValueError(
            "The paid_late column must contain both 0 and 1."
        )

    return cleaned_data


def train_payment_model(
    invoice_data: pd.DataFrame,
):
    """
    Train a Random Forest model and calculate test accuracy.
    """

    X = invoice_data[FEATURE_COLUMNS]
    y = invoice_data["paid_late"]

    stratify_value = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_value,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    return model, accuracy


def assign_risk_level(
    probability: float,
) -> str:
    """
    Convert a probability into a business risk level.
    """

    if probability >= 0.70:
        return "HIGH"

    if probability >= 0.40:
        return "MEDIUM"

    return "LOW"


def collection_recommendation(
    probability: float,
) -> str:
    """
    Return a recommended collection action.
    """

    if probability >= 0.70:
        return (
            "Contact the customer today. Confirm that the invoice "
            "was received, ask whether payment has been scheduled, "
            "and check for disputes or missing documentation."
        )

    if probability >= 0.40:
        return (
            "Send a professional reminder and monitor the invoice "
            "for the next 48 hours. Escalate if no response is received."
        )

    return (
        "Continue normal monitoring. No immediate collection action "
        "is required."
    )


def generate_local_reminder(
    customer_name: str,
    invoice_number: str,
    invoice_amount: float,
    due_date: date,
    days_overdue: int,
    risk_level: str,
) -> str:
    """
    Generate a professional payment reminder through local Ollama.
    """

    prompt = f"""
Write a concise and professional business payment reminder.

Customer name: {customer_name}
Invoice number: {invoice_number}
Invoice amount: ${invoice_amount:,.2f}
Due date: {due_date.isoformat()}
Days overdue: {days_overdue}
Risk level: {risk_level}

Requirements:
- Address the customer by name.
- Mention the exact invoice number.
- Mention the exact invoice amount.
- Mention the exact due date.
- Ask whether payment has already been processed.
- Ask the customer to report any dispute or missing documentation.
- Be polite and professional.
- Do not threaten the customer.
- Do not invent information.
- Keep the message under 120 words.
- Return only the final payment reminder.
"""

    response = ollama.chat(
        model="gemma3:1b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"].strip()


# =========================================================
# SESSION STATE
# =========================================================

if "invoice_data" not in st.session_state:
    st.session_state["invoice_data"] = None

if "model" not in st.session_state:
    st.session_state["model"] = None

if "model_accuracy" not in st.session_state:
    st.session_state["model_accuracy"] = None

if "ranked_invoices" not in st.session_state:
    st.session_state["ranked_invoices"] = None

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

if "generated_reminder" not in st.session_state:
    st.session_state["generated_reminder"] = ""


# =========================================================
# HEADER
# =========================================================

st.title("Invoice Risk AI")

st.write(
    "Identify invoices at risk of late payment, prioritize collection "
    "work, and generate professional payment reminders."
)

st.warning(
    "This prototype is a decision-support tool. Predictions are not "
    "guarantees and must be reviewed by a human."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("Prototype controls")

    st.write(
        "Use demo data for testing or upload a business CSV."
    )

    if st.button(
        "Load demo data",
        use_container_width=True,
    ):
        st.session_state["invoice_data"] = create_demo_data()
        st.session_state["model"] = None
        st.session_state["ranked_invoices"] = None
        st.session_state["last_prediction"] = None

        st.success("Demo data loaded.")

    if st.button(
        "Reset application",
        use_container_width=True,
    ):
        st.session_state["invoice_data"] = None
        st.session_state["model"] = None
        st.session_state["model_accuracy"] = None
        st.session_state["ranked_invoices"] = None
        st.session_state["last_prediction"] = None
        st.session_state["generated_reminder"] = ""

        st.success("Application reset.")


# =========================================================
# DATA UPLOAD
# =========================================================

st.divider()

st.header("1. Load historical invoice data")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
)

with st.expander("Required CSV columns"):
    st.code(
        "\n".join(REQUIRED_COLUMNS),
        language="text",
    )

    st.write(
        "`paid_late` must be 1 when the invoice was paid late "
        "and 0 when it was paid on time."
    )


if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)

        st.session_state["invoice_data"] = (
            validate_and_clean_data(uploaded_data)
        )

        st.session_state["model"] = None
        st.session_state["ranked_invoices"] = None
        st.session_state["last_prediction"] = None

        st.success(
            "CSV loaded and validated successfully."
        )

    except Exception as error:
        st.error(str(error))


# =========================================================
# DASHBOARD AND MODEL TRAINING
# =========================================================

invoice_data = st.session_state["invoice_data"]

if invoice_data is not None:
    st.divider()

    st.header("2. Invoice portfolio dashboard")

    total_invoice_value = invoice_data[
        "invoice_amount"
    ].sum()

    historically_late_value = invoice_data.loc[
        invoice_data["paid_late"] == 1,
        "invoice_amount",
    ].sum()

    historically_late_count = int(
        invoice_data["paid_late"].sum()
    )

    historical_late_rate = (
        invoice_data["paid_late"].mean()
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Historical invoices",
        f"{len(invoice_data):,}",
    )

    metric2.metric(
        "Total invoice value",
        f"${total_invoice_value:,.2f}",
    )

    metric3.metric(
        "Historically late value",
        f"${historically_late_value:,.2f}",
    )

    metric4.metric(
        "Historical late rate",
        f"{historical_late_rate:.1%}",
    )

    with st.expander("View historical invoice data"):
        st.dataframe(
            invoice_data,
            use_container_width=True,
        )

    if st.button(
        "Train payment-risk model",
        type="primary",
    ):
        try:
            with st.spinner(
                "Training payment-risk model..."
            ):
                model, accuracy = train_payment_model(
                    invoice_data
                )

                ranked_invoices = invoice_data.copy()

                probabilities = model.predict_proba(
                    ranked_invoices[FEATURE_COLUMNS]
                )[:, 1]

                ranked_invoices[
                    "late_payment_probability"
                ] = probabilities

                ranked_invoices[
                    "risk_level"
                ] = ranked_invoices[
                    "late_payment_probability"
                ].apply(assign_risk_level)

                ranked_invoices[
                    "recommended_action"
                ] = ranked_invoices[
                    "late_payment_probability"
                ].apply(collection_recommendation)

                ranked_invoices = ranked_invoices.sort_values(
                    by="late_payment_probability",
                    ascending=False,
                )

                st.session_state["model"] = model
                st.session_state["model_accuracy"] = accuracy
                st.session_state[
                    "ranked_invoices"
                ] = ranked_invoices

            st.success(
                "Payment-risk model trained successfully."
            )

        except Exception as error:
            st.error(
                f"Model training failed: {error}"
            )


# =========================================================
# PRIORITIZED RISK LIST
# =========================================================

ranked_invoices = st.session_state["ranked_invoices"]

if ranked_invoices is not None:
    st.divider()

    st.header("3. Prioritized invoice-risk list")

    model_accuracy = st.session_state[
        "model_accuracy"
    ]

    high_risk_count = int(
        (
            ranked_invoices["risk_level"]
            == "HIGH"
        ).sum()
    )

    estimated_high_risk_value = ranked_invoices.loc[
        ranked_invoices["risk_level"] == "HIGH",
        "invoice_amount",
    ].sum()

    result1, result2, result3 = st.columns(3)

    result1.metric(
        "Test accuracy",
        f"{model_accuracy:.1%}",
    )

    result2.metric(
        "High-risk invoices",
        f"{high_risk_count:,}",
    )

    result3.metric(
        "High-risk invoice value",
        f"${estimated_high_risk_value:,.2f}",
    )

    display_columns = [
        column
        for column in [
            "invoice_id",
            "customer_name",
            "invoice_amount",
            "payment_terms_days",
            "previous_late_payments",
            "total_previous_invoices",
            "late_payment_probability",
            "risk_level",
            "recommended_action",
        ]
        if column in ranked_invoices.columns
    ]

    st.dataframe(
        ranked_invoices[display_columns],
        use_container_width=True,
        column_config={
            "invoice_amount": st.column_config.NumberColumn(
                "Invoice amount",
                format="$%.2f",
            ),
            "late_payment_probability": (
                st.column_config.ProgressColumn(
                    "Late-payment probability",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.1%%",
                )
            ),
        },
    )

    ranked_csv = ranked_invoices.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download prioritized invoice list",
        data=ranked_csv,
        file_name="prioritized_invoice_risk.csv",
        mime="text/csv",
    )


# =========================================================
# SINGLE-INVOICE PREDICTION
# =========================================================

model = st.session_state["model"]

if model is not None:
    st.divider()

    st.header("4. Analyze a new invoice")

    with st.form("new_invoice_form"):
        invoice_amount = st.number_input(
            "Invoice amount",
            min_value=1.0,
            value=10000.0,
            step=100.0,
        )

        payment_terms_days = st.number_input(
            "Payment terms in days",
            min_value=1,
            value=30,
            step=1,
        )

        previous_late_payments = st.number_input(
            "Previous late payments",
            min_value=0,
            value=2,
            step=1,
        )

        total_previous_invoices = st.number_input(
            "Total previous invoices",
            min_value=1,
            value=10,
            step=1,
        )

        analyze_invoice = st.form_submit_button(
            "Calculate payment risk",
            type="primary",
        )

    if analyze_invoice:
        if (
            previous_late_payments
            > total_previous_invoices
        ):
            st.error(
                "Previous late payments cannot be greater "
                "than total previous invoices."
            )

        else:
            new_invoice = pd.DataFrame(
                [
                    {
                        "invoice_amount": invoice_amount,
                        "payment_terms_days": payment_terms_days,
                        "previous_late_payments": previous_late_payments,
                        "total_previous_invoices": total_previous_invoices,
                    }
                ]
            )

            probability = model.predict_proba(
                new_invoice
            )[0][1]

            risk_level = assign_risk_level(
                probability
            )

            recommended_action = (
                collection_recommendation(
                    probability
                )
            )

            st.session_state[
                "last_prediction"
            ] = {
                "probability": probability,
                "risk_level": risk_level,
                "recommended_action": (
                    recommended_action
                ),
            }

    prediction = st.session_state[
        "last_prediction"
    ]

    if prediction is not None:
        prediction1, prediction2 = st.columns(2)

        prediction1.metric(
            "Late-payment probability",
            f"{prediction['probability']:.1%}",
        )

        prediction2.metric(
            "Risk level",
            prediction["risk_level"],
        )

        if prediction["risk_level"] == "HIGH":
            st.error(
                prediction["recommended_action"]
            )

        elif prediction["risk_level"] == "MEDIUM":
            st.warning(
                prediction["recommended_action"]
            )

        else:
            st.success(
                prediction["recommended_action"]
            )


# =========================================================
# LOCAL AI PAYMENT REMINDER
# =========================================================

st.divider()

st.header("5. Generate a payment reminder")

st.caption(
    "This uses Gemma 3 locally through Ollama. "
    "No paid OpenAI API request is required."
)

with st.form("reminder_form"):
    customer_name = st.text_input(
        "Customer name",
        value="Example Customer",
    )

    invoice_number = st.text_input(
        "Invoice number",
        value="INV-1001",
    )

    reminder_invoice_amount = st.number_input(
        "Invoice amount for reminder",
        min_value=0.0,
        value=10000.0,
        step=100.0,
    )

    invoice_due_date = st.date_input(
        "Invoice due date",
    )

    days_overdue = max(
        (date.today() - invoice_due_date).days,
        0,
    )

    reminder_risk_level = st.selectbox(
        "Risk level",
        options=["LOW", "MEDIUM", "HIGH"],
        index=1,
    )

    generate_reminder = st.form_submit_button(
        "Generate AI reminder",
        type="primary",
    )


if generate_reminder:
    try:
        with st.spinner(
            "Generating reminder locally..."
        ):
            reminder = generate_local_reminder(
                customer_name=customer_name,
                invoice_number=invoice_number,
                invoice_amount=(
                    reminder_invoice_amount
                ),
                due_date=invoice_due_date,
                days_overdue=days_overdue,
                risk_level=reminder_risk_level,
            )

            st.session_state[
                "generated_reminder"
            ] = reminder

    except Exception as error:
        st.error(
            f"Could not connect to Ollama: {error}"
        )

        st.info(
            "Make sure the Ollama application is open "
            "and gemma3:1b is installed."
        )


if st.session_state["generated_reminder"]:
    st.text_area(
        "Review and edit before sending",
        value=st.session_state[
            "generated_reminder"
        ],
        height=240,
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Prototype only. Use anonymized information during testing. "
    "Do not upload sensitive customer data until authentication, "
    "encryption, access controls, backups, and privacy policies "
    "have been implemented."
)
