import random
import pandas as pd

random.seed(42)

rows = []

for i in range(500):

    invoice_amount = random.randint(500, 100000)

    payment_terms_days = random.choice(
        [15, 30, 45, 60]
    )

    total_previous_invoices = random.randint(1, 50)

    previous_late_payments = random.randint(
        0,
        total_previous_invoices
    )

    late_rate = (
        previous_late_payments
        / total_previous_invoices
    )

    risk = late_rate

    if invoice_amount > 50000:
        risk += 0.15

    if payment_terms_days >= 45:
        risk += 0.10

    risk += random.uniform(-0.20, 0.20)

    if risk >= 0.50:
        paid_late = 1
    else:
        paid_late = 0

    rows.append(
        {
            "invoice_amount": invoice_amount,
            "payment_terms_days": payment_terms_days,
            "previous_late_payments": previous_late_payments,
            "total_previous_invoices": total_previous_invoices,
            "paid_late": paid_late
        }
    )

data = pd.DataFrame(rows)

data.to_csv(
    "demo_invoices.csv",
    index=False
)

print("Created 500 training invoices.")
