import streamlit as st
import requests


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)


st.title("📊 Customer Churn Prediction")

st.write(
    "Deep Learning ANN Customer Churn Prediction"
)


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)


partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)


dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)


tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=100,
    value=12
)


phone = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)


multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)


internet = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


security = st.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


backup = st.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


device = st.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


support = st.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


stream_tv = st.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


stream_movies = st.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)


payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)


total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)


if st.button("Predict Churn"):

    payload = {

        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    if response.status_code == 200:

        result = response.json()

        probability = result[
            "churn_probability"
        ]

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

        if result["prediction"] == 1:

            st.error(
                "⚠️ Customer is likely to churn"
            )

        else:

            st.success(
                "✅ Customer is unlikely to churn"
            )

    else:

        st.error(
            "Prediction API error"
        )