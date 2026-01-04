import streamlit as st
import pandas as pd
import joblib
import time

# ================= LOAD SAVED OBJECTS =================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le_education = joblib.load("education_encoder.pkl")
le_self_employed = joblib.load("self_employed_encoder.pkl")

# ================= FEATURE ORDER (MATCH TRAINING – NO LOAN ID) =================
FEATURE_COLUMNS = [
    " no_of_dependents",
    " education",
    " self_employed",
    " income_annum",
    " loan_amount",
    " loan_term",
    " cibil_score",
    " residential_assets_value",
    " commercial_assets_value",
    " luxury_assets_value",
    " bank_asset_value"
]

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter all details below and click **Predict Loan Status**.")

# ================= INPUTS =================

# ---- Row 1 ----
col1, col2 = st.columns(2)
with col1:
    loan_id = st.number_input("Loan ID", step=1)
with col2:
    no_of_dependents = st.number_input("Number of Dependents", step=1)

# ---- Row 2 (STRING VALUES TOGETHER) ----
col3, col4 = st.columns(2)
with col3:
    education = st.radio(
        "Education",
        le_education.classes_.tolist()
    )
with col4:
    self_employed = st.radio(
        "Self Employed",
        le_self_employed.classes_.tolist()
    )

# ---- Row 3 ----
col5, col6 = st.columns(2)
with col5:
    income_annum = st.number_input("Annual Income")
with col6:
    loan_amount = st.number_input("Loan Amount")

# ---- Row 4 ----
col7, col8 = st.columns(2)
with col7:
    loan_term = st.number_input("Loan Term (months)")
with col8:
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)

# ---- Row 5 ----
col9, col10 = st.columns(2)
with col9:
    residential_assets_value = st.number_input("Residential Assets Value")
with col10:
    commercial_assets_value = st.number_input("Commercial Assets Value")

# ---- Row 6 ----
col11, col12 = st.columns(2)
with col11:
    luxury_assets_value = st.number_input("Luxury Assets Value")
with col12:
    bank_asset_value = st.number_input("Bank Asset Value")

# ================= PREDICTION =================
if st.button("Predict Loan Status"):

    with st.spinner("🔍 Evaluating loan application..."):
        time.sleep(1)

        # ---------- DISPLAY DATA (INCLUDES LOAN ID) ----------
        display_df = pd.DataFrame({
            "Loan ID": [loan_id],
            "Dependents": [no_of_dependents],
            "Education": [education],
            "Self Employed": [self_employed],
            "Income": [income_annum],
            "Loan Amount": [loan_amount],
            "Loan Term": [loan_term],
            "CIBIL Score": [cibil_score],
            "Residential Assets": [residential_assets_value],
            "Commercial Assets": [commercial_assets_value],
            "Luxury Assets": [luxury_assets_value],
            "Bank Assets": [bank_asset_value]
        })

        # ---------- MODEL INPUT (NO LOAN ID) ----------
        input_df = pd.DataFrame({
            " no_of_dependents": [no_of_dependents],
            " education": [education],
            " self_employed": [self_employed],
            " income_annum": [income_annum],
            " loan_amount": [loan_amount],
            " loan_term": [loan_term],
            " cibil_score": [cibil_score],
            " residential_assets_value": [residential_assets_value],
            " commercial_assets_value": [commercial_assets_value],
            " luxury_assets_value": [luxury_assets_value],
            " bank_asset_value": [bank_asset_value]
        })

        # ---------- ENCODING ----------
        input_df[" education"] = le_education.transform(input_df[" education"])
        input_df[" self_employed"] = le_self_employed.transform(input_df[" self_employed"])

        # ---------- ORDER ----------
        input_df = input_df[FEATURE_COLUMNS]

        # ---------- SCALING ----------
        input_scaled = scaler.transform(input_df.values)

        # ---------- PREDICTION ----------
        prediction = model.predict(input_scaled)

    # ================= HIGHLIGHTED OUTPUT =================
    if prediction[0] == 1:
        st.markdown(
            """
            <div style="padding:16px; border-radius:10px;
                        background-color:#e6ffe6;
                        border:2px solid #2ecc71;
                        text-align:center;">
                <h2 style="color:#2ecc71;">✅ Loan Approved</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="padding:16px; border-radius:10px;
                        background-color:#ffe6e6;
                        border:2px solid #e74c3c;
                        text-align:center;">
                <h2 style="color:#e74c3c;">❌ Loan Rejected</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ================= DETAILS TABLE =================
    st.subheader("📋 Entered Details")
    st.dataframe(display_df, use_container_width=True)

    # ================= FEEDBACK (AFTER RESULT) =================
    st.subheader("⭐ Give Your Feedback")
    st.feedback("stars")
