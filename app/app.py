import streamlit as st
import pandas as pd
import joblib

# Set page title and layout
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

# Load the three saved models
@st.cache_resource
def load_models():
    nb = joblib.load('models/loan_model_nb.pkl')
    svm = joblib.load('models/loan_model_svm.pkl')
    dt = joblib.load('models/loan_model_dt.pkl')
    return nb, svm, dt

nb_model, svm_model, dt_model = load_models()

st.title("🏦 Loan Approval Predictor")
st.write("Enter the applicant details below to predict loan approval status using **Naive Bayes**, **Support Vector Machine (SVM)**, and **Decision Tree**.")

# Create input form
with st.form("loan_form"):
    st.subheader("Applicant Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
    with col2:
        applicant_income = st.number_input("Applicant Income (₹ Monthly)", min_value=0, value=50000, step=1000)
        coapplicant_income = st.number_input("Coapplicant Income (₹ Monthly)", min_value=0, value=15000, step=1000)
        loan_amount = st.number_input("Loan Amount (in ₹ Thousands, e.g. 1400 = 14 Lakhs)", min_value=1, value=1400, step=50)
        loan_term = st.number_input("Loan Amount Term (in months)", min_value=12, value=360, step=12)
        credit_history = st.selectbox("Credit History", [("Good (1.0)", 1.0), ("Bad (0.0)", 0.0)], format_func=lambda x: x[0])

    # Submit button
    submit_button = st.form_submit_button(label="Predict Loan Approval", use_container_width=True)

# If the user clicks the button, process the input
if submit_button:
    # 1. Format the input exactly like our training data
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history[1]], # Extract the number (1.0 or 0.0)
        'Property_Area': [property_area]
    })
    
    # 2. Make predictions with all three models
    nb_pred = nb_model.predict(input_data)[0]
    nb_prob = nb_model.predict_proba(input_data)[0][1]
    
    svm_pred = svm_model.predict(input_data)[0]
    svm_prob = svm_model.predict_proba(input_data)[0][1]

    dt_pred = dt_model.predict(input_data)[0]
    dt_prob = dt_model.predict_proba(input_data)[0][1]
    
    # 3. Display the side-by-side results
    st.divider()
    st.subheader("Prediction Results")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.markdown("### 1. Naive Bayes")
        if nb_pred == 1:
            st.success(f"✅ **APPROVED**\n\n**Confidence:** {nb_prob*100:.1f}%\n\n**Risk:** {(1-nb_prob)*100:.1f}%")
        else:
            st.error(f"❌ **REJECTED**\n\n**Confidence:** {(1-nb_prob)*100:.1f}%\n\n**Risk:** {(1-nb_prob)*100:.1f}%")
            
    with col_res2:
        st.markdown("### 2. SVM (RBF)")
        if svm_pred == 1:
            st.success(f"✅ **APPROVED**\n\n**Confidence:** {svm_prob*100:.1f}%\n\n**Risk:** {(1-svm_prob)*100:.1f}%")
        else:
            st.error(f"❌ **REJECTED**\n\n**Confidence:** {(1-svm_prob)*100:.1f}%\n\n**Risk:** {(1-svm_prob)*100:.1f}%")

    with col_res3:
        st.markdown("### 3. Decision Tree")
        if dt_pred == 1:
            st.success(f"✅ **APPROVED**\n\n**Confidence:** {dt_prob*100:.1f}%\n\n**Risk:** {(1-dt_prob)*100:.1f}%")
        else:
            st.error(f"❌ **REJECTED**\n\n**Confidence:** {(1-dt_prob)*100:.1f}%\n\n**Risk:** {(1-dt_prob)*100:.1f}%")

    # 4. Output Comparison Table
    st.divider()
    st.subheader("📊 Detailed Output Comparison")
    
    nb_result = "Approved" if nb_pred == 1 else "Rejected"
    svm_result = "Approved" if svm_pred == 1 else "Rejected"
    dt_result = "Approved" if dt_pred == 1 else "Rejected"
    
    comparison_data = {
        "Model": ["Naive Bayes", "SVM (RBF)", "Decision Tree"],
        "Prediction": [nb_result, svm_result, dt_result],
        "Confidence Level": [f"{max(nb_prob, 1-nb_prob)*100:.1f}%", f"{max(svm_prob, 1-svm_prob)*100:.1f}%", f"{max(dt_prob, 1-dt_prob)*100:.1f}%"],
        "Probability of Approval": [f"{nb_prob*100:.1f}%", f"{svm_prob*100:.1f}%", f"{dt_prob*100:.1f}%"],
        "Probability of Default": [f"{(1-nb_prob)*100:.1f}%", f"{(1-svm_prob)*100:.1f}%", f"{(1-dt_prob)*100:.1f}%"]
    }
    
    df_results = pd.DataFrame(comparison_data)
    st.table(df_results.set_index("Model"))
    
    total_approved = nb_pred + svm_pred + dt_pred
    if total_approved == 3:
        st.info("🤝 **All 3 models agree** -> Approve Loan.")
    elif total_approved == 0:
        st.error("❌ **All 3 models agree** -> Reject Loan (High Default Risk).")
    elif total_approved == 2:
        st.warning("⚠️ **Majority Consensus (2/3):** Approve Loan (Review marginal risk factors).")
    else:
        st.warning("⚠️ **Majority Consensus (2/3):** Reject Loan (High Default Risk).")

# --- Model Training Metrics Section ---
st.divider()
with st.expander("ℹ️ View Model Training Performance (from Notebook)"):
    st.write("Performance achieved on the 20% test data:")
    
    metrics_data = {
        "Model": ["Naive Bayes", "SVM (RBF)", "Decision Tree (max_depth=4)"],
        "Accuracy": ["84.55%", "85.37%", "84.55%"],
        "ROC-AUC": ["80.77%", "83.03%", "70.60%"],
        "Precision": ["83.67%", "83.17%", "82.35%"],
        "Recall": ["96.47%", "98.82%", "98.82%"],
        "F1-Score": ["89.62%", "90.32%", "89.84%"],
        "Interpretability": ["Probabilistic", "Black Box / Kernel Hyperplane", "Direct Visual Logic (Tree)"]
    }
    
    st.table(pd.DataFrame(metrics_data).set_index("Model"))
    st.caption("SVM achieves highest Accuracy (85.37%) and ROC-AUC (83.03%), while Decision Tree provides full visual interpretability.")
