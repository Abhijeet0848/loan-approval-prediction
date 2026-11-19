"""
Loan Approval Predictor - Terminal Edition
Runs directly in the command prompt / terminal (No browser required).
Supports fast 1-letter shortcuts:
  • Gender:        m = Male, f = Female
  • Married:       y = Yes, n = No
  • Dependents:    0, 1, 2, 3 (or 3+)
  • Education:     g = Graduate, ng = Not Graduate
  • Self Employed: y = Yes, n = No
  • Credit:        1 = Good, 0 = Bad
  • Property:      u = Urban, s = Semiurban, r = Rural

Models Evaluated:
  1. Naive Bayes (GaussianNB)
  2. Support Vector Machine (SVC RBF Kernel)
  3. Decision Tree (max_depth=4)
"""

import os
import sys
import warnings
import joblib
import pandas as pd

warnings.filterwarnings('ignore')

# Path to models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(BASE_DIR, 'models', 'loan_model_nb.pkl')
SVM_PATH = os.path.join(BASE_DIR, 'models', 'loan_model_svm.pkl')
DT_PATH = os.path.join(BASE_DIR, 'models', 'loan_model_dt.pkl')


def load_models():
    missing = [p for p in [NB_PATH, SVM_PATH, DT_PATH] if not os.path.exists(p)]
    if missing:
        print("❌ Error: Missing model files:")
        for m in missing:
            print(f"   • {os.path.basename(m)}")
        print("Please run 'python train.py' first to train and save all 3 models.")
        sys.exit(1)
    
    nb = joblib.load(NB_PATH)
    svm = joblib.load(SVM_PATH)
    dt = joblib.load(DT_PATH)
    return nb, svm, dt


def get_mapped_choice(prompt, default_key, alias_map):
    """
    Accepts 1-letter shortcuts or full names, and maps to the model's required format.
    Example: 'm' -> 'Male', 'g' -> 'Graduate', 'u' -> 'Urban'
    """
    default_canonical = alias_map[default_key.lower()]
    while True:
        raw = input(f"{prompt} [{default_key}]: ").strip().lower()
        if not raw:
            return default_canonical
        
        if raw in alias_map:
            return alias_map[raw]
        
        valid_keys = list(set(alias_map.keys()))
        print(f"  ⚠️ Invalid choice '{raw}'. Please use: {', '.join(sorted(valid_keys))}")


def get_number(prompt, default_val, cast_type=float):
    """Prompts for number with default value."""
    while True:
        raw = input(f"{prompt} [{default_val}]: ").strip()
        if not raw:
            return cast_type(default_val)
        try:
            val = cast_type(raw)
            if val < 0:
                print("  ⚠️ Value cannot be negative.")
                continue
            return val
        except ValueError:
            print("  ⚠️ Please enter a valid number.")


def run_prediction_cli(nb_model, svm_model, dt_model):
    print("\n" + "=" * 65)
    print("        🏦 LOAN APPROVAL PREDICTOR (TERMINAL EDITION)       ")
    print("=" * 65)
    print("Type 1-letter shortcuts or press Enter to accept [default]:\n")

    # 1. Gender (m / f)
    gender = get_mapped_choice(
        "1. Gender (m = Male, f = Female)",
        default_key="m",
        alias_map={"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
    )

    # 2. Married (y / n)
    married = get_mapped_choice(
        "2. Married (y = Yes, n = No)",
        default_key="y",
        alias_map={"y": "Yes", "yes": "Yes", "n": "No", "no": "No"}
    )

    # 3. Dependents (0, 1, 2, 3+)
    dependents = get_mapped_choice(
        "3. Dependents (0, 1, 2, 3+)",
        default_key="0",
        alias_map={"0": "0", "1": "1", "2": "2", "3": "3+", "3+": "3+", "3plus": "3+"}
    )

    # 4. Education (g / ng)
    education = get_mapped_choice(
        "4. Education (g = Graduate, ng = Not Graduate)",
        default_key="g",
        alias_map={"g": "Graduate", "graduate": "Graduate", "ng": "Not Graduate", "n": "Not Graduate", "not graduate": "Not Graduate"}
    )

    # 5. Self Employed (y / n)
    self_employed = get_mapped_choice(
        "5. Self Employed (y = Yes, n = No)",
        default_key="n",
        alias_map={"n": "No", "no": "No", "y": "Yes", "yes": "Yes"}
    )

    # 6. Applicant Income
    applicant_income = get_number("6. Applicant Income (₹/month)", default_val=50000, cast_type=float)

    # 7. Coapplicant Income
    coapplicant_income = get_number("7. Coapplicant Income (₹/month)", default_val=140000, cast_type=float)

    # 8. Loan Amount (in thousands)
    loan_amount = get_number("8. Loan Amount (in ₹ Thousands, e.g. 1400 for ₹14 Lakhs)", default_val=1400, cast_type=float)
    # Smart auto-scaling: if user entered full dollars (>= 1000), convert to thousands
    if loan_amount >= 1000:
        loan_amount = loan_amount / 1000.0

    # 9. Loan Term
    loan_term = get_number("9. Loan Term (e.g. 360 for months or 30 for years)", default_val=360, cast_type=float)
    # Smart auto-scaling: if user entered years (<= 30), convert to months
    if loan_term <= 30 and loan_term > 0:
        loan_term = loan_term * 12.0

    # 10. Credit History (1 / 0)
    credit_choice = get_mapped_choice(
        "10. Credit History (1 = Good, 0 = Bad)",
        default_key="1",
        alias_map={"1": 1.0, "good": 1.0, "y": 1.0, "yes": 1.0, "0": 0.0, "bad": 0.0, "n": 0.0, "no": 0.0}
    )

    # 11. Property Area (u / s / r)
    property_area = get_mapped_choice(
        "11. Property Area (u = Urban, s = Semiurban, r = Rural)",
        default_key="u",
        alias_map={
            "u": "Urban", "urban": "Urban",
            "s": "Semiurban", "semi": "Semiurban", "semiurban": "Semiurban",
            "r": "Rural", "rural": "Rural"
        }
    )

    # Build input DataFrame
    applicant_df = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_choice,
        'Property_Area': property_area
    }])

    # Print summary of input
    print("\n" + "-" * 65)
    print(f" Profile: {gender}, Married: {married}, Dependents: {dependents}, {education}")
    print(f" Income : ₹{applicant_income:,.0f} + ₹{coapplicant_income:,.0f} / mo")
    print(f" Loan   : ₹{loan_amount*1000:,.0f} for {int(loan_term)} months ({int(loan_term//12)} yrs)")
    print(f" Credit : {'Good (1.0)' if credit_choice == 1.0 else 'Bad / Default (0.0)'}, Area: {property_area}")
    print("-" * 65)

    # Predict Probabilities
    nb_prob = nb_model.predict_proba(applicant_df)[0][1]
    svm_prob = svm_model.predict_proba(applicant_df)[0][1]
    dt_prob = dt_model.predict_proba(applicant_df)[0][1]

    nb_pred = 1 if nb_prob >= 0.5 else 0
    svm_pred = 1 if svm_prob >= 0.5 else 0
    dt_pred = 1 if dt_prob >= 0.5 else 0

    nb_verdict = "APPROVED ✅" if nb_pred == 1 else "REJECTED ❌"
    svm_verdict = "APPROVED ✅" if svm_pred == 1 else "REJECTED ❌"
    dt_verdict = "APPROVED ✅" if dt_pred == 1 else "REJECTED ❌"

    nb_conf = max(nb_prob, 1.0 - nb_prob)
    svm_conf = max(svm_prob, 1.0 - svm_prob)
    dt_conf = max(dt_prob, 1.0 - dt_prob)

    # Print Results Box
    print("\n" + "=" * 65)
    print("                      PREDICTION RESULTS                     ")
    print("=" * 65)
    print(f" 1. NAIVE BAYES MODEL:")
    print(f"    • Verdict              : {nb_verdict}")
    print(f"    • Confidence Level     : {nb_conf * 100:.2f}%")
    print(f"    • Approval Probability : {nb_prob * 100:.2f}%")
    print(f"    • Risk of Default      : {(1.0 - nb_prob) * 100:.2f}%")
    print("-" * 65)
    print(f" 2. SVM (SUPPORT VECTOR MACHINE) MODEL:")
    print(f"    • Verdict              : {svm_verdict}")
    print(f"    • Confidence Level     : {svm_conf * 100:.2f}%")
    print(f"    • Approval Probability : {svm_prob * 100:.2f}%")
    print(f"    • Risk of Default      : {(1.0 - svm_prob) * 100:.2f}%")
    print("-" * 65)
    print(f" 3. DECISION TREE MODEL (max_depth=4):")
    print(f"    • Verdict              : {dt_verdict}")
    print(f"    • Confidence Level     : {dt_conf * 100:.2f}%")
    print(f"    • Approval Probability : {dt_prob * 100:.2f}%")
    print(f"    • Risk of Default      : {(1.0 - dt_prob) * 100:.2f}%")
    print("=" * 65)

    # Consensus across 3 models
    approvals = nb_pred + svm_pred + dt_pred
    if approvals == 3:
        print(" 🤝 UNANIMOUS CONSENSUS: ALL 3 MODELS AGREE -> APPROVE LOAN ✅")
    elif approvals == 2:
        print(" 📊 MAJORITY CONSENSUS (2/3): APPROVE LOAN (Review Marginal Risks) ⚠️")
    elif approvals == 1:
        print(" 📊 MAJORITY CONSENSUS (2/3): REJECT LOAN (High Default Risk) ⚠️")
    else:
        print(" ❌ UNANIMOUS CONSENSUS: ALL 3 MODELS AGREE -> REJECT LOAN (HIGH RISK)")
    print("=" * 65 + "\n")


def main():
    nb_model, svm_model, dt_model = load_models()

    if len(sys.argv) > 1 and sys.argv[1].lower() in ['--demo', '-d', 'demo']:
        print("\n=== Running Quick Demo with Sample Applicant ===")
        sample = pd.DataFrame([{
            'Gender': 'Male', 'Married': 'Yes', 'Dependents': '1', 'Education': 'Graduate',
            'Self_Employed': 'No', 'ApplicantIncome': 5849, 'CoapplicantIncome': 0,
            'LoanAmount': 128, 'Loan_Amount_Term': 360, 'Credit_History': 1.0, 'Property_Area': 'Urban'
        }])
        nb_prob = nb_model.predict_proba(sample)[0][1]
        svm_prob = svm_model.predict_proba(sample)[0][1]
        dt_prob = dt_model.predict_proba(sample)[0][1]
        print(f"Naive Bayes   : {'APPROVED ✅' if nb_prob>=0.5 else 'REJECTED ❌'} (Confidence: {max(nb_prob, 1-nb_prob)*100:.1f}%)")
        print(f"SVM (RBF)     : {'APPROVED ✅' if svm_prob>=0.5 else 'REJECTED ❌'} (Confidence: {max(svm_prob, 1-svm_prob)*100:.1f}%)")
        print(f"Decision Tree : {'APPROVED ✅' if dt_prob>=0.5 else 'REJECTED ❌'} (Confidence: {max(dt_prob, 1-dt_prob)*100:.1f}%)")
        return

    while True:
        run_prediction_cli(nb_model, svm_model, dt_model)
        again = input("Evaluate another applicant? (y/n) [n]: ").strip().lower()
        if again not in ['y', 'yes']:
            print("\nExiting. Thank you!")
            break


if __name__ == '__main__':
    main()
