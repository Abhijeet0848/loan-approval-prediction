# 🏦 Loan Approval Prediction System

An end-to-end Machine Learning system for predicting loan approval decisions using **Naive Bayes**, **Support Vector Machine (SVM)**, and **Decision Tree**, ready to run in **Google Colab**, **Local Jupyter**, or the **Command Terminal**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

---

## 📊 Dataset Provenance & Original Source
- **Original Source**: **Kaggle** & **Analytics Vidhya** (*[Loan Prediction Practice Problem](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/)* / *[Kaggle Dataset](https://www.kaggle.com/datasets/altruistdelphi01/loan-prediction-problem-dataset)*).
- **Domain**: Retail Banking Credit Risk Underwriting (Indian Banking Context).
- **Scale**: 10,000 applicant applications across 11 predictive financial and demographic attributes (salaries, loan amounts, credit histories, property locations).

## 📁 Project Structure

```
Loan/
├── Loan_Classifier_Comparison_Report.pdf # 📄 10-Page Colab-style PDF Report
├── Loan_Classifier_Comparison_Report.html# 🌐 HTML version of the report
├── Loan_Approval_Google_Colab.ipynb      # 🌟 1-Click Google Colab Notebook (Complete Pipeline)
├── Data/
│   ├── loan.csv                          # Dataset (10,000 rows × 13 columns)
│   └── loan_original_614.csv             # Original baseline dataset (614 rows)
├── models/
│   ├── loan_model_nb.pkl                 # Model 1: Naive Bayes Pipeline (GaussianNB)
│   ├── loan_model_svm.pkl                # Model 2: Support Vector Machine (RBF) Pipeline
│   └── loan_model_dt.pkl                 # Model 3: Decision Tree (max_depth=4) Pipeline
├── run.py                                # Fast Terminal CLI predictor (with 1-letter shortcuts & consensus)
├── run.bat                               # 1-click terminal launcher (type 'run')
├── run.ps1                               # PowerShell terminal launcher (type '.\run')
├── train.py                              # Training & evaluation script for all 3 models
├── train.bat                             # Retrain launcher (type 'train')
├── notebooks/
│   └── loan_approval.ipynb               # Documented Jupyter Notebook
├── app/
│   └── app.py                            # Streamlit web app with 3-model comparison
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Clean exclusions
└── README.md                             # Documentation
```

---

## 🏆 Model Performance Benchmark (10,000 Sample Scale)

| Model | Test Accuracy | ROC-AUC | Precision | Recall | F1-Score | Interpretability |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Decision Tree (`max_depth=4`)** ⭐ | **88.45%** | **83.11%** | **87.83%** | **98.05%** | **92.66%** | **Very High** (Direct Visual Logic & Real ₹ Splits) |
| **SVM (RBF Kernel)** | **88.50%** | 82.37% | 87.79% | **98.18%** | **92.70%** | Geometric (Hyperplane / Black Box) |
| **Naive Bayes (GaussianNB)** | 87.40% | 81.72% | 87.72% | 96.57% | 91.93% | Probabilistic (Feature Independence) |

---

## ☁️ How to Run in Google Colab

The project is fully prepared for Google Colab with **smart multi-path data detection** and **automatic file uploaders**:

### Step-by-Step Instructions:
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select [`Loan_Approval_Google_Colab.ipynb`](file:///c:/Users/abhij/Desktop/Loan/Loan_Approval_Google_Colab.ipynb) (or [`notebooks/loan_approval.ipynb`](file:///c:/Users/abhij/Desktop/Loan/notebooks/loan_approval.ipynb)).
3. In Colab, click the **Files icon (📁)** on the left sidebar and drag-and-drop [`loan.csv`](file:///c:/Users/abhij/Desktop/Loan/Data/loan.csv) into the files area.
   *(Note: If you forget to upload `loan.csv`, the notebook automatically pops up an interactive **Upload** button!)*
4. Click **Runtime** > **Run all** (or press `Ctrl + F9`).
5. All cells run sequentially from data ingestion through 3-model training, evaluation, comparative visualization, interpretability tree plotting, and export to `models/`.

---

## 💻 How to Run Locally in the Terminal

Run the interactive prediction tool directly in your terminal with fast 1-letter shortcuts:

* **In PowerShell**:
  ```powershell
  .\run
  ```
* **In Command Prompt (CMD)**:
  ```cmd
  run
  ```
* **Or directly with Python**:
  ```bash
  python run.py
  ```

### Quick Demo Mode:
```bash
python run.py --demo
```

---

## ⚡ Fast 1-Letter Shortcuts Supported

When running the terminal predictor, you don't need to type full words:

| Field | 1-Letter Shortcut | Supported Variations |
| :--- | :---: | :--- |
| **Gender** | `m` / `f` | `Male`, `Female` |
| **Married** | `y` / `n` | `Yes`, `No` |
| **Dependents** | `0`, `1`, `2`, `3` | `3+` |
| **Education** | `g` / `ng` | `Graduate`, `Not Graduate` |
| **Self Employed** | `y` / `n` | `Yes`, `No` |
| **Credit History** | `1` / `0` | `Good` / `Bad`, `y` / `n` |
| **Property Area** | `u` / `s` / `r` | `Urban`, `Semiurban`, `Rural` |

---

## 🧪 4 Standard Test Cases

### Pass Cases (Expected: APPROVED ✅)
1. **Prime Professional Applicant**:
   - `m`, `y`, `0`, `g`, `n`, `8000`, `2500`, `180`, `360`, `1`, `u`
   - *Verdict*: **Unanimous Approval (3/3)** with >90% confidence.
2. **Stable Moderate-Income Applicant**:
   - `f`, `n`, `0`, `g`, `n`, `5000`, `1500`, `130`, `360`, `1`, `s`
   - *Verdict*: **Unanimous Approval (3/3)** with >80% confidence.

### Fail Cases (Expected: REJECTED ❌)
3. **Severe Credit Default History**:
   - `m`, `y`, `2`, `ng`, `y`, `3000`, `0`, `150`, `360`, `0`, `r`
   - *Verdict*: **Unanimous Rejection (3/3)** due to bad credit (`0.0`).
4. **Low Income & High Loan Request**:
   - `m`, `y`, `0`, `g`, `n`, `100`, `0`, `10000`, `12`, `0`, `r`
   - *Verdict*: **Unanimous Rejection (3/3)** (100% default risk).
