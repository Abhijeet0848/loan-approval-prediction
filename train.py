"""
Loan Approval Prediction - Model Training Script
Trains the 3 production models:
1. Naive Bayes (GaussianNB)
2. Support Vector Machine (SVC with RBF Kernel)
3. Decision Tree (DecisionTreeClassifier with max_depth=4)
"""

import os
import warnings
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("1. Loading and Preparing Data")
    print("=" * 60)
    
    data_path = os.path.join(os.path.dirname(__file__), 'Data', 'loan.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Features and Target
    X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # Train/Test Split (80% train, 20% test stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

    print("\n" + "=" * 60)
    print("2. Building Preprocessing Pipelines")
    print("=" * 60)
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    # Standard scaled preprocessor (for Naive Bayes and SVM)
    preprocessor_scaled = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

    # Unscaled preprocessor (for Decision Tree - preserves real-world dollar and credit units)
    preprocessor_tree = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

    print("\n" + "=" * 60)
    print("3. Training the Three Models")
    print("=" * 60)

    # 1. Naive Bayes
    nb_model = Pipeline([
        ('preprocessor', preprocessor_scaled),
        ('classifier', GaussianNB())
    ])
    nb_model.fit(X_train, y_train)
    print("✅ Model 1: Naive Bayes trained successfully.")

    # 2. Support Vector Machine (RBF Kernel)
    svm_model = Pipeline([
        ('preprocessor', preprocessor_scaled),
        ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
    ])
    svm_model.fit(X_train, y_train)
    print("✅ Model 2: SVM (RBF) trained successfully.")

    # 3. Decision Tree (unscaled thresholds for full visual explainability)
    dt_model = Pipeline([
        ('preprocessor', preprocessor_tree),
        ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
    ])
    dt_model.fit(X_train, y_train)
    print("✅ Model 3: Decision Tree (max_depth=4) trained successfully.")

    print("\n" + "=" * 60)
    print("4. Evaluating on Test Set (20% Holdout)")
    print("=" * 60)

    models = {
        'Naive Bayes': nb_model,
        'SVM (RBF)': svm_model,
        'Decision Tree': dt_model
    }

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n--- {name} ---")
        print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
        print(f"Confusion Matrix (TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}):")
        print(cm)

    print("\n" + "=" * 60)
    print("5. Saving Models to 'models/' Directory")
    print("=" * 60)

    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    nb_save_path = os.path.join(models_dir, 'loan_model_nb.pkl')
    svm_save_path = os.path.join(models_dir, 'loan_model_svm.pkl')
    dt_save_path = os.path.join(models_dir, 'loan_model_dt.pkl')
    
    joblib.dump(nb_model, nb_save_path)
    joblib.dump(svm_model, svm_save_path)
    joblib.dump(dt_model, dt_save_path)
    print(f"Saved: {nb_save_path}")
    print(f"Saved: {svm_save_path}")
    print(f"Saved: {dt_save_path}")
    print("\nTraining complete! All 3 models (Naive Bayes, SVM, Decision Tree) saved.")


if __name__ == '__main__':
    main()
