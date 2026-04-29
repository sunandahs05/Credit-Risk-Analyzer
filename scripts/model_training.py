import os
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE
import pmdarima as pm
warnings.filterwarnings('ignore')

model_features = [
    "debt_to_income", "loan_to_income", "repayment_years",
    "late_payment_rate", "payment_completeness", "active_loan_ratio",
    "external_debt_burden", "age_years", "employed_years",
    "doc_submission_score", "contact_score",
    "is_cash_loan", "is_female", "owns_car", "owns_realty",
    "higher_education", "prime_working_age"
]

def ks_stat(y_true, y_prob):
    df = pd.DataFrame({'target': y_true, 'prob': y_prob}).sort_values('prob')
    df['cum_good'] = (1 - df['target']).cumsum() / (1 - df['target']).sum()
    df['cum_bad'] = df['target'].cumsum() / df['target'].sum()
    return np.abs(df['cum_good'] - df['cum_bad']).max()

def run_model_training(data_processed_path="../data/processed/"):
    print("--- 3. Enterprise Model Training (Loopholes Closed) ---")
    
    master = pd.read_pickle(os.path.join(data_processed_path, "master_features.pkl"))
    
    print("Preparing data...")
    df = master[model_features + ['target']].dropna()
    X = df[model_features]
    y = df['target']
    
    # 1. FIX: 3-Way Split (60% Train, 20% Val, 20% Test) to prevent test-set threshold leakage
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 3. FIX: Algorithmic Fairness via Sample Reweighing (Pre-Processing)
    w = np.ones(len(y_train_res))
    # Approximation of female flag due to SMOTE interpolation
    f_idx = X_train_res['is_female'] >= 0.5
    m_idx = ~f_idx
    
    # Reweighing formula (Cast to float to prevent Windows Numpy int32 overflow)
    y_neg = (y_train_res == 0)
    y_pos = (y_train_res == 1)
    
    # Reweighing formula
    w_f_pos = (float(f_idx.sum()) * float(y_pos.sum())) / (len(y_train_res) * max(y_pos[f_idx].sum(), 1))
    w_f_neg = (float(f_idx.sum()) * float(y_neg.sum())) / (len(y_train_res) * max(y_neg[f_idx].sum(), 1))
    w_m_pos = (float(m_idx.sum()) * float(y_pos.sum())) / (len(y_train_res) * max(y_pos[m_idx].sum(), 1))
    w_m_neg = (float(m_idx.sum()) * float(y_neg.sum())) / (len(y_train_res) * max(y_neg[m_idx].sum(), 1))
    
    w[f_idx & y_pos] = w_f_pos
    w[f_idx & y_neg] = w_f_neg
    w[m_idx & y_pos] = w_m_pos
    w[m_idx & y_neg] = w_m_neg
    
    models = {
        "Logistic Regression (Lasso)": LogisticRegression(C=0.01, penalty='l1', solver='liblinear', max_iter=1000, class_weight='balanced'),
        "CART (pruned)": DecisionTreeClassifier(max_depth=6, min_samples_split=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, 
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss'
        )
    }
    
    results = []
    trained_models = {}
    best_thresh = 0.5
    calibrated_xgb = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_res, y_train_res, sample_weight=w)
        
        # 2. FIX: Pure Isotonic Calibration to undo SMOTE probability warping
        if name == "XGBoost":
            # Train an independent Isotonic Calibrator using the Validation Set raw probabilities
            raw_val_probs = model.predict_proba(X_val)[:, 1]
            calibrated_xgb = IsotonicRegression(out_of_bounds='clip')
            calibrated_xgb.fit(raw_val_probs, y_val)
            
            # Tune threshold blindly on the Validation framework (calibrated)
            y_prob_val = calibrated_xgb.predict(raw_val_probs)
            fpr, tpr, thresholds = roc_curve(y_val, y_prob_val)
            j_scores = tpr - fpr
            best_thresh = thresholds[np.argmax(j_scores)]
            
            # Final rigid evaluation on the BLIND Test framework
            raw_test_probs = model.predict_proba(X_test)[:, 1]
            y_prob_test = calibrated_xgb.predict(raw_test_probs)
            y_pred_test = (y_prob_test >= best_thresh).astype(int)
            trained_models[name] = model # raw for SHAP
        else:
            y_prob_test = model.predict_proba(X_test)[:, 1]
            y_pred_test = (y_prob_test >= 0.5).astype(int)
            trained_models[name] = model
            
        auc = roc_auc_score(y_test, y_prob_test)
        ks = ks_stat(y_test, y_prob_test)
        gini = 2 * auc - 1
        f1 = f1_score(y_test, y_pred_test)
        
        results.append({
            "Model": name,
            "AUC": auc, "F1_Score": f1, "KS_Stat": ks, "Gini": gini
        })

    comp_df = pd.DataFrame(results)
    comp_df.to_pickle(os.path.join(data_processed_path, "model_comparison.pkl"))
    
    # --- INSIGHT: Lasso Selection Efficiency ---
    logit_model = trained_models["Logistic Regression (Lasso)"]
    lasso_df = pd.DataFrame({
        "Feature": model_features,
        "Coefficient": logit_model.coef_[0]
    }).sort_values("Coefficient", ascending=False)
    lasso_df.to_pickle(os.path.join(data_processed_path, "lasso_coefs.pkl"))
    
    print("Saving Models...")
    with open(os.path.join(data_processed_path, "logit_model.pkl"), "wb") as f:
        pickle.dump(logit_model, f)
    with open(os.path.join(data_processed_path, "xgb_model.pkl"), "wb") as f:
        pickle.dump(trained_models["XGBoost"], f)
    with open(os.path.join(data_processed_path, "xgb_calibrated.pkl"), "wb") as f:
        pickle.dump(calibrated_xgb, f)
    with open(os.path.join(data_processed_path, "cart_model.pkl"), "wb") as f:
        pickle.dump(trained_models["CART (pruned)"], f)
        
    with open(os.path.join(data_processed_path, "best_threshold.pkl"), "wb") as f:
        pickle.dump(best_thresh, f)
        
    print("Calculating Full Portfolio Bias Audit (DIR for All Attributes)...")
    test_df = X_test.copy()
    raw_test_p = trained_models["XGBoost"].predict_proba(X_test)[:, 1]
    test_df['prob'] = calibrated_xgb.predict(raw_test_p)
    test_df['pred_reject'] = (test_df['prob'] >= best_thresh).astype(int)
    test_df['pred_approve'] = 1 - test_df['pred_reject'] # approval rate focus
    
    # Full Portfolio Bias Loop
    bias_results = []
    
    # 1. Binary Features
    binary_attrs = ["is_female", "higher_education", "prime_working_age", "owns_realty", "owns_car", "is_cash_loan"]
    for attr in binary_attrs:
        dr_privileged = test_df[test_df[attr] == 1]['pred_approve'].mean()
        dr_unprivileged = test_df[test_df[attr] == 0]['pred_approve'].mean()
        dir_score = dr_unprivileged / max(dr_privileged, 1e-6)
        bias_results.append({"Attribute": attr, "Category": "Categorical", "DIR": dir_score})
    
    # 2. Continuous Quantile Bias (Top 20% vs Bottom 20%)
    cont_attrs = ["age_years", "debt_to_income", "employed_years", "late_payment_rate"]
    for attr in cont_attrs:
        q_low = test_df[attr].quantile(0.2)
        q_high = test_df[attr].quantile(0.8)
        dr_high = test_df[test_df[attr] >= q_high]['pred_approve'].mean()
        dr_low = test_df[test_df[attr] <= q_low]['pred_approve'].mean()
        # For DTI/LateRate, low is privileged. For Age/Emp, high is privileged.
        if attr in ["debt_to_income", "late_payment_rate"]:
            dir_score = dr_high / max(dr_low, 1e-6) # higher risk should have lower approval
        else:
            dir_score = dr_low / max(dr_high, 1e-6)
        bias_results.append({"Attribute": f"{attr} (Quantile)", "Category": "Continuous", "DIR": dir_score})
    
    fairness_audit = pd.DataFrame(bias_results)
    fairness_audit.to_pickle(os.path.join(data_processed_path, "fairness_audit_full.pkl"))
    
    # Legacy compatibility metrics
    fairness_summary = {
        "dir_gender": bias_results[0]["DIR"],
        "dir_education": bias_results[1]["DIR"],
        "dir_gender_rw": bias_results[0]["DIR"] 
    }
    with open(os.path.join(data_processed_path, "fairness_summary.pkl"), "wb") as f:
        pickle.dump(fairness_summary, f)
        
    print("Model compilation complete!")
    
if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(script_dir, "../data/processed"))
    run_model_training(data_processed_path=processed_dir)
