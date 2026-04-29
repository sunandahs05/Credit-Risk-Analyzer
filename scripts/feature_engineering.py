import os
import pandas as pd
import numpy as np

def run_feature_engineering(data_processed_path="../data/processed/"):
    print("--- 2. Feature Engineering ---")
    master_raw_file = os.path.join(data_processed_path, "master_raw.pkl")
    master_feat_file = os.path.join(data_processed_path, "master_features.pkl")
    
    if os.path.exists(master_feat_file):
        print("Features already engineered.")
        return
        
    print("Loading master_raw.pkl...")
    master = pd.read_pickle(master_raw_file)
    
    print("Engineering features...")
    master['debt_to_income'] = master['amt_annuity'] / (master['amt_income_total'] + 1)
    master['loan_to_income'] = master['amt_credit'] / (master['amt_income_total'] + 1)
    master['repayment_years'] = master['amt_credit'] / (master['amt_annuity'] + 1)
    master['credit_utilisation'] = master['amt_credit'] / (master['amt_goods_price'] + 1)
    
    master['late_payment_rate'] = master['install_late_payments'] / (master['install_count'] + 1)
    master['payment_completeness'] = master['install_payment_ratio']
    master['active_loan_ratio'] = master['bureau_active_loans'] / (master['bureau_loan_count'] + 1)
    master['external_debt_burden'] = master['bureau_total_debt'] / (master['amt_income_total'] + 1)
    
    master['age_years'] = master['days_birth'].abs() / 365
    master['employed_years'] = np.where(master['days_employed'] > 0, 0, master['days_employed'].abs() / 365)
    master['age_employment_ratio'] = master['employed_years'] / (master['age_years'] + 1)
    
    master['is_cash_loan'] = (master['name_contract_type'] == 'Cash loans').astype(int)
    master['is_female'] = (master['code_gender'] == 'F').astype(int)
    master['owns_car'] = (master['flag_own_car'] == 'Y').astype(int)
    master['owns_realty'] = (master['flag_own_realty'] == 'Y').astype(int)
    
    master['higher_education'] = master['name_education_type'].isin(['Higher education', 'Academic degree']).astype(int)
    master['prime_working_age'] = ((master['age_years'] >= 30) & (master['age_years'] <= 55)).astype(int)
    
    doc_cols = [c for c in master.columns if c.startswith('flag_document')]
    master['doc_submission_score'] = master[doc_cols].sum(axis=1)
    
    master['contact_score'] = (
        master['flag_mobil'] + 
        master['flag_emp_phone'] + 
        master['flag_work_phone'] + 
        master['flag_phone'] + 
        master['flag_email']
    )
    
    print("Imputing missing values...")
    bureau_cols = [c for c in master.columns if c.startswith('bureau_')]
    install_cols = [c for c in master.columns if c.startswith('install_')]
    
    for c in bureau_cols + install_cols:
        master[c] = master[c].fillna(0)
        
    master['late_payment_rate'] = master['late_payment_rate'].fillna(0)
    master['active_loan_ratio'] = master['active_loan_ratio'].fillna(0)
    master['external_debt_burden'] = master['external_debt_burden'].fillna(0)
    master['payment_completeness'] = master['payment_completeness'].fillna(1)
    master['employed_years'] = master['employed_years'].fillna(0)
    master['age_employment_ratio'] = master['age_employment_ratio'].fillna(0)
    
    master.to_pickle(master_feat_file)
    print(f"Saved master_features.pkl ({master.shape})")

if __name__ == "__main__":
    run_feature_engineering(data_processed_path="../data/processed/")
