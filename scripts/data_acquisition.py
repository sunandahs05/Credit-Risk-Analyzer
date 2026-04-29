import os
import pandas as pd
import numpy as np
import wbgapi as wb

def run_data_acquisition(data_raw_path="../../data/raw/", data_processed_path="../data/processed/"):
    print("--- 1. Data Acquisition ---")
    os.makedirs(data_processed_path, exist_ok=True)
    
    master_raw_file = os.path.join(data_processed_path, "master_raw.pkl")
    rbi_npa_file = os.path.join(data_processed_path, "rbi_npa_clean.pkl")
    
    if os.path.exists(master_raw_file) and os.path.exists(rbi_npa_file):
        print("Data already processed.")
        return

    print("Loading raw tables...")
    app_train = pd.read_csv(os.path.join(data_raw_path, "application_train.csv"))
    bureau = pd.read_csv(os.path.join(data_raw_path, "bureau.csv"))
    installments = pd.read_csv(os.path.join(data_raw_path, "installments_payments.csv"))
    
    # Pre-aggregations
    print("Aggregating bureau...")
    bureau['is_active'] = (bureau['CREDIT_ACTIVE'] == 'Active').astype(int)
    bureau['is_closed'] = (bureau['CREDIT_ACTIVE'] == 'Closed').astype(int)
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
        bureau_loan_count=('SK_ID_BUREAU', 'count'),
        bureau_active_loans=('is_active', 'sum'),
        bureau_closed_loans=('is_closed', 'sum'),
        bureau_avg_credit_days=('DAYS_CREDIT', 'mean'),
        bureau_max_overdue=('AMT_CREDIT_MAX_OVERDUE', 'max'),
        bureau_total_debt=('AMT_CREDIT_SUM_DEBT', 'sum')
    ).reset_index()
    bureau_agg['bureau_max_overdue'] = bureau_agg['bureau_max_overdue'].replace([np.inf, -np.inf], np.nan)
    
    print("Aggregating installments...")
    installments['is_late'] = (installments['DAYS_INSTALMENT'] < installments['DAYS_ENTRY_PAYMENT']).astype(int)
    installments['days_late'] = np.maximum(installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT'], 0)
    installments['payment_ratio'] = installments['AMT_PAYMENT'] / (installments['AMT_INSTALMENT'] + 1e-6)
    installments_agg = installments.groupby('SK_ID_CURR').agg(
        install_count=('SK_ID_PREV', 'count'),
        install_late_payments=('is_late', 'sum'),
        install_avg_days_late=('days_late', 'mean'),
        install_payment_ratio=('payment_ratio', 'mean')
    ).reset_index()
    
    print("Merging to master...")
    master = app_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    master = master.merge(installments_agg, on='SK_ID_CURR', how='left')
    master.columns = [col.lower() for col in master.columns]
    
    print(f"Master table built: {master.shape}")
    master.to_pickle(master_raw_file)
    print("Saved master_raw.pkl")
    
    print("Fetching RBI NPA data from World Bank (FB.AST.NPER.ZS)...")
    try:
        # Try fetching real data
        npa_df = wb.data.DataFrame('FB.AST.NPER.ZS', 'IN', time=range(2000, 2025)).reset_index()
        npa_df = npa_df.melt(id_vars=['series', 'economy'], var_name='year', value_name='npa_ratio')
        npa_df['year'] = npa_df['year'].str.replace('YR', '').astype(int)
        npa_df = npa_df.dropna(subset=['npa_ratio']).sort_values('year')
        npa_df.to_pickle(rbi_npa_file)
        print("Saved rbi_npa_clean.pkl")
    except Exception as e:
        print(f"Failed to fetch NPA data: {e}, falling back to static data")
        
        # Hardcoded realistic values for Indian NPA ratio (2000-2024)
        npa_ratios = [12.8, 11.4, 10.4, 8.8, 7.2, 5.2, 3.3, 2.3, 2.3, 2.4, 2.5, 2.9, 
                      3.4, 4.1, 4.3, 5.9, 9.3, 10.0, 11.2, 9.1, 8.2, 7.3, 5.9, 3.9, 2.8]
                      
        npa_df = pd.DataFrame({
            'year': list(range(2000, 2025)),
            'npa_ratio': npa_ratios
        })
        npa_df.to_pickle(rbi_npa_file)
        print("Saved static rbi_npa_clean.pkl")

if __name__ == "__main__":
    # Adjust paths if running directly from scripts folder
    run_data_acquisition(data_raw_path="../../data/raw/", data_processed_path="../data/processed/")
