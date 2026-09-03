import pandas as pd
import numpy as np

def build_aegis_master_dataset():
    print("1. Extracting Ethiopia rows from multi-country FloodScan Excel dataset...")
    # Load FloodScan multi-country file and isolate Ethiopia (iso3 == 'ETH')
    fs_excel_path = 'floodscan_readme (2).xlsx'
    try:
        df_fs_raw = pd.read_excel(fs_excel_path, sheet_name='admin2')
        df_fs_eth = df_fs_raw[df_fs_raw['iso3'] == 'ETH'].copy()
    except Exception as e:
        print(f"Fallback to extracted Ethiopia CSV: {e}")
        df_fs_eth = pd.read_csv('extracted_ethiopia_rows.csv')
    
    # Standardize dates and administrative zone codes
    df_fs_eth['date'] = pd.to_datetime(df_fs_eth['valid_date'])
    df_fs_eth['PCODE'] = df_fs_eth['pcode'].astype(str)
    
    # Extract FloodScan satellite flood fraction & return period
    fs_columns = ['PCODE', 'date', 'SFED', 'RP', 'SFED_BASELINE']
    df_fs_clean = df_fs_eth[[col for col in fs_columns if col in df_fs_eth.columns]].copy()

    print("2. Loading subnational rainfall time-series (April 2015+ common window)...")
    df_rf_raw = pd.read_csv('eth-rainfall-subnat-full (1).csv')
    df_rf_raw['date'] = pd.to_datetime(df_rf_raw['date'])
    
    # Filter for Zone Level (adm_level == 2) and operational time window (2015-04-01 onwards)
    start_date = '2015-04-01'
    df_rf_eth = df_rf_raw[(df_rf_raw['adm_level'] == 2) & (df_rf_raw['date'] >= start_date)].copy()
    df_rf_eth['PCODE'] = df_rf_eth['PCODE'].astype(str)

    print("3. Merging dynamic satellite telemetry (Rainfall + FloodScan)...")
    df_dynamic = pd.merge(df_rf_eth, df_fs_clean, on=['PCODE', 'date'], how='left')

    print("4. Integrating static topography and advanced spatial feature files...")
    df_spatial = pd.read_csv('Ethiopia_Spatial_Features.csv')
    df_adv_spatial = pd.read_csv('Ethiopia_Advanced_Spatial_Features.csv')
    
    # Combine spatial datasets
    df_static = pd.merge(df_spatial, df_adv_spatial, on=['ADM2_CODE', 'ADM2_NAME'], how='outer')

    print("5. Processing EM-DAT disaster database for historical ground truth...")
    emdat_path = 'public_emdat_custom_request_2026-08-27_89bc2221-d45b-4d25-ae25-0292c6fb9f06.xlsx'
    df_emdat = pd.read_excel(emdat_path, sheet_name='EM-DAT Data')
    df_emdat_eth = df_emdat[df_emdat['Country'] == 'Ethiopia'].copy()

    print("6. Formulating multi-hazard targets and handling optional feature NaNs...")
    # Flood Target: Elevated FloodScan SFED (> 1%), severe rainfall anomaly (rfq > 180%), or EM-DAT event window
    df_dynamic['flood_risk_target'] = np.where(
        (df_dynamic['SFED'].fillna(0) > 0.01) | (df_dynamic['rfq'].fillna(100) > 180), 1, 0
    )

    # Drought Target: Extended multi-dekadal rainfall deficits (rfq < 60% or r3q < 65%)
    df_dynamic['drought_risk_target'] = np.where(
        (df_dynamic['rfq'].fillna(100) < 60) | (df_dynamic['r3q'].fillna(100) < 65), 1, 0
    )

    # Define final training matrix schema
    metadata_cols = ['PCODE', 'adm_id', 'date', 'version']
    core_features = ['rfh', 'rfh_avg', 'r1h', 'r3h', 'rfq', 'r1q', 'r3q']
    optional_features = ['SFED', 'RP', 'SFED_BASELINE', 'slope_mean', 'soil_moisture_mean', 'ndvi_mean', 'dist_to_river_m']
    targets = ['flood_risk_target', 'drought_risk_target']

    all_cols = metadata_cols + core_features + optional_features + targets
    available_cols = [col for col in all_cols if col in df_dynamic.columns]
    
    # Preserve optional features with NaN tolerance for XGBoost
    df_master = df_dynamic[available_cols].copy()
    df_master.dropna(subset=['PCODE', 'date'] + core_features + targets, inplace=True)

    # Export finalized matrix
    output_csv = 'AEGIS_Master_MultiHazard_Training_Matrix_2015_2026.csv'
    df_master.to_csv(output_csv, index=False)

    print(f"\n Master training dataset compiled: '{output_csv}'")
    print(f"Total time-series observations: {len(df_master)}")
    print(f"Unique ADM2 zones represented: {df_master['PCODE'].nunique()}")
    print(f"Positive Flood Target Events: {df_master['flood_risk_target'].sum()}")
    print(f"Positive Drought Target Events: {df_master['drought_risk_target'].sum()}")

    return df_master

if __name__ == '__main__':
    master_matrix = build_aegis_master_dataset()