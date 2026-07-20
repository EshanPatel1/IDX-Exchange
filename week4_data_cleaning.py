#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os

# Setup directory
PROCESSED_DIR = r'C:\Users\PC\Downloads\idx-exchange-internship\data\processed'
os.chdir(PROCESSED_DIR)

# Load the Week 2-3 mortgage-rate-enriched datasets
sold = pd.read_csv('sold_with_mortgage_rates.csv', low_memory=False)
listings = pd.read_csv('listings_with_mortgage_rates.csv', low_memory=False)

sold_rows_start = len(sold)
listings_rows_start = len(listings)
print('Sold shape before cleaning:', sold.shape)
print('Listings shape before cleaning:', listings.shape)

# ============================================================
# Step 1 - Convert date fields to datetime format
# ============================================================
date_fields = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate']

for df, name in [(sold, 'SOLD'), (listings, 'LISTINGS')]:
    for col in date_fields:
        before_non_null = df[col].notnull().sum()
        df[col] = pd.to_datetime(df[col], errors='coerce')
        coerced = before_non_null - df[col].notnull().sum()
        print(f'{name}.{col}: converted to datetime (dtype={df[col].dtype}), '
              f'{coerced} unparsable value(s) coerced to NaT')

# ============================================================
# Step 2 - Remove unnecessary or redundant columns
# Columns above 90% missing (per the Week 2-3 missing value reports) carry too little
# signal to be usable and are dropped. Core market fields stay under this threshold
# and are retained even when partially missing -- handled in Step 4 below.
# ============================================================
DROP_THRESHOLD = 90.0

def high_missing_columns(df, threshold=DROP_THRESHOLD):
    pct_missing = df.isnull().mean() * 100
    return pct_missing[pct_missing > threshold].index.tolist()

sold_drop_cols = high_missing_columns(sold)
listings_drop_cols = high_missing_columns(listings)

print(f'Dropping {len(sold_drop_cols)} column(s) from SOLD (>{DROP_THRESHOLD}% missing): {sold_drop_cols}')
print(f'Dropping {len(listings_drop_cols)} column(s) from LISTINGS (>{DROP_THRESHOLD}% missing): {listings_drop_cols}')

sold = sold.drop(columns=sold_drop_cols)
listings = listings.drop(columns=listings_drop_cols)

print('Sold shape after dropping high-missing columns:', sold.shape)
print('Listings shape after dropping high-missing columns:', listings.shape)

# ============================================================
# Step 3 - Ensure numeric fields are properly typed
# ============================================================
numeric_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea', 'LotSizeAcres',
    'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt'
]

for df, name in [(sold, 'SOLD'), (listings, 'LISTINGS')]:
    for col in numeric_fields:
        before_non_null = df[col].notnull().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        coerced = before_non_null - df[col].notnull().sum()
        if coerced:
            print(f'{name}.{col}: {coerced} non-numeric value(s) coerced to NaN')

print('Numeric field dtypes (SOLD):')
print(sold[numeric_fields].dtypes)

# ============================================================
# Step 4 - Handle missing values appropriately
# Core market fields (ClosePrice, LivingArea, DaysOnMarket, etc.) are left as NaN rather
# than imputed -- fabricating a sale price or square footage would misrepresent the
# transaction. The one exception is SOLD.ClosePrice: it is the field that defines a
# closed transaction, so a sold record without one is not usable and is dropped.
# LISTINGS.ClosePrice/CloseDate are expected to be null for still-active listings
# (confirmed in Week 2-3: ~77%/~73% null) and are left as-is.
# ============================================================
before = len(sold)
sold = sold.dropna(subset=['ClosePrice'])
print(f'SOLD: dropped {before - len(sold)} row(s) missing ClosePrice (required field for a closed sale)')

# ============================================================
# Step 5 - Remove invalid numeric values
# ClosePrice <= 0, LivingArea <= 0, DaysOnMarket < 0, and negative Bedrooms/Bathrooms are
# hard data-entry errors, not statistical outliers, so they are removed outright here.
# Statistical outlier trimming (IQR) is handled separately in Week 7.
# ============================================================
def invalid_numeric_mask(df):
    return (
        (df['ClosePrice'].notna() & (df['ClosePrice'] <= 0)) |
        (df['LivingArea'].notna() & (df['LivingArea'] <= 0)) |
        (df['DaysOnMarket'].notna() & (df['DaysOnMarket'] < 0)) |
        (df['BedroomsTotal'].notna() & (df['BedroomsTotal'] < 0)) |
        (df['BathroomsTotalInteger'].notna() & (df['BathroomsTotalInteger'] < 0))
    )

sold_invalid_mask = invalid_numeric_mask(sold)
listings_invalid_mask = invalid_numeric_mask(listings)
print(f'SOLD: {sold_invalid_mask.sum():,} row(s) with invalid numeric values (removed)')
print(f'LISTINGS: {listings_invalid_mask.sum():,} row(s) with invalid numeric values (removed)')

sold = sold[~sold_invalid_mask].copy()
listings = listings[~listings_invalid_mask].copy()

print('Sold shape after removing invalid numeric rows:', sold.shape)
print('Listings shape after removing invalid numeric rows:', listings.shape)

# ============================================================
# Save the Week 4 checkpoint datasets
# Date consistency flags and geographic data quality checks are Week 5 work and
# pick up from these files.
# ============================================================
sold.to_csv('sold_week4_cleaned.csv', index=False)
listings.to_csv('listings_week4_cleaned.csv', index=False)

print('=== Before / after row counts ===')
print(f'SOLD:      {sold_rows_start:,} -> {len(sold):,}')
print(f'LISTINGS:  {listings_rows_start:,} -> {len(listings):,}')
print('=== Date field dtype confirmation (SOLD) ===')
print(sold[date_fields].dtypes)
print('=== Numeric field dtype confirmation (SOLD) ===')
print(sold[numeric_fields].dtypes)
print('Saved sold_week4_cleaned.csv:', sold.shape)
print('Saved listings_week4_cleaned.csv:', listings.shape)
