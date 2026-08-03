#!/usr/bin/env python
# coding: utf-8

import os

import pandas as pd

# Setup directories
PROCESSED_DIR = r'C:\Users\PC\Downloads\idx-exchange-internship\data\processed'
os.chdir(PROCESSED_DIR)

# Load the Week 6 feature-engineered datasets
sold = pd.read_csv('sold_engineered.csv', low_memory=False)
listings = pd.read_csv('listings_engineered.csv', low_memory=False)

print('Sold shape before outlier detection:', sold.shape)
print('Listings shape before outlier detection:', listings.shape)

# ============================================================
# Step 1 - IQR outlier detection
# Records are flagged, not dropped in place, so the raw engineered datasets stay intact
# for anyone who wants the full record. A separate, filtered "analysis-ready" dataset is
# built afterward with flagged rows removed.
# ============================================================
IQR_COLUMNS = ['ClosePrice', 'price_per_sqft', 'close_to_original_list_ratio', 'DaysOnMarket']


def iqr_bounds(series):
    """Return the (lower, upper) Tukey fences for a series, computed from its
    non-null values."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def add_outlier_flags(df, name, columns=IQR_COLUMNS):
    flag_cols = []
    for col in columns:
        lower, upper = iqr_bounds(df[col])
        flag_col = f'{col}_outlier_flag'
        # NaN comparisons evaluate to False, so rows with a missing value for this
        # column are never flagged as an outlier on it.
        df[flag_col] = (df[col] < lower) | (df[col] > upper)
        flag_cols.append(flag_col)
        print(f'{name}.{col}: IQR bounds [{lower:,.2f}, {upper:,.2f}], '
              f'{df[flag_col].sum():,} outlier(s) flagged')

    df['is_outlier'] = df[flag_cols].any(axis=1)
    print(f'{name}: {df["is_outlier"].sum():,} of {len(df):,} rows flagged as an outlier '
          f'on at least one metric ({df["is_outlier"].mean():.1%})')
    return df


sold = add_outlier_flags(sold, 'SOLD')
listings = add_outlier_flags(listings, 'LISTINGS')

# ============================================================
# Step 2 - Save flagged (raw, all rows preserved) and filtered (analysis-ready) datasets
# ============================================================
sold.to_csv('sold_flagged.csv', index=False)
listings.to_csv('listings_flagged.csv', index=False)
print('Saved sold_flagged.csv:', sold.shape)
print('Saved listings_flagged.csv:', listings.shape)

sold_filtered = sold[~sold['is_outlier']].copy()
listings_filtered = listings[~listings['is_outlier']].copy()

sold_filtered.to_csv('sold_outliers_removed.csv', index=False)
listings_filtered.to_csv('listings_outliers_removed.csv', index=False)

print('=== Before / after row counts ===')
print(f'SOLD:      {len(sold):,} -> {len(sold_filtered):,}')
print(f'LISTINGS:  {len(listings):,} -> {len(listings_filtered):,}')
print('Saved sold_outliers_removed.csv:', sold_filtered.shape)
print('Saved listings_outliers_removed.csv:', listings_filtered.shape)
