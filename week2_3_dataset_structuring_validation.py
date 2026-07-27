#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Setup directories
RAW_DIR = r'C:\Users\PC\Downloads\idx-exchange-internship\data\raw'
PROCESSED_DIR = r'C:\Users\PC\Downloads\idx-exchange-internship\data\processed'
os.chdir(PROCESSED_DIR)

# Load the Week 1 combined, Residential-filtered datasets
sold = pd.read_csv('combined_residential_sold.csv', low_memory=False)
listings = pd.read_csv('combined_residential_listings.csv', low_memory=False)

print('Sold shape:', sold.shape)
print('Listings shape:', listings.shape)

# Property type documentation: reload PropertyType from the raw (pre-filter) monthly files
# to confirm what categories exist before the Week 1 Residential filter was applied
sold_files = sorted(glob.glob(os.path.join(RAW_DIR, 'CRMLSSold*.csv')))
listing_files = sorted(glob.glob(os.path.join(RAW_DIR, 'CRMLSListing*.csv')))

sold_types_raw = pd.concat(
    [pd.read_csv(f, usecols=['PropertyType'], low_memory=False) for f in sold_files],
    ignore_index=True
)
listing_types_raw = pd.concat(
    [pd.read_csv(f, usecols=['PropertyType'], low_memory=False) for f in listing_files],
    ignore_index=True
)

print('--- Sold: PropertyType value counts (pre-filter) ---')
print(sold_types_raw['PropertyType'].value_counts(dropna=False))
print()
print('--- Listings: PropertyType value counts (pre-filter) ---')
print(listing_types_raw['PropertyType'].value_counts(dropna=False))

# Residential share of each raw dataset
sold_res_share = (sold_types_raw['PropertyType'] == 'Residential').mean()
listing_res_share = (listing_types_raw['PropertyType'] == 'Residential').mean()
print(f'Sold: Residential share = {sold_res_share:.2%} ({(sold_types_raw.PropertyType == "Residential").sum():,} of {len(sold_types_raw):,})')
print(f'Listings: Residential share = {listing_res_share:.2%} ({(listing_types_raw.PropertyType == "Residential").sum():,} of {len(listing_types_raw):,})')

# Confirm the filtering logic already applied in Week 1: both combined datasets
# should now contain only PropertyType == 'Residential'
print('Sold filtered PropertyType values:', sold['PropertyType'].unique())
print('Listings filtered PropertyType values:', listings['PropertyType'].unique())
assert (sold['PropertyType'] == 'Residential').all()
assert (listings['PropertyType'] == 'Residential').all()
print('Filter confirmed: both datasets contain Residential records only.')

# Data quality check: the raw Listing export contains duplicate columns (e.g. 'PropertyType.1',
# 'ListPrice.1') carried over from the source CSVs. Confirm the duplicates are exact copies, then drop them.
dupe_cols = [c for c in listings.columns if c.endswith('.1')]
print(f'Found {len(dupe_cols)} duplicate-suffixed columns in listings: {dupe_cols}')

for c in dupe_cols:
    base = c[:-2]
    identical = (listings[base].astype(str) == listings[c].astype(str)).all()
    print(f'  {base} vs {c} -> identical: {identical}')

listings = listings.drop(columns=dupe_cols)
print('Listings shape after dropping duplicate columns:', listings.shape)

# Dataset understanding: shape and column data types
print('=== SOLD ===')
print('Rows:', sold.shape[0], '| Columns:', sold.shape[1])
print(sold.dtypes.value_counts())
print()
print('=== LISTINGS ===')
print('Rows:', listings.shape[0], '| Columns:', listings.shape[1])
print(listings.dtypes.value_counts())

# Separate market-analysis fields (used for metrics/dashboards) from metadata/administrative fields
market_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea', 'LotSizeAcres',
    'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt',
    'PropertyType', 'PropertySubType', 'CountyOrParish', 'MLSAreaMajor', 'City', 'PostalCode',
    'CloseDate', 'ListingContractDate', 'PurchaseContractDate', 'ContractStatusChangeDate',
    'Latitude', 'Longitude', 'ListOfficeName', 'BuyerOfficeName'
]
metadata_fields = [c for c in sold.columns if c not in market_fields]
print(f'Market analysis fields ({len(market_fields)}):')
print(market_fields)
print()
print(f'Metadata / administrative fields ({len(metadata_fields)}):')
print(metadata_fields)

# Missing value analysis: counts and percentages per column
def missing_report(df, name):
    report = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().mean() * 100).round(2)
    }).sort_values('missing_pct', ascending=False)
    high_missing = report[report['missing_pct'] > 90]
    print(f'--- {name}: top 15 columns by missing % ---')
    print(report.head(15))
    print(f'\n{name}: {len(high_missing)} column(s) above 90% missing')
    if len(high_missing):
        print(high_missing)
    return report

sold_missing = missing_report(sold, 'SOLD')
listings_missing = missing_report(listings, 'LISTINGS')

# Decision: retain core fields (ClosePrice, LivingArea, DaysOnMarket, etc.) even if partially missing.
# Only columns above the 90% missing threshold are candidates for dropping; save the full null-count
# tables so the decision is documented and auditable.
sold_missing.to_csv('sold_missing_value_report.csv')
listings_missing.to_csv('listings_missing_value_report.csv')
print('Saved missing value reports.')

# Numeric distribution review: percentile summary for the key numeric fields
numeric_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea', 'LotSizeAcres',
    'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt'
]
percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
dist_summary = sold[numeric_fields].describe(percentiles=percentiles).T
print(dist_summary)

# Histograms for the key numeric fields (Sold dataset)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, col in zip(axes.flatten(), numeric_fields):
    sold[col].dropna().plot(kind='hist', bins=50, ax=ax, title=col)
plt.tight_layout()
plt.savefig('sold_numeric_histograms.png')
plt.close(fig)

# Boxplots for the key numeric fields (Sold dataset)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, col in zip(axes.flatten(), numeric_fields):
    sold.boxplot(column=col, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig('sold_numeric_boxplots.png')
plt.close(fig)

# Extreme outlier identification (flagging only -- removal is handled in Week 7 with IQR)
print('Values outside the 1st-99th percentile range (candidates for later outlier handling):')
for col in numeric_fields:
    p01, p99 = sold[col].quantile([0.01, 0.99])
    n_extreme = (~sold[col].between(p01, p99)).sum()
    print(f'  {col}: {n_extreme:,} rows outside [{p01:,.1f}, {p99:,.1f}]')

# Required deliverable: distribution summary (min, max, mean, median, percentiles) for
# ClosePrice, LivingArea, DaysOnMarket
required_fields = ['ClosePrice', 'LivingArea', 'DaysOnMarket']
required_summary = sold[required_fields].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
required_summary.to_csv('sold_numeric_distribution_summary.csv')
print(required_summary)

# Note on scope: the Suggested Intern Questions above (median/avg close price, % over/under list,
# date consistency) are computed on the SOLD dataset only, since it is the authoritative closed-transaction
# table. The LISTINGS dataset does carry ClosePrice/CloseDate columns, but they are populated only for
# listings that have since closed (~77% and ~73% null respectively) -- those closed records are already
# captured in SOLD, so re-deriving these metrics from LISTINGS would be redundant and partially null.
print('Listings ClosePrice null %:', round(listings["ClosePrice"].isnull().mean() * 100, 1))
print('Listings CloseDate null %:', round(listings["CloseDate"].isnull().mean() * 100, 1))
print('Listings MlsStatus breakdown:')
print(listings["MlsStatus"].value_counts(dropna=False))



# 1. Residential vs. other property type share (computed above from raw pre-filter data)
print(f'1. Residential share -- Sold: {sold_res_share:.2%}, Listings: {listing_res_share:.2%}')

# 2. Median and average close prices
print('2. Median close price: ${:,.0f}'.format(sold['ClosePrice'].median()))
print('   Average close price: ${:,.0f}'.format(sold['ClosePrice'].mean()))

# 3. Days on Market distribution
print('3. Days on Market distribution:')
print(sold['DaysOnMarket'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]))

# 4. Percentage of homes sold above vs. below list price
valid_price = sold.dropna(subset=['ClosePrice', 'ListPrice'])
above = (valid_price['ClosePrice'] > valid_price['ListPrice']).mean()
below = (valid_price['ClosePrice'] < valid_price['ListPrice']).mean()
at_list = (valid_price['ClosePrice'] == valid_price['ListPrice']).mean()
print(f'4. Sold above list: {above:.2%} | Sold below list: {below:.2%} | Sold at list: {at_list:.2%}')

# 5. Date consistency: does CloseDate ever precede ListingContractDate?
close_dt = pd.to_datetime(sold['CloseDate'], errors='coerce')
listing_dt = pd.to_datetime(sold['ListingContractDate'], errors='coerce')
inconsistent = (close_dt < listing_dt).sum()
print(f'5. Records where CloseDate precedes ListingContractDate: {inconsistent:,} '
      f'({inconsistent / len(sold):.2%} of sold records)')

# 6. Counties with the highest median close prices (min 30 sales to avoid noise from tiny counties)
county_counts = sold['CountyOrParish'].value_counts()
eligible_counties = county_counts[county_counts >= 30].index
county_median = (
    sold[sold['CountyOrParish'].isin(eligible_counties)]
    .groupby('CountyOrParish')['ClosePrice']
    .median()
    .sort_values(ascending=False)
)
print('6. Top 10 counties by median close price (>=30 sales):')
print(county_median.head(10))

# Segment summary tables (feeds the Week 6 feature engineering step)
segment_by_county = sold.groupby('CountyOrParish').agg(
    median_close_price=('ClosePrice', 'median'),
    avg_days_on_market=('DaysOnMarket', 'mean'),
    n_sales=('ClosePrice', 'count')
).sort_values('n_sales', ascending=False)
segment_by_county.to_csv('sold_segment_summary_by_county.csv')
print(segment_by_county.head(10))

# Save the structured/validated datasets as new CSVs (Weeks 2-3 checkpoint)
sold.to_csv('sold_structured_validated.csv', index=False)
listings.to_csv('listings_structured_validated.csv', index=False)
print('Saved sold_structured_validated.csv:', sold.shape)
print('Saved listings_structured_validated.csv:', listings.shape)

# ============================================================
# Mortgage Rate Enrichment
# Fetch the FRED MORTGAGE30US series, resample to monthly, and merge
# onto both combined datasets using a year_month key.
# ============================================================

# Step 1 - Fetch the mortgage rate data from FRED
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=['observation_date'])
mortgage.columns = ['date', 'rate_30yr_fixed']
print('Fetched', len(mortgage), 'weekly mortgage rate observations.')
print(mortgage.tail())

# Step 2 - Resample weekly rates to monthly averages
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (
    mortgage.groupby('year_month')['rate_30yr_fixed']
    .mean()
    .reset_index()
)
print(mortgage_monthly.tail())

# Step 3 - Create a matching year_month key on the MLS datasets
# Sold dataset -- key off CloseDate
sold['year_month'] = pd.to_datetime(sold['CloseDate'], errors='coerce').dt.to_period('M')

# Listings dataset -- key off ListingContractDate
listings['year_month'] = pd.to_datetime(listings['ListingContractDate'], errors='coerce').dt.to_period('M')

# Step 4 - Merge
sold_with_rates = sold.merge(mortgage_monthly, on='year_month', how='left')
listings_with_rates = listings.merge(mortgage_monthly, on='year_month', how='left')

# Step 5 - Validate the merge: rate should not be null for any row
sold_nulls = sold_with_rates['rate_30yr_fixed'].isnull().sum()
listings_nulls = listings_with_rates['rate_30yr_fixed'].isnull().sum()
print('Sold rows with null rate after merge:', sold_nulls)
print('Listings rows with null rate after merge:', listings_nulls)
assert sold_nulls == 0, 'Unmatched rows found in sold_with_rates'
assert listings_nulls == 0, 'Unmatched rows found in listings_with_rates'
print('Validation passed: no null mortgage rate values after merge.')

# Preview
print(sold_with_rates[['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']].head())

# Save enriched datasets
sold_with_rates.to_csv('sold_with_mortgage_rates.csv', index=False)
listings_with_rates.to_csv('listings_with_mortgage_rates.csv', index=False)
print('Saved sold_with_mortgage_rates.csv:', sold_with_rates.shape)
print('Saved listings_with_mortgage_rates.csv:', listings_with_rates.shape)
