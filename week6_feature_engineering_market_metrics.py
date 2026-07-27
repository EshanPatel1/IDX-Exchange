#!/usr/bin/env python
# coding: utf-8

import os
import urllib.request

import pandas as pd
import geopandas as gpd

# Setup directories
RAW_DIR = r'C:\Users\PC\Downloads\idx-exchange-internship\data\raw'
PROCESSED_DIR = r'C:\Users\PC\Downloads\idx-exchange-internship\data\processed'
os.chdir(PROCESSED_DIR)

# Load the Week 4-5 cleaned, analysis-ready datasets
sold = pd.read_csv('sold_cleaned.csv', low_memory=False)
listings = pd.read_csv('listings_cleaned.csv', low_memory=False)

print('Sold shape before feature engineering:', sold.shape)
print('Listings shape before feature engineering:', listings.shape)


def safe_divide(numerator, denominator):
    """Divide element-wise, treating a zero/negative/missing denominator as NaN
    instead of raising or returning inf."""
    return numerator / denominator.where(denominator > 0)


# ============================================================
# Step 1 - Engineered market metrics
#
# Note on Price Ratio vs. Close-to-Original-List Ratio: the handbook lists the same
# formula (ClosePrice / OriginalListPrice) for both. To make the two columns actually
# distinct and useful, this script interprets them as:
#   price_ratio                  = ClosePrice / ListPrice          (vs. the list price in
#                                   effect at contract acceptance -- negotiation strength)
#   close_to_original_list_ratio = ClosePrice / OriginalListPrice   (vs. the very first list
#                                   price -- captures the full markdown history)
# For listings with no price reductions, ListPrice == OriginalListPrice and the two ratios
# are identical.
# ============================================================
def add_market_metrics(df, name):
    df['CloseDate'] = pd.to_datetime(df['CloseDate'], errors='coerce')
    df['ListingContractDate'] = pd.to_datetime(df['ListingContractDate'], errors='coerce')
    df['PurchaseContractDate'] = pd.to_datetime(df['PurchaseContractDate'], errors='coerce')

    df['price_ratio'] = safe_divide(df['ClosePrice'], df['ListPrice'])
    df['close_to_original_list_ratio'] = safe_divide(df['ClosePrice'], df['OriginalListPrice'])
    df['price_per_sqft'] = safe_divide(df['ClosePrice'], df['LivingArea'])

    # Time-series keys derived from CloseDate
    df['close_year'] = df['CloseDate'].dt.year
    df['close_month'] = df['CloseDate'].dt.month
    df['close_yrmo'] = df['CloseDate'].dt.to_period('M').astype(str)
    df.loc[df['CloseDate'].isna(), 'close_yrmo'] = pd.NA

    # Contract timeline durations (days)
    df['listing_to_contract_days'] = (df['PurchaseContractDate'] - df['ListingContractDate']).dt.days
    df['contract_to_close_days'] = (df['CloseDate'] - df['PurchaseContractDate']).dt.days

    new_cols = [
        'price_ratio', 'close_to_original_list_ratio', 'price_per_sqft',
        'close_year', 'close_month', 'close_yrmo',
        'listing_to_contract_days', 'contract_to_close_days'
    ]
    print(f'{name}: engineered metric non-null counts (of {len(df):,} rows)')
    for col in new_cols:
        print(f'  {col}: {df[col].notna().sum():,}')
    return df


sold = add_market_metrics(sold, 'SOLD')
listings = add_market_metrics(listings, 'LISTINGS')

# ============================================================
# Step 2 - School district enrichment
# Spatial-join each record's Latitude/Longitude against the CA Dept. of Education's
# 2024-25 school district boundary polygons to attach the district each property falls in.
# Source: https://data.ca.gov/dataset/california-school-district-areas-2024-25
#         (resource 7dfaf005-58eb-45db-93b1-7aff091b2172, downloaded here as GeoJSON)
# ============================================================
SCHOOL_DISTRICTS_GEOJSON = os.path.join(RAW_DIR, 'ca_school_district_areas_2024_25.geojson')
SCHOOL_DISTRICTS_URL = (
    'https://hub.arcgis.com/api/v3/datasets/b0e3b936426a47ce9d9a2e77e2bb86cc_0/'
    'downloads/data?format=geojson&spatialRefId=4326&where=1%3D1'
)

if not os.path.exists(SCHOOL_DISTRICTS_GEOJSON):
    print('Downloading CA school district boundaries from data.ca.gov...')
    urllib.request.urlretrieve(SCHOOL_DISTRICTS_URL, SCHOOL_DISTRICTS_GEOJSON)

districts = gpd.read_file(SCHOOL_DISTRICTS_GEOJSON)[['DistrictName', 'DistrictType', 'geometry']]
districts = districts.to_crs(epsg=4326)
print(f'Loaded {len(districts):,} school district boundary polygons.')

# Reuse the same plausible-CA-coordinate bounds applied in Week 4-5's geographic data checks
CA_LAT_RANGE = (32.0, 42.5)
CA_LON_RANGE = (-125.0, -114.0)


def add_school_district(df, name):
    valid = (
        df['Latitude'].notna() & df['Longitude'].notna() &
        (df['Latitude'] != 0) & (df['Longitude'] != 0) &
        df['Latitude'].between(*CA_LAT_RANGE) & df['Longitude'].between(*CA_LON_RANGE)
    )
    print(f'{name}: {valid.sum():,} of {len(df):,} rows have plausible CA coordinates '
          f'eligible for the spatial join')

    points = gpd.GeoDataFrame(
        index=df.index[valid],
        geometry=gpd.points_from_xy(df.loc[valid, 'Longitude'], df.loc[valid, 'Latitude']),
        crs='EPSG:4326'
    )
    joined = gpd.sjoin(points, districts, how='left', predicate='within')
    # A point sitting exactly on a shared boundary could match more than one polygon;
    # keep the first match so row counts don't inflate.
    joined = joined[~joined.index.duplicated(keep='first')]

    df['SchoolDistrictName'] = joined['DistrictName']
    df['SchoolDistrictType'] = joined['DistrictType']

    matched = df['SchoolDistrictName'].notna().sum()
    print(f'{name}: matched {matched:,} of {valid.sum():,} eligible rows to a school district '
          f'({matched / valid.sum():.1%})')
    return df


sold = add_school_district(sold, 'SOLD')
listings = add_school_district(listings, 'LISTINGS')

# ============================================================
# Step 3 - Segment analysis
# Summary statistics grouped by the key dimensions called out in the handbook, using the
# metrics engineered above.
#
# Note: avg_close_to_original_list_ratio comes out wildly inflated (100x+) for some
# segments below. That's not a bug -- a handful of records have a tiny/erroneous
# OriginalListPrice, which blows up the ratio and the mean along with it. This is exactly
# the kind of distortion Week 7's IQR outlier filtering is meant to catch; it's left
# unfiltered here since Week 6 is feature engineering, not outlier removal.
# ============================================================
def segment_summary(df, group_cols):
    return df.groupby(group_cols, dropna=False).agg(
        n_sales=('ClosePrice', 'count'),
        median_close_price=('ClosePrice', 'median'),
        avg_price_per_sqft=('price_per_sqft', 'mean'),
        avg_days_on_market=('DaysOnMarket', 'mean'),
        avg_close_to_original_list_ratio=('close_to_original_list_ratio', 'mean'),
    ).sort_values('n_sales', ascending=False)


by_property_subtype = segment_summary(sold, ['PropertyType', 'PropertySubType'])
by_county = segment_summary(sold, ['CountyOrParish', 'MLSAreaMajor'])
by_list_office = segment_summary(sold, ['ListOfficeName'])
by_buyer_office = segment_summary(sold, ['BuyerOfficeName'])

print('=== Segment summary: PropertyType / PropertySubType (top 10 by volume) ===')
print(by_property_subtype.head(10))
print('=== Segment summary: CountyOrParish / MLSAreaMajor (top 10 by volume) ===')
print(by_county.head(10))
print('=== Segment summary: ListOfficeName -- top 10 offices by sales volume (competitive intel) ===')
print(by_list_office.head(10))

by_property_subtype.to_csv('sold_segment_by_propertysubtype.csv')
by_county.to_csv('sold_segment_by_county.csv')
by_list_office.to_csv('sold_segment_by_listoffice.csv')
by_buyer_office.to_csv('sold_segment_by_buyeroffice.csv')

# ============================================================
# Save engineered datasets
# ============================================================
sold.to_csv('sold_engineered.csv', index=False)
listings.to_csv('listings_engineered.csv', index=False)

print('Saved sold_engineered.csv:', sold.shape)
print('Saved listings_engineered.csv:', listings.shape)

sample_cols = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea', 'DaysOnMarket',
    'price_ratio', 'close_to_original_list_ratio', 'price_per_sqft',
    'close_year', 'close_month', 'close_yrmo',
    'listing_to_contract_days', 'contract_to_close_days',
    'SchoolDistrictName', 'SchoolDistrictType'
]
print('=== Sample output (SOLD) ===')
print(sold[sample_cols].head(10))
