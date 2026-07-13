#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import glob
import os
from datetime import datetime

# Setup Directory
os.chdir(r'C:\Users\PC\Downloads\idx-exchange-internship\data\raw')

# CRMLSListing Files
listing_files = sorted(glob.glob('CRMLSListing*.csv'))
print(f'Found {len(listing_files)} Listing files.\n')

for f in listing_files:
    df = pd.read_csv(f, low_memory=False)
    print(f'{f}: {len(df):,} rows')

# CRMLSSold Files
sold_files = sorted(glob.glob('CRMLSSold*.csv'))
print(f'Found {len(sold_files)} Sold files.\n')

for f in sold_files:
    df = pd.read_csv(f, low_memory=False)
    print(f'{f}: {len(df):,} rows')

# Missing Month Check
today = datetime.today()
end_year  = today.year  if today.month > 1 else today.year - 1
end_month = today.month - 1 if today.month > 1 else 12

expected_months = []
y, m = 2024, 1
while (y, m) <= (end_year, end_month):
    expected_months.append(f'{y}{m:02d}')
    m += 1
    if m > 12:
        m, y = 1, y + 1

listing_months = set(f.replace('CRMLSListing', '').replace('.csv', '') for f in listing_files)
sold_months    = set(f.replace('CRMLSSold', '').replace('.csv', '') for f in sold_files)

missing_listings = [ym for ym in expected_months if ym not in listing_months]
missing_sold     = [ym for ym in expected_months if ym not in sold_months]

if missing_listings:
    for ym in missing_listings:
        print(f'MISSING: CRMLSListing{ym}.csv')
else:
    print('All Listing months present.')

if missing_sold:
    for ym in missing_sold:
        print(f'MISSING: CRMLSSold{ym}.csv')
else:
    print('All Sold months present.')
