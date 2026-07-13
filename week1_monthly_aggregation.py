#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import glob
import os

# Setup Directory
os.chdir(r'C:\Users\PC\Downloads\idx-exchange-internship\data\raw')

# Load and combine all Sold files
sold_files = sorted(glob.glob('CRMLSSold*.csv'))
sold = pd.concat([pd.read_csv(f, low_memory=False) for f in sold_files], ignore_index=True)
print('Sold rows before filter:', len(sold))
sold = sold[sold['PropertyType'] == 'Residential']
print('Sold rows after Residential filter:', len(sold))
sold.to_csv('sold_combined_residential.csv', index=False)

# Load and combine all Listing files
listing_files = sorted(glob.glob('CRMLSListing*.csv'))
listings = pd.concat([pd.read_csv(f, low_memory=False) for f in listing_files], ignore_index=True)
print('Listings rows before filter:', len(listings))
listings = listings[listings['PropertyType'] == 'Residential']
print('Listings rows after Residential filter:', len(listings))
listings.to_csv('listings_combined_residential.csv', index=False)
