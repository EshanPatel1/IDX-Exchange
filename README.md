# IDX Exchange Data Analyst Internship
**Author:** Eshan Patel

## Overview
IDX Exchange is a real estate technology company offering SaaS solutions for brokers, agents, and homeowners. The company's cloud-based platform helps real estate professionals generate and convert leads, deepen client relationships, and run smarter operations — powered by home valuation tools, predictive lead scoring, market intelligence dashboards, and fully customizable IDX websites, all through a flexible subscription model.

---

## Week 0 - MLS Data Pipeline Orientation
Before analytics begins, interns are introduced to how the MLS datasets used throughout the program are generated. Understanding the data's origins is essential context for everything that follows.
### Objective
Understand how monthly MLS datasets are produced from the CoreLogic Trestle API and exported as CSV files for analysis. Confirm all required monthly files are available before analytics begins.
### Outputs
* File inventory confirming which monthly CSVs are present
* List of any missing months flagged for re-download
### Skills Learned
* API authentication and token management
* Pagination handling for large dataset extraction
* Filtering listing and sold records by date ranges
* Exporting monthly datasets to structured CSV format

---

## Week 1 - Monthly Dataset Aggregation
Individual monthly files are combined into unified datasets that span multiple months, enabling trend analysis over time.
### Objective
Load and concatenate all monthly MLS files from January 2024 through the most recently completed calendar month into analysis-ready combined datasets.
### Outputs
* Combined sold transactions dataset
* Combined listing dataset
### Skills Learned
* Multi-file dataset management
* Data aggregation with Pandas
* Preparing time-series datasets for analysis

---

## Week 2&3 - Dataset Structuring and Validation
Before analytics begins, the dataset must be inspected and filtered to ensure only relevant residential property records are used. This week also covers foundational EDA steps that inform every phase that follows.
### Dataset Understanding
* Identify number of rows and columns
* Review column data types
* Identify high-missing columns
* Separate market analysis fields from metadata fields
### Objective
Fetch the FRED MORTGAGE30US series, resample it from weekly to monthly frequency, and merge it onto both combined datasets using a year-month key derived from transaction dates.
### Skills Learned
* Fetching live data from a public API (FRED)
* Resampling time-series data from weekly to monthly frequency
* Creating join keys from datetime fields
* Left merging external economic data onto a transaction dataset
* Validating merge completeness with null checks

---

## Week 4&5 - Data Cleaning and Preparation
Raw MLS data contains formatting inconsistencies, missing values, and fields that need transformation before analysis. This phase prepares the dataset for reliable analytics.
### Tasks
* Convert date fields to datetime format (CloseDate, PurchaseContractDate, ListingContractDate, ContractStatusChangeDate)
* Remove unnecessary or redundant columns
* Handle missing values appropriately
* Ensure numeric fields are properly typed
* Remove or flag invalid numeric values: ClosePrice <= 0, LivingArea <= 0, DaysOnMarket < 0, negative Bedrooms or Bathrooms
### Skills Learned
* Data cleaning and transformation
* Preparing analysis-ready datasets
* Working with datetime fields in Pandas

---

## Week 6 - Feature Engineering and Market Metrics
With clean data in hand, engineer the key market indicators that power the Tableau dashboards.
### Segment Analysis
After creating engineered metrics, group the analysis by key dimensions to uncover market patterns. Generate summary statistics for each segment:
* PropertyType and PropertySubType
* CountyOrParish and MLSAreaMajor
* ListOfficeName and BuyerOfficeName (for competitive intelligence)

---

## Week 7 - Outlier Detection and Data Quality
Extreme values in price, price per square footage, close to list ratio, or days on market can distort market averages and trends. Implement a statistical method to identify and remove these records.
### Method: Interquartile Range (IQR)
The IQR method removes records that fall outside a defined statistical range:

```
Q1 = df['ClosePrice'].quantile(0.25)
Q3 = df['ClosePrice'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['ClosePrice'] >= lower) & (df['ClosePrice'] <= upper)]
```
### Skills Learned
* Statistical data cleaning
* Outlier detection using IQR
* Improving dataset reliability for analytics
* Flagging vs. removing: building analysis-ready filtered datasets while preserving raw records

---

## Week 8-10 - Tableau Dashboard Development
With clean, engineered datasets, import the data into Tableau and build interactive dashboards that tell the story of the housing market.

---

## Week 11-12 - Final Presentation and Market Intelligence Report
The final weeks are dedicated to communicating findings. Produce a complete package: polished Tableau dashboards, a concise market intelligence report, and a live presentation.
### 1-Page Market Intelligence Report on any County, Zip Code, or City of Your Choice
* Market Overview – median home price trends, days on market
* Pricing Trends – price per square foot, sold-to-list ratio
* Market Activity – new listings vs. homes sold, transaction volume
* Competitive Landscape – top agents and brokerages by volume and units
* Key Takeaways – 3–5 data-driven insights discovered during analysis
