# Part 2: Pipeline Setup Documentation

This document outlines the process used to create a data pipeline to load store sales data from Google Cloud Storage (GCS) into BigQuery.

## Method Used

A custom Apache Beam pipeline written in Python was used instead of the standard "Text Files to BigQuery" template. This approach was chosen to programmatically handle the header row in the source CSV files, which is a more robust method. The pipeline was executed using the Dataflow runner from Google Cloud Shell.

## Pipeline for `store_info` Table

1.  A Python script named `load_stores_pipeline.py` was created.
2.  The script was configured to read `stores.csv` from GCS.
3.  It used the `skip_header_lines=1` option to ignore the CSV header during ingestion.
4.  It was configured with `create_disposition=CREATE_IF_NEEDED` to automatically create the `store_info` table in the `store_sales_team_dn4` dataset.
5.  The script was launched using the command `python load_stores_pipeline.py` in Cloud Shell.

## Pipeline for `sales_data` Table

1.  A similar process was repeated for the `train.csv` file using a second script named `load_sales_pipeline.py`.
2.  This script was configured to read the large `train.csv` file and load it into the `sales_data` table with the correct schema.
3.  The script was also launched from Cloud Shell.

## Data Verification

After both Dataflow jobs completed successfully, a consolidated SQL query was run in BigQuery to verify both tables at once. The results confirmed the data was loaded correctly.
store_info table: Loaded exactly 54 rows.
sales_data table: Loaded exactly 3,000,888 rows.
Date Range: The sales data correctly spans from January 1, 2013, to August 15, 2017.