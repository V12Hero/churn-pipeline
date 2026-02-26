# Pipeline transactions_pipeline

> This pipeline is a mock pipeline with dummy data that needs to be replaced and modified with client data and their own necessities.

## Overview
The pipeline uses the general pipe to convert certain columns using the tag_dictionary, then it cleans the primary input and generates features based on different levels like client, or client-segment or client-microsegment. It also calculates temporal features based on 90 days or 120 days, specified by the user.

## Pipeline inputs
**tag_dictionary:**  tag dictionary with information for the transactions data columns.

**transactions_dummy_data:** mocked data to be used as dummy.

## Pipeline outputs
table on customer_id with all the features.
## Usage
```sh
$  pip  install  segai.transactions_pipeline
```
