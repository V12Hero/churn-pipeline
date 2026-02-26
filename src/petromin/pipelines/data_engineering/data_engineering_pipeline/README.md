# Pipeline data_engineering_pipeline

> This pipeline calls all the needed pipelines to perform the data engineering process for the OSM geospatial data.

## Overview

The pipeline is composed of the following pipelines:

- **raw_general_pipe**: tag dictionary process
- **users_pipe**: meshgrid generation process
- **osm_pipe**: osmosis process to reduce processing time
- **spatial_mastertable_pipe**: process to merge and generate features based in POI's, subway and highways amenities.

## Pipeline inputs
The main inputs are the **country-latest.osm** and the **tag_dictionary** of amenities.
## Pipeline outputs
This outputs the process and generates a datacatalog with the calculated table per country.
## Usage
```sh
$  pip  install  segai.data_engineering_pipeline
```
