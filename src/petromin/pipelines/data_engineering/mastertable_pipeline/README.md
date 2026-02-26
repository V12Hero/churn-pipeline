# Pipeline mastertable_pipeline

> This pipeline generates the mastertable combining all the other pipelines.

## Overview
The pipeline merges all the other pipelines outputs into a mastertable that then is used as input for segmentation purposes.

## Pipeline inputs
**geospatial:** This is the geospatial output per country

**census:** This is the census output per country

**worldpop:** This is the worldpop output per country

**transactions:** This is the transactions output (is used keep in mind to perform the merge with fewer columns)
## Pipeline outputs
a mastertable containing all the provided inputs on a geometry and lat/long id basis.
## Usage
```sh
$  pip  install  segai.mastertable_pipeline
```
