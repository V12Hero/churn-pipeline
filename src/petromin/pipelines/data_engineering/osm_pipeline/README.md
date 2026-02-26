# Pipeline osm_pipeline

> This pipeline generates the process for osmosis and reduction of data based on the bbox per country/city

## Overview
The pipeline uses the following functions:

**run_osmosis:** Run the osmosis command with the given latitude and longitude coordinates, using the input file at input_path and saving the output to output_path.

**run_osmosis_from_dict:** Run the osmosis command for multiple cities using a dictionary of city names and their corresponding latitude and longitude coordinates. The output files will be saved to the output_folder using the city name as the filename.

**merge_osm_files:** Merge a list of OSM files using osmosis and save the output to a single file.

**osmosis_preprocessing:** It takes a bounding box, a filepath, and a country name, and returns a boolean indicating whether the file was successfully created.

## Pipeline inputs
tag_dictionary and bbox parameters of globals.
## Pipeline outputs
merged result of only the parameters city bbox per country information of OSM.
## Usage
```sh
$  pip  install  segai.osm_pipeline
```
