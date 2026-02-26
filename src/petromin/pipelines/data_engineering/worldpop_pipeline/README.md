# Pipeline segai_worldpop_de_pipe

> This pipeline uses the information of population density and in a modular way generates the column in 1km squared for the meshgrid of every country/city defined in the globals.

## Overview

The pipeline uses the meshgrid generator, spark processing, h3 and transformation functions to ultimately generate the population density for every meshgrid of each city/country.

## Pipeline inputs

**meshgrid:** meshgrid generated for each city-country

**worldpop:** 1km squared data of population density (open data)

## Pipeline outputs

table with lat/long and population density for each meshgrid.

## Usage

```sh
$  pip  install  segai.segai_worldpop_de_pipe
```
