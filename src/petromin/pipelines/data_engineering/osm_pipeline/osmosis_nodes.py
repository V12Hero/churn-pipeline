# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.

"""This code generates bounding boxes from an OpenStreetMap PBF file, which can
be a large file. By splitting the file into smaller bounding boxes, we can
process each one faster, allowing us to process them dramatically faster.

The run_osmosis() function takes four coordinates representing the
boundaries of the bounding box, along with an output file name, and uses
Osmosis to generate the bounding box file.

The run_osmosis_from_dict() function takes a dictionary of locations and
their corresponding latitude and longitude ranges, along with the input
PBF file name, and generates bounding box files for each location using
run_osmosis().
"""
import os
import shutil
import subprocess
from typing import Dict, List, Tuple

from segmentation_core.definitions import ROOT_DIR  # noqa


def run_osmosis(
    lat: List[float], lon: List[float], input_path: str, output_path: str
) -> None:
    """Run the osmosis command with the given latitude and longitude
    coordinates, using the input file at input_path and saving the output to
    output_path.

    Args:
        lat (List[float]): A list of two latitude values (bottom and top).
        lon (List[float]): A list of two longitude values (left and right).
        input_path (str): The path to the input PBF file.
        output_path (str): The path to the output OSM file.

    Returns:
        None
    """
    # Extract the latitude and longitude values
    left, right = lon
    bottom, top = lat

    # Build the osmosis command
    cmd = f"osmosis --read-pbf file='{input_path}' --bb left={left} right={right} bottom={bottom} top={top} --write-pbf '{output_path}'"

    print(cmd)
    # Run the osmosis command
    subprocess.run(cmd, shell=True)


def run_osmosis_from_dict(
    data: Dict[str, Dict[str, List[float]]], input_path: str, output_folder: str
) -> List[str]:
    """Run the osmosis command for multiple cities using a dictionary of city
    names and their corresponding latitude and longitude coordinates. The
    output files will be saved to the output_folder using the city name as the
    filename.

    Args:
        data (Dict[str, Dict[str, List[float]]]): A dictionary of city names and their corresponding latitude and longitude coordinates.
        input_path (str): The path to the input PBF file.
        output_folder (str): The path to the output folder.

    Returns:
        List[str]
    """
    output_paths = []
    # Iterate over the dictionary of cities and coordinates
    for city, coords in data.items():
        lat = coords["lat"]
        lon = coords["lon"]

        # Build the output file path for the city
        output_path = f"{output_folder}/{city}.osm.pbf"

        # Call the run_osmosis function to process the city's coordinates
        run_osmosis(lat, lon, input_path, output_path)

        output_paths.append(output_path)

    return output_paths


def merge_osm_files(filepaths, output_path) -> str:
    """Merge a list of OSM files using osmosis and save the output to a single
    file.

    Args:
        filepaths (list): A list of filepaths to the OSM files to be merged.
        output_path (str): The path and filename of the output merged OSM file.

    Returns:
        str: The path and filename of the merged OSM file.
    """
    # Build the osmosis command to merge the files
    if len(filepaths) == 1:
        assert os.path.isfile(filepaths[0])
        shutil.move(filepaths[0], output_path)
    else:
        cmd = f"osmosis --read-pbf '{filepaths[0]}'"
        for filepath in filepaths[1:]:
            cmd += f" --read-pbf '{filepath}' --merge"
        cmd += f" --write-pbf '{output_path}'"

        # Run the osmosis command to merge the files
        print(cmd)
        subprocess.run(cmd, shell=True)

    return output_path


def osmosis_preprocessing(bbox: Dict, filepath: str, country_name: str) -> Tuple[bool]:
    """> It takes a bounding box, a filepath, and a country name, and returns a
    boolean indicating whether the file was successfully created.

    Args:
      bbox (Dict): A dictionary with the following keys:
      filepath (str): The name of the file you want to download.
      country_name (str): The name of the country you're working with.

    Returns:
      True if file was written.
    """
    input_path = ROOT_DIR + f"/data/01_raw/{country_name}/osm/{filepath}.osm.pbf"
    output_folder = ROOT_DIR + f"/data/01_raw/{country_name}/osm"

    filepaths = run_osmosis_from_dict(
        bbox, input_path=input_path, output_folder=output_folder
    )

    merged_file = f"{output_folder}/merged.osm.pbf"

    merge_osm_files(filepaths, merged_file)

    return (os.path.isfile(merged_file),)
