"""Raw layer nodes."""
from typing import Dict

import pyspark


def validate_schema(parameters: Dict, **dfs: Dict[str, pyspark.sql.DataFrame]) -> bool:
    """Validates the schema.

    Args:
        parameters (dict): A dictionary containing schema information for the dataframes.
        **dfs (Dict[str, pyspark.sql.DataFrame]): Keyword arguments where the key is the dataframe name
            and the value is the corresponding pyspark.sql.DataFrame.

    Returns:
        bool: True if the schema is valid for all dataframes, otherwise raises a RuntimeError.

    Raises:
        RuntimeError: If there is a mismatch between the desired schema and the input schema.

    Examples:
        Usage examples of the validate_schema function:

        1. Validating a single dataframe:
        ```python
        parameters = {
            'df1': {
                'fields': {
                    'column1': 'string',
                    'column2': 'integer',
                }
            }
        }

        df1 = spark.createDataFrame([(1, 'foo'), (2, 'bar')], ['column2', 'column1'])

        try:
            result = validate_schema(parameters, df1=df1)
            print("Schema is valid.")
        except RuntimeError as e:
            print(f"Schema validation failed: {str(e)}")
        ```

        2. Validating multiple dataframes:
        ```python
        parameters = {
            'df1': {
                'fields': {
                    'column1': 'string',
                    'column2': 'integer',
                }
            },
            'df2': {
                'fields': {
                    'columnA': 'string',
                    'columnB': 'double',
                }
            }
        }

        df1 = spark.createDataFrame([(1, 'foo'), (2, 'bar')], ['column2', 'column1'])
        df2 = spark.createDataFrame([('x', 1.0), ('y', 2.5)], ['columnA', 'columnB'])

        try:
            result = validate_schema(parameters, df1=df1, df2=df2)
            print("Schema is valid for all dataframes.")
        except RuntimeError as e:
            print(f"Schema validation failed: {str(e)}")
        ```
    """
    # df_name is the name of each element dataframe passed as argument and df_spark is the dataframe itself
    for df_name, df_spark in dfs.items():
        # desired_schema is a dictionary with desired schema passed as argument. Contains the column names and data types
        desired_schema = parameters[df_name]["fields"]

        # inpus_schema is the schema (column names and types) of each of the tables passed as argument.
        input_schema = {col.name: col.dataType for col in list(df_spark.schema)}
        if desired_schema == "all":
            continue
        # columns_desired_schema and columns_desired_schema are sets containing the column names of the input and desired schema.
        columns_desired_schema = set(desired_schema.keys())
        columns_input_schema = set(input_schema.keys())

        # If there is any different or extra element in either of the two sets, an error is raised.
        if (len(columns_desired_schema - columns_input_schema) > 0) or (
            len(columns_input_schema - columns_desired_schema) > 0
        ):
            raise RuntimeError(
                f"""Invalid schema in the {df_name} dataframe
            Please make sure your data has the correct schema.
            Check the following:
                - The number of columns in your data may have changed.
                - One of the column names in your data may have changed.

            Difference between desired schema and input schema: {columns_desired_schema-columns_input_schema}
            Difference between input schema and desired schema: {columns_input_schema-columns_desired_schema}"""
            )
    return True
