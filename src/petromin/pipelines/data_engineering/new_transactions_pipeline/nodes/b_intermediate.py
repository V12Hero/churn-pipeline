"""Intermediate layer."""
import datetime as dt
import typing as tp

import pyspark
import pyspark.sql.functions as f


def reformat_columns(
    parameters: tp.Dict,
    scema_validation_boolean: bool,
    **dfs: tp.Dict[str, pyspark.sql.DataFrame],
) -> tp.Dict[str, pyspark.sql.DataFrame]:
    """Reformats the columns of dataframes according to specified schema.

    Args:
        parameters (dict): A dictionary containing schema information for the dataframes.
        schema_validation_boolean (bool): A boolean indicating whether schema validation should be performed.
        **dfs (Dict[str, pyspark.sql.DataFrame]): Keyword arguments where the key is the dataframe name
            and the value is the corresponding pyspark.sql.DataFrame.

    Returns:
        Dict[str, pyspark.sql.DataFrame]: A dictionary with dataframe names as keys and reformatted dataframes as values.

    Raises:
        Exception: If schema validation is disabled.

    Examples:
        Usage examples of the reformat_columns function:

        1. Reformatting columns with schema validation:
        ```python
        parameters = {
            'df1': {
                'fields': {
                    'column1': {'name': 'Name', 'data_type': 'STRING'},
                    'column2': {'name': 'Age', 'data_type': 'INTEGER'},
                    'column3': {'name': 'DOB', 'data_type': 'DATE'},
                }
            }
        }

        df1 = spark.createDataFrame([(1, 'foo', '1990-01-01'), (2, 'bar', '1985-05-15')], ['column2', 'column1', 'column3'])

        try:
            result_dfs = reformat_columns(parameters, True, df1=df1)
            print("Columns reformatted successfully.")
        except Exception as e:
            print(f"Column reformatting failed: {str(e)}")
        ```

        2. Reformatting columns without schema validation:
        ```python
        parameters = {
            'df1': {
                'fields': {
                    'column1': {'name': 'Name', 'data_type': 'STRING'},
                    'column2': {'name': 'Age', 'data_type': 'INTEGER'},
                    'column3': {'name': 'DOB', 'data_type': 'DATE'},
                }
            }
        }

        df1 = spark.createDataFrame([(1, 'foo', '1990-01-01'), (2, 'bar', '1985-05-15')], ['column2', 'column1', 'column3'])

        try:
            result_dfs = reformat_columns(parameters, False, df1=df1)
            print("Columns reformatted successfully without schema validation.")
        except Exception as e:
            print(f"Column reformatting failed: {str(e)}")
        ```
    """
    if scema_validation_boolean:
        pass
    else:
        raise Exception
    # df_name is the name of each element dataframe passed as argument and df_spark is the dataframe itself
    for df_name, df_spark in dfs.items():
        # desired_schema is a dictionary with desired schema passed as argument. Contains the column names and data types
        desired_schema = parameters[df_name]["fields"]

        if desired_schema == "all":
            dfs[df_name] = df_spark
            continue

        for field, metadata in desired_schema.items():
            # If the intended data type is string, the column is casted to string
            # , the text is converted to upper case, a trim is applied and
            # the new line character is removed.
            if metadata["data_type"] == "STRING":
                df_spark = df_spark.withColumn(field, f.col(field).cast(metadata["data_type"]))
                df_spark = df_spark.withColumn(field, f.trim(f.upper(f.col(field))))
                df_spark = df_spark.withColumn(field, f.regexp_replace(f.col(field), "\\\n", ". "))

            # If the intended data type is date, the column is casted to date, and all nulls and dates before
            # 01-01-1900 are changed to 01-01-1900
            elif metadata["data_type"] == "DATE":

                if dict(df_spark.dtypes)[field] == 'bigint':
                    df_spark = df_spark.withColumn(
                        "seconds",
                        (f.col(field)/1000000000)
                    ).withColumn(
                        field,
                        f.to_date(f.from_unixtime(f.col('seconds')))
                    ).drop("seconds")
                else:
                    # If the column is a string combining year, month and date, it is converted to date.
                    df_spark = df_spark.withColumn(field, f.col(field).cast("STRING"))
                    df_spark = df_spark.withColumn(
                        field, f.to_date(f.date_format(f.col(field), "yyyy-MM-dd"))
                    )

                df_spark = df_spark.withColumn(
                    field,
                    f.when(
                        (f.col(field).isNull()) | (f.year(field) < 1900),
                        dt.date(1900, 1, 1),
                    ).otherwise(f.col(field)),
                )
            elif metadata["data_type"] == "TIMESTAMP":

                if dict(df_spark.dtypes)[field] == 'bigint':
                    df_spark = df_spark.withColumn(
                        "seconds",
                        (f.col(field)/1000000000)
                    ).withColumn(
                        field,
                        f.to_timestamp(f.from_unixtime(f.col('seconds')))
                    ).drop("seconds")
                else:
                    # If the column is a string combining year, month and date, it is converted to date.
                    df_spark = df_spark.withColumn(field, f.col(field).cast("STRING"))
                    df_spark = df_spark.withColumn(
                        field, f.to_timestamp(f.date_format(f.col(field), "yyyy-MM-dd"))
                    )

                df_spark = df_spark.withColumn(
                    field,
                    f.when(
                        (f.col(field).isNull()) | (f.year(field) < 1900),
                        dt.date(1900, 1, 1),
                    ).otherwise(f.col(field)),
                )
            # For the rest of intended data types, the column is only casted.
            else:
                df_spark = df_spark.withColumn(field, f.col(field).cast(metadata["data_type"]))

            # The column names are replaced to they do not contain spaces
            df_spark = df_spark.withColumnRenamed(field, metadata["name"])

        dfs[df_name] = df_spark
    return dfs
