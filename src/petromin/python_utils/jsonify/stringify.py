import typing as tp


def stringify_dict(params: tp.Dict[str, tp.Any]):
    """
    Recursively converts all values inside keys to strings.

    Args:
        data (dict): The input dictionary to be transformed.

    Returns:
        dict: A new dictionary with all values as strings.
    """
    result = {}
    for key, value in params.items():
        result[key] = str(value)
    return result
