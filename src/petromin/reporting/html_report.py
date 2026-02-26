"""Functionality for creating reports based on jupyter notebook templates."""
import logging
from pathlib import Path
from typing import Union

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
from traitlets.config import Config

from .html_report_utils import set_env_var

logger = logging.getLogger(__name__)


def _run_template(
    template_path: Union[str, Path],
    namespace: str = "",
    kernel: str = "python3",
    timeout: int = 600,
    env: str = "local",
) -> tuple[nbformat.notebooknode.NotebookNode, bool]:
    """Loads and runs an ipynb template.

    Args:
        template_path: path of template notebook
        kernel: ipython kernel. Use "python3" for currently active
                virtualenv
        namespace (str): modular pipeline model namespace.
        timeout: max run time in seconds
        env: kedro env
    Returns:
        nbconvert notebook object
    Raises:
        ValueError for wrong inputs
        CellExecutionError in case the execution fails
    """
    template_path = Path(template_path)
    if not template_path.is_file():
        raise ValueError(f"Template `{template_path}` is not a file.")

    with template_path.open("r") as file_:
        nb = nbformat.read(file_, as_version=4)

    target = f"{namespace}"
    nb["cells"] = [nbformat.v4.new_code_cell(f"namespace={target!r}")] + nb["cells"]

    epp = ExecutePreprocessor(kernel_name=kernel, timeout=timeout)

    with set_env_var("KEDRO_ENV", env):
        try:
            epp.preprocess(nb, {"metadata": {"path": str(template_path.parent)}})
        except CellExecutionError as cell_ex:
            logger.error(cell_ex)

            return nb, cell_ex

    return nb, False


def validate_notebook_error(error: CellExecutionError):
    """Raise CellExecutionError if exists."""
    if error:
        raise error


def create_html_report(
    params: dict,
    _wait_on: list[str] = None,
    *args,
):
    """Creates an html report from an ipynb template.

    Args:
        params: parameters for input models
        _wait_on: any catalog entry to wait on - not used in method
        *args (tp.Optional): add or not more arguments to the function

    Raises:
        ValueError for wrong inputs
    """
    template_path = params["template_path"]
    namespace = params.get("namespace", None)
    environment = params.get("environment", "base")

    logger.info(f"Running: {template_path}")

    processed, error = _run_template(
        template_path=template_path,
        namespace=namespace,
        kernel=params.get("kernel", "python3"),
        timeout=params.get("timeout", 600),
        env=environment,
    )

    c = Config()
    # we remove input and output prompts by default
    c.HTMLExporter.exclude_input_prompt = True
    c.HTMLExporter.exclude_output_prompt = True

    if params.get("remove_code", True):
        c.HTMLExporter.exclude_input = True

    html_exporter = HTMLExporter(c)
    body, _ = html_exporter.from_notebook_node(processed)

    return body, error
