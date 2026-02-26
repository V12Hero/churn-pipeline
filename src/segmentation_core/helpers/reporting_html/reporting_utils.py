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

"""
notebook utils
"""
import os
from contextlib import contextmanager
from pathlib import Path

from IPython.display import Markdown, display
from kedro.framework.context import KedroContext
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import _get_project_metadata


@contextmanager
def set_env_var(key, value):
    """
    Simple context manager to temporarily set an env var.
    """
    current = os.environ.get(key)
    try:
        os.environ[key] = value
        yield
    finally:
        if current is None:
            del os.environ[key]
        else:
            os.environ[key] = current


def load_context(start_path=None, max_depth=4, env=None, **kwargs) -> KedroContext:
    """
    Tries to load the kedro context from a notebook of unknown location.
     Assumes that the notebook is placed somewhere in the kedro project.

    Args:
        start_path: starting point for the search. We try to load the
         context from here and continue up the chain of parents
        max_depth: max number of parents to visit
        env: kedro environment to use. Defaults to `KEDRO_ENV`
         environment variable if available
        **kwargs: kwargs for `kedro.context.load_context`

    """
    start_path = Path(start_path) if start_path is not None else Path.cwd()
    env = env or os.environ.get("KEDRO_ENV")

    project_path = Path.cwd().resolve().parents[3]  # Assumes dir location for utils.py
    package_name = _get_project_metadata(project_path).package_name
    configure_project(package_name)

    session_kwargs = dict(
        package_name=package_name,
        project_path=project_path,
        env=env,
        save_on_close=False,
    )
    session = KedroSession.create(**session_kwargs)
    context = session.load_context()
    return context


def mprint(text: str, **kwargs):
    """
    Renders Markdown text in a notebook.
    Args:
        text: raw Markdown
        **kwargs: arguments for Ipython.display.display
    """
    display(Markdown(text), **kwargs)
