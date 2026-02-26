"""notebook utils."""
import os
from contextlib import contextmanager


@contextmanager
def set_env_var(key: str, value: str):
    """Simple context manager to temporarily set an env var."""
    current = os.environ.get(key)
    try:
        os.environ[key] = value
        yield
    finally:
        if current is None:
            del os.environ[key]
        else:
            os.environ[key] = current
