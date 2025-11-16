"""
Dynamic configuration loading utilities.

This module provides functions for dynamically loading configuration modules
based on environment variables.
"""
import importlib
import logging
import os
import sys

root_logger = logging.getLogger(__name__)

def load_config(env_var, base_module):
    """
    Dynamically load and execute a configuration module based on an environment variable.

    This function checks for the presence of an environment variable and, if found,
    imports the corresponding configuration module and executes its config() function.
    This allows for runtime configuration selection without code changes.

    Parameters
    ----------
    env_var : str
        Name of the environment variable to check (e.g., "APP_CONFIG").
    base_module : str
        Base module path to prepend to the config value
        (e.g., "myapp.configs"). The final module path will be
        "{base_module}.{config_value}".

    Returns
    -------
    None

    Notes
    -----
    - The environment variable should contain the config module name (not the full path).
    - The target module must have a `config()` function that will be called.
    - If the environment variable is not set, the function does nothing.

    Examples
    --------
    >>> # Set environment variable
    >>> os.environ["APP_CONFIG"] = "production"
    >>> # This will import "myapp.configs.production" and call its config() function
    >>> load_config("APP_CONFIG", "myapp.configs")

    >>> # Without environment variable set
    >>> load_config("MISSING_VAR", "myapp.configs")  # Does nothing
    """
    if env_var in os.environ:
        config = os.environ[env_var]
        config_module = f"{base_module}.{config}"
        root_logger.info(f"Loading config from {config_module}")
        importlib.import_module(config_module)
        # load the module
        module = sys.modules[config_module]
        module.config()
