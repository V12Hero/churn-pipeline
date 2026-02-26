# # (c) McKinsey & Company 2016 – Present
# # All rights reserved
# #
# #
# # This material is intended solely for your internal use and may not be reproduced,
# # disclosed or distributed without McKinsey & Company's express prior written consent.
# # Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# # or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# # update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# # information provided by Client as authorised herein will not violate any law
# # or contractual right of a third party. Client is responsible for the operation
# # and security of its operating environment. Client is responsible for performing final
# # testing (including security testing and assessment) of the code, model validation,
# # and final implementation of any model in a production environment. McKinsey is not
# # liable for modifications made to Deliverables by anyone other than McKinsey
# # personnel, (ii) for use of any Deliverables in a live production environment or
# # (iii) for use of the Deliverables by third parties; or
# # (iv) the use of the Deliverables for a purpose other than the intended use
# # case covered by the agreement with the Client.
# # Client warrants that it will not use the Deliverables in a "closed-loop" system,
# # including where no Client employee or agent is materially involved in implementing
# # the Deliverables and/or insights derived from the Deliverables.
# """Implementation of Jinja2TemplatedConfigLoader.

# - Template files must always be contained within a template folder
# due to anyconfig limitations.

# This works:

# anyconfig.load(path_to_yml, paths=[templates_folder])

# This doesn't

# anyconfig.load(path_to_yml, paths=[path_to_template_file])

# """

# import fnmatch
# import logging
# import os
# from collections.abc import Iterable
# from pathlib import Path
# from typing import Any

# from kedro.config import BadConfigException, MissingConfigException
# from kedro.config.common import _check_duplicate_keys, _lookup_config_filepaths
# from kedro.config.templated_config import TemplatedConfigLoader, _format_object
# from yaml.parser import ParserError

# from kedro.config.abstract_config import

# from segmentation_core.definitions import ROOT_DIR

# SUPPORTED_EXTENSIONS = [
#     ".yml",
#     ".yaml",
#     ".json",
#     ".ini",
#     ".pickle",
#     ".properties",
#     ".xml",
# ]
# TEMPLATES_DIR = ROOT_DIR + "/conf/base/catalog/templates"
# logger = logging.getLogger(__name__)


# def find_config_files() -> list[str]:
#     """Find all .yml files under current working directory."""
#     cwd = os.getcwd()
#     paths = []
#     for dirpath, _dirnames, filenames in os.walk(cwd):
#         folders = dirpath.split(os.sep)

#         # Check if "conf" is in any of the parents names
#         if any("conf" in folder for folder in folders):
#             current_folder_name = os.path.basename(dirpath)
#             for filename in fnmatch.filter(filenames, "*.yml"):
#                 filepath = os.path.join(dirpath, filename)
#                 logger.debug(f"{current_folder_name=}")
#                 logger.debug(f"{filepath=}")

#                 paths.append(filepath)

#     return paths


# def find_templates_folders() -> list[str]:
#     """Utility to find templates folder in current working directory.

#     A folder is included if it contains *.j2 files and has "templates" in its name.
#     """
#     cwd = os.getcwd()
#     templates_paths = []
#     for dirpath, _dirnames, filenames in os.walk(cwd):
#         os.path.basename(dirpath)
#         template_folder = False
#         for _filename in fnmatch.filter(filenames, "*.j2"):
#             template_folder = True
#             break

#         if "templates" in dirpath and template_folder:
#             templates_paths.append(dirpath)

#     return templates_paths


# def load_templated_files(namespaces_folder_path: str) -> list[dict]:
#     """Load templated files while specifying additional template folder."""
#     # for performance reasons
#     import anyconfig  # pylint: disable=import-outside-toplevel

#     config_paths = find_config_files()
#     templates_paths = find_templates_folders()

#     logger.debug(f"{templates_paths=}")

#     templates_paths = [*templates_paths, namespaces_folder_path]
#     configs = []
#     # Load files individually to isolate errors
#     for path in config_paths:
#         logger.debug(f"Config path: {path=}")
#         logger.debug(f"Templates {templates_paths=}")

#         try:
#             config = anyconfig.load(
#                 [path],
#                 paths=templates_paths,
#                 ac_template=True,
#             )
#             configs.append(config)
#         except Exception as e:
#             logger.error(f"Error loading {path=}")
#             raise Exception(e)

#     return configs


# class Jinja2TemplatedConfigLoader(TemplatedConfigLoader):
#     """Templated config loader with template folder enabled for catalog.

#     We can improve it to include templates in other subfolders of catalog with automatic discovery
#     of such folders.
#     """

#     def __init__(self, *args, **kwargs):
#         """Templated config loader with template folder enabled for catalog."""
#         super().__init__(*args, **kwargs)
#         msg = f"Using templated config loader, loading templates from {TEMPLATES_DIR}"
#         logger.info(msg)
#         if self.runtime_params is not None:
#             self._config_mapping.update(self.runtime_params)

#     def get(self, *patterns: str) -> dict:  # type: ignore
#         """Get configuration from patterns.

#         Tries to resolve the template variables in the config dictionary
#         provided by the ``ConfigLoader`` (super class) ``get`` method using the
#         dictionary of replacement values obtained in the ``__init__`` method.

#         Args:
#             *patterns: Glob patterns to match. Files, which names match
#                 any of the specified patterns, will be processed.

#         Returns:
#             A Python dictionary with the combined configuration from all
#             configuration files. **Note:** any keys that start with `_`
#             will be ignored. String values wrapped in `${...}` will be
#             replaced with the result of the corresponding JMESpath
#             expression evaluated against globals.

#         Raises:
#             ValueError: malformed config found.
#         """
#         config_raw = _get_config_from_patterns(
#             conf_paths=self.conf_paths, patterns=patterns, ac_template=True
#         )
#         return _format_object(config_raw, self._config_mapping)


# def _get_config_from_patterns(
#     conf_paths: Iterable[str],
#     patterns: Iterable[str] = None,
#     ac_template: bool = False,
#     ac_context: dict[str, Any] = None,
# ) -> dict[str, Any]:
#     """Recursively scan for configuration files.

#     Load and merge them, and return them in the form of a config dictionary.

#     Args:
#         conf_paths: List of configuration paths to directories
#         patterns: Glob patterns to match. Files, which names match
#             any of the specified patterns, will be processed.
#         ac_template: Boolean flag to indicate whether to use the `ac_template`
#             argument of the ``anyconfig.load`` method. Used in the context of
#             `_load_config_file` function.
#         ac_context: anyconfig context to pass to ``anyconfig.load`` method.
#             Used in the context of `_load_config_file` function.

#     Raises:
#         ValueError: If 2 or more configuration files inside the same
#             config path (or its subdirectories) contain the same
#             top-level key.
#         MissingConfigException: If no configuration files exist within
#             a specified config path.
#         BadConfigException: If configuration is poorly formatted and
#             cannot be loaded.

#     Returns:
#         Dict[str, Any]:  A Python dictionary with the combined
#             configuration from all configuration files. **Note:** any keys
#             that start with `_` will be ignored.
#     """
#     if not patterns:
#         raise ValueError(
#             "'patterns' must contain at least one glob "
#             "pattern to match config filenames against."
#         )

#     config = {}  # type: dict
#     processed_files = set()  # type: set

#     for conf_path in conf_paths:
#         if not Path(conf_path).is_dir():
#             raise ValueError(
#                 f"Given configuration path either does not exist "
#                 f"or is not a valid directory: {conf_path}"
#             )

#         config_filepaths = _lookup_config_filepaths(
#             Path(conf_path), patterns, processed_files, logger
#         )
#         new_conf = _load_configs(
#             config_filepaths=config_filepaths,
#             ac_template=ac_template,
#             ac_context=ac_context,
#         )

#         common_keys = config.keys() & new_conf.keys()
#         if common_keys:
#             sorted_keys = ", ".join(sorted(common_keys))
#             msg = (
#                 "Config from path '%s' will override the following "
#                 "existing top-level config keys: %s"
#             )
#             logger.info(msg, conf_path, sorted_keys)

#         config.update(new_conf)
#         processed_files |= set(config_filepaths)

#     if not processed_files:
#         raise MissingConfigException(
#             f"No files found in {conf_paths} matching the glob "
#             f"pattern(s): {patterns}"
#         )
#     return config


# def _load_config_file_templated(
#     config_file: Path, ac_template: bool = False, ac_context: dict[str, Any] = None
# ) -> dict[str, Any]:
#     """Load an individual config file using `anyconfig` as a backend.

#     Args:
#         config_file: Path to a config file to process.
#         ac_template: Boolean flag to indicate whether to use the `ac_template`
#             argument of the ``anyconfig.load`` method.
#         ac_context: anyconfig context to pass to ``anyconfig.load`` method.

#     Raises:
#         BadConfigException: If configuration is poorly formatted and
#             cannot be loaded.
#         ParserError: If file is invalid and cannot be parsed.

#     Returns:
#         Parsed configuration.
#     """
#     # for performance reasons
#     import anyconfig  # pylint: disable=import-outside-toplevel

#     try:
#         # Default to UTF-8, which is Python 3 default encoding, to decode the file
#         with open(config_file, encoding="utf8") as yml:
#             logger.debug("Loading config file: '%s'", config_file)

#             return {
#                 k: v
#                 for k, v in anyconfig.load(
#                     yml,
#                     ac_template=ac_template,
#                     ac_context=ac_context,
#                     paths=[TEMPLATES_DIR],
#                 ).items()
#                 if not k.startswith("_")
#             }
#     except AttributeError as exc:
#         raise BadConfigException(f"Couldn't load config file: {config_file}") from exc

#     except ParserError as exc:
#         assert exc.problem_mark is not None
#         line = exc.problem_mark.line
#         cursor = exc.problem_mark.column
#         raise ParserError(
#             f"Invalid YAML file {config_file}, unable to read line {line}, position {cursor}."
#         ) from exc


# def _load_configs(
#     config_filepaths: list[Path], ac_template: bool, ac_context: dict[str, Any] = None
# ) -> dict[str, Any]:
#     """Recursively load all configuration files.

#     Which satisfy a given list of glob patterns from a specific path.

#     Args:
#         config_filepaths: Configuration files sorted in the order of precedence.
#         ac_template: Boolean flag to indicate whether to use the `ac_template`
#             argument of the ``anyconfig.load`` method. Used in the context of
#             `_load_config_file` function.
#         ac_context: anyconfig context to pass to ``anyconfig.load`` method.
#             Used in the context of `_load_config_file` function.

#     Raises:
#         ValueError: If 2 or more configuration files contain the same key(s).
#         BadConfigException: If configuration is poorly formatted and
#             cannot be loaded.

#     Returns:
#         Resulting configuration dictionary.

#     """
#     aggregate_config = {}
#     seen_file_to_keys = {}

#     for config_filepath in config_filepaths:
#         single_config = _load_config_file_templated(
#             config_filepath, ac_template=ac_template, ac_context=ac_context
#         )
#         _check_duplicate_keys(seen_file_to_keys, config_filepath, single_config)
#         seen_file_to_keys[config_filepath] = single_config.keys()
#         aggregate_config.update(single_config)

#     return aggregate_config
