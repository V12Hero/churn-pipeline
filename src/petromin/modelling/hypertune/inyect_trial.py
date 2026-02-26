import typing as tp

import optuna


def inject_trial_parameter(d: tp.Dict[str, tp.Any], trial: optuna.Trial):
    """Inject optuna trial parameter for a objective function.

    It takes a dictionary and a trial object, and replaces any string that starts
    with "trial." with the corresponding value from the trial object.

    Args:
      d (dict): dict
      trial (optuna.Trial): optuna.Trial

    Returns:
      A dictionary with the values of the dictionary d replaced with the values of the
      optuna trial.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            inject_trial_parameter(v, trial)
        elif isinstance(v, str) and "trial." in v:
            d[k] = eval(v, {"trial": trial})
    return d
