"""Chance level computation utilities.

Implements theoretical and adjusted chance levels based on:
Combrisson & Jerbi (2015), "Exceeding chance level by chance:
The caveat of theoretical chance levels in brain signal classification
and statistical assessment of decoding accuracy."
Journal of Neuroscience Methods, 250, 126-136.
"""

from __future__ import annotations

import logging
from typing import Any

from scipy.stats import binom


log = logging.getLogger(__name__)


def adjusted_chance_level(n_classes: int, n_trials: int, alpha: float = 0.05) -> float:
    """Return the adjusted chance level using a binomial significance threshold.

    Computes the minimum accuracy that significantly exceeds chance at
    significance level ``alpha``, given the number of classes and trials.
    Based on the inverse survival function of the binomial distribution
    as recommended by Combrisson & Jerbi (2015).

    Parameters
    ----------
    n_classes : int
        Number of classes in the classification task.
    n_trials : int
        Number of trials (test samples) used for classification.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    float
        Adjusted chance level (proportion of correct classifications
        needed to significantly exceed chance).

    Examples
    --------
    >>> adjusted_chance_level(2, 100, 0.05)  # doctest: +SKIP
    0.58
    """
    # theoretical chance level: 1 / n_classes
    p_chance = 1.0 / n_classes
    return binom.isf(alpha, n_trials, p_chance) / n_trials


def chance_levels_from_dataframe(
    data,
    alpha: float | list[float] = 0.05,
) -> dict[str, dict[str, Any]]:
    """Compute chance levels directly from a results DataFrame.

    Uses ``n_samples_test`` and ``n_classes`` columns stored by the
    evaluation to compute theoretical and adjusted chance levels without
    needing dataset objects.

    Parameters
    ----------
    data : DataFrame
        Results dataframe with ``dataset``, ``n_samples_test``, and
        ``n_classes`` columns.
    alpha : float or list of float, default=0.05
        Significance level(s) for adjusted chance levels.

    Returns
    -------
    dict
        Mapping of ``{dataset_name: {'theoretical': float,
        'adjusted': {alpha: float, ...}}}``.
    """
    if isinstance(alpha, (int, float)):
        alpha = [alpha]

    result = {}
    for dname, grp in data.groupby("dataset"):
        n_classes = int(grp["n_classes"].iloc[0])
        n_trials = int(grp["n_samples_test"].iloc[0])
        # theoretical chance level: 1 / n_classes
        entry: dict[str, Any] = {
            "theoretical": 1.0 / n_classes,
            "adjusted": {a: adjusted_chance_level(n_classes, n_trials, a) for a in alpha},
        }
        result[dname] = entry

    return result
