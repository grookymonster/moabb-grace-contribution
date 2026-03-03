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

from moabb.analysis._utils import _compute_n_trials, _match_int


log = logging.getLogger(__name__)


def theoretical_chance_level(n_classes: int) -> float:
    """Return the theoretical chance level for a classification task.

    Parameters
    ----------
    n_classes : int
        Number of classes in the classification task.

    Returns
    -------
    float
        Theoretical chance level (1 / n_classes).

    Examples
    --------
    >>> theoretical_chance_level(2)
    0.5
    >>> theoretical_chance_level(4)
    0.25
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    return 1.0 / n_classes


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
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    p_chance = theoretical_chance_level(n_classes)
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
        entry: dict[str, Any] = {
            "theoretical": theoretical_chance_level(n_classes),
            "adjusted": {a: adjusted_chance_level(n_classes, n_trials, a) for a in alpha},
        }
        result[dname] = entry

    return result


def get_chance_levels(
    datasets,
    alpha: float | list[float] | None = None,
    paradigm=None,
) -> dict[str, dict[str, Any]]:
    """Extract chance levels from a list of MOABB dataset objects.

    Uses ``len(dataset.event_id)`` for the number of classes by default.
    If a ``paradigm`` is provided, uses ``len(paradigm.used_events(dataset))``
    instead, which correctly reflects filtered paradigms such as
    ``LeftRightImagery`` on multi-class datasets.
    Uses ``dataset._summary_table`` for trial counts when computing adjusted
    levels.

    Parameters
    ----------
    datasets : list of dataset objects
        MOABB dataset objects. Each must have an ``event_id`` attribute.
    alpha : float or list of float or None, default=None
        Significance level(s) for adjusted chance levels. If None,
        only theoretical levels are returned. If a float, a single
        adjusted level is computed. If a list, adjusted levels are
        computed for each alpha value.
    paradigm : object or None, default=None
        Optional paradigm instance used for evaluation. If provided, the
        number of classes is inferred with ``paradigm.used_events(dataset)``
        instead of all classes present in ``dataset.event_id``.

    Returns
    -------
    dict
        Mapping of ``{dataset_name: {'theoretical': float, 'adjusted': {alpha: float, ...}}}``.
        The ``'adjusted'`` key is only present when ``alpha`` is not None
        and trial information is available.

    Examples
    --------
    >>> from moabb.datasets.fake import FakeDataset
    >>> levels = get_chance_levels([FakeDataset()])
    >>> levels['FakeDataset']['theoretical']  # doctest: +SKIP
    0.333...
    """
    if alpha is not None:
        if isinstance(alpha, (int, float)):
            alpha = [alpha]
        for a in alpha:
            if not 0 < a < 1:
                raise ValueError(f"alpha must be in (0, 1), got {a}")

    result = {}
    for dataset in datasets:
        name = dataset.code
        n_classes = _extract_n_classes(dataset, paradigm)
        entry: dict[str, Any] = {
            "theoretical": theoretical_chance_level(n_classes),
        }

        if alpha is not None:
            n_trials = _extract_n_trials(dataset, n_classes=n_classes)
            if n_trials is not None:
                entry["adjusted"] = {
                    a: adjusted_chance_level(n_classes, n_trials, a) for a in alpha
                }
            else:
                log.warning(
                    "Could not extract n_trials for dataset %s. "
                    "Adjusted chance levels will not be computed.",
                    name,
                )

        result[name] = entry

    return result


def _extract_n_classes(dataset, paradigm=None) -> int:
    """Extract the number of classes for a dataset/paradigm pair."""
    if paradigm is None:
        return len(dataset.event_id)
    try:
        return len(paradigm.used_events(dataset))
    except Exception as exc:
        log.debug(
            "Could not infer paradigm-specific classes for %s: %s. "
            "Falling back to dataset.event_id.",
            dataset.__class__.__name__,
            exc,
        )
        return len(dataset.event_id)


def _extract_n_trials(dataset, n_classes: int | None = None) -> int | None:
    """Extract the total number of trials per session from a dataset.

    Attempts to read from ``dataset._summary_table``. Falls back to
    None if the information is unavailable.
    """
    try:
        row = dataset._summary_table
    except AttributeError:
        return None

    if not isinstance(row, dict):
        return None

    dataset_paradigm = getattr(dataset, "paradigm", None)

    if n_classes is not None:
        if dataset_paradigm in ("imagery", "ssvep", "cvep"):
            trials_per_class = _match_int(row.get("#Trials / class", ""), default=None)
            if trials_per_class is not None:
                return trials_per_class * n_classes
        elif dataset_paradigm == "rstate":
            blocks_per_class = _match_int(row.get("#Blocks / class", ""), default=None)
            if blocks_per_class is not None:
                return blocks_per_class * n_classes

    try:
        return _compute_n_trials(row, dataset_paradigm)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        log.debug(
            "Could not parse trial info for %s: %s",
            dataset.__class__.__name__,
            exc,
        )
        return None
