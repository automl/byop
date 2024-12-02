from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from more_itertools import all_unique
from pytest_cases import case, parametrize, parametrize_with_cases

from amltk.optimization import Metric, Optimizer, Trial
from amltk.pipeline import Component
from amltk.pipeline.components import Choice
from amltk.profiling import Timer

if TYPE_CHECKING:
    from amltk.optimization.optimizers.hebo import HEBOOptimizer
    from amltk.optimization.optimizers.neps import NEPSOptimizer
    from amltk.optimization.optimizers.optuna import OptunaOptimizer
    from amltk.optimization.optimizers.smac import SMACOptimizer

logger = logging.getLogger(__name__)


class _A:
    pass


class _B:
    pass


metrics = [
    Metric("score_bounded", minimize=False, bounds=(0, 1)),
    Metric("score_unbounded", minimize=False),
    Metric("loss_unbounded", minimize=True),
    Metric("loss_bounded", minimize=True, bounds=(-1, 1)),
]


def target_function(trial: Trial, err: Exception | None = None) -> Trial.Report:
    """A target function for testing optimizers."""
    with trial.profile("trial"):
        # Do stuff with trail.info here
        logger.debug(trial.info)

        if err is not None:
            return trial.fail(err)

        return trial.success(
            **{
                metric_name: metric.optimal
                for metric_name, metric in trial.metrics.items()
            },
        )


def valid_time_interval(interval: Timer.Interval) -> bool:
    """Check if the start and end time are valid."""
    return interval.start <= interval.end


@parametrize("metric", [*metrics, metrics])  # Single obj and multi
def opt_smac_hpo(metric: Metric, tmp_path: Path) -> SMACOptimizer:
    try:
        from amltk.optimization.optimizers.smac import SMACOptimizer
    except ImportError:
        pytest.skip("SMAC is not installed")

    pipeline = Component(_A, name="hi", space={"a": (1.0, 10.0)})
    return SMACOptimizer.create(
        space=pipeline,
        bucket=tmp_path,
        metrics=metric,
        seed=42,
    )


@case
@parametrize("metric", [*metrics, metrics])  # Single obj and multi
def opt_optuna(metric: Metric, tmp_path: Path) -> OptunaOptimizer:
    try:
        from amltk.optimization.optimizers.optuna import OptunaOptimizer
    except ImportError:
        pytest.skip("Optuna is not installed")

    pipeline = Component(_A, name="hi", space={"a": (1, 10)})
    return OptunaOptimizer.create(
        space=pipeline,
        metrics=metric,
        seed=42,
        bucket=tmp_path,
    )


# NOTE: HEBO does not support unbounded optimals in metrics
hebo_metrics = [
    Metric("score_bounded", minimize=False, bounds=(0, 1)),
    Metric("score_unbounded", minimize=False, bounds=(-np.inf, 10)),
    Metric("loss_unbounded", minimize=True, bounds=(-10, np.inf)),
    Metric("loss_bounded", minimize=True, bounds=(-1, 1)),
]


@case
@parametrize("metric", [*hebo_metrics, hebo_metrics])  # Single obj and multi
def opt_hebo(metric: Metric, tmp_path: Path) -> HEBOOptimizer:
    try:
        from amltk.optimization.optimizers.hebo import HEBOOptimizer
    except ImportError:
        pytest.skip("HEBO is not installed")

    pipeline = Component(_A, name="hi", space={"a": (1, 10)})
    return HEBOOptimizer.create(
        space=pipeline,
        metrics=metric,
        seed=42,
        bucket=tmp_path,
    )


@case
@parametrize("metric", [*metrics, metrics])  # Single obj and multi
def opt_optuna_choice_hierarchical(metric: Metric, tmp_path: Path) -> OptunaOptimizer:
    try:
        from amltk.optimization.optimizers.optuna import OptunaOptimizer
    except ImportError:
        pytest.skip("Optuna is not installed")

    c1 = Component(_A, name="hi1", space={"a": [1, 2, 3]})
    c2 = Component(_B, name="hi2", space={"b": [4, 5, 6]})
    pipeline = Choice(c1, c2, name="hi")
    return OptunaOptimizer.create(
        space=pipeline,
        metrics=metric,
        seed=42,
        bucket=tmp_path,
    )


@case
@parametrize("metric", [*metrics])  # Single obj
def opt_neps(metric: Metric, tmp_path: Path) -> NEPSOptimizer:
    try:
        from amltk.optimization.optimizers.neps import NEPSOptimizer
    except ImportError:
        pytest.skip("NEPS is not installed")

    pipeline = Component(_A, name="hi", space={"a": (1, 10)})
    return NEPSOptimizer.create(
        space=pipeline,
        metrics=metric,
        overwrite=True,
        bucket=tmp_path,
        working_dir=tmp_path,
    )


@parametrize_with_cases("optimizer", cases=".", prefix="opt_")
def test_report_success(optimizer: Optimizer) -> None:
    """Test that the optimizer can report a success."""
    trial = optimizer.ask()
    report = target_function(trial, err=None)
    optimizer.tell(report)

    assert report.status == Trial.Status.SUCCESS
    assert valid_time_interval(report.profiles["trial"].time)
    assert report.trial.info is trial.info
    assert report.values == {
        name: metric.optimal for name, metric in optimizer.metrics.items()
    }


@parametrize_with_cases("optimizer", cases=".", prefix="opt_")
def test_report_failure(optimizer: Optimizer):
    trial = optimizer.ask()
    report = target_function(trial, err=ValueError("Error inside Target Function"))
    optimizer.tell(report)
    assert report.status is Trial.Status.FAIL

    assert valid_time_interval(report.profiles["trial"].time)
    assert isinstance(report.exception, ValueError)
    assert isinstance(report.traceback, str)
    assert report.values == {}


@parametrize_with_cases("optimizer", cases=".", prefix="opt_")
def test_batched_ask_generates_unique_configs(optimizer: Optimizer):
    """Test that batched ask generates unique configs."""
    # NOTE: This was tested with up to 100, at least from SMAC and Optuna.
    # It was quite slow for smac so I've reduced it to 10.
    # This is not a hard requirement of optimizers (maybe it should be?)
    batch = list(optimizer.ask(10))
    assert len(batch) == 10
    assert all_unique(batch)


@parametrize_with_cases("optimizer", cases=".", prefix="opt_optuna_choice")
def test_optuna_choice_output(optimizer: Optimizer):
    trial = optimizer.ask()
    keys = list(trial.config.keys())
    assert any("__choice__" in k for k in keys), trial.config


@parametrize_with_cases("optimizer", cases=".", prefix="opt_optuna_choice")
def test_optuna_choice_no_params_left(optimizer: Optimizer):
    trial = optimizer.ask()
    keys_without_choices = [
        k for k in list(trial.config.keys()) if "__choice__" not in k
    ]
    for k, v in trial.config.items():
        if "__choice__" in k:
            name_without_choice = k.removesuffix("__choice__")
            params_for_choice = [
                k for k in keys_without_choices if k.startswith(name_without_choice)
            ]
            # Check that only params for the chosen choice are left
            assert all(v in k for k in params_for_choice), params_for_choice
