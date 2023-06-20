from __future__ import annotations

import logging

from pytest_cases import case, parametrize, parametrize_with_cases

from amltk.optimization import Optimizer, RandomSearch, Trial
from amltk.optuna import OptunaOptimizer, OptunaParser
from amltk.pipeline import Pipeline, step
from amltk.smac import SMACOptimizer
from amltk.timing import TimeInterval, TimeKind

logger = logging.getLogger(__name__)


def target_function(trial: Trial, /, time_kind: TimeKind, err=None) -> Trial.Report:
    """A target function for testing optimizers."""
    with trial.begin(time=time_kind):
        # Do stuff with trail.info here
        logger.debug(trial.info)

        if err is not None:
            raise err

        return trial.success(cost=1)

    return trial.fail(cost=2000)  # pyright: ignore


def valid_time_interval(interval: TimeInterval) -> bool:
    """Check if the start and end time are valid."""
    return interval.start <= interval.end


@case
def opt_random_search() -> RandomSearch:
    pipeline = Pipeline.create(step("hi", 1, space={"a": (1, 10)}))
    return RandomSearch(space=pipeline.space())


@case
def opt_smac_hpo() -> SMACOptimizer:
    pipeline = Pipeline.create(step("hi", 1, space={"a": (1, 10)}))
    return SMACOptimizer.create(space=pipeline.space(), seed=2**32 - 1)


@case
def opt_optuna() -> OptunaOptimizer:
    pipeline = Pipeline.create(step("hi", 1, space={"a": (1, 10)}))
    space = pipeline.space(parser=OptunaParser())
    return OptunaOptimizer.create(space=space)


@parametrize_with_cases("optimizer", cases=".", prefix="opt_")
@parametrize("time_kind", [TimeKind.WALL, TimeKind.CPU, TimeKind.PROCESS])
def test_report_success(optimizer: Optimizer, time_kind: TimeKind):
    """Test that the optimizer can report a success."""
    trial = optimizer.ask()
    report = target_function(trial, time_kind=time_kind, err=None)
    optimizer.tell(report)

    assert isinstance(report, Trial.SuccessReport)
    assert valid_time_interval(report.time)
    assert report.trial.info is trial.info
    assert report.results == {"cost": 1}


@parametrize_with_cases("optimizer", cases=".", prefix="opt_")
@parametrize("time_kind", [TimeKind.WALL, TimeKind.CPU, TimeKind.PROCESS])
def test_report_failure(optimizer: Optimizer, time_kind: TimeKind):
    """Test that the optimizer can report a success."""
    trial = optimizer.ask()
    report = target_function(
        trial,
        time_kind=time_kind,
        err=ValueError("ValueError happened"),
    )
    optimizer.tell(report)
    assert isinstance(report, Trial.FailReport)

    assert valid_time_interval(report.time)
    assert isinstance(report.exception, ValueError)
    assert isinstance(report.traceback, str)
    assert report.results == {"cost": 2000}
