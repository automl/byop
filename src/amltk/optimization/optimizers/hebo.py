"""An optimizer from HEBO to optimize a HEBO design space.

### In progress
"""
# TODO
# Constraints
# Parallel ask/suggest
# I imagine iterative tell is fine, we don't need concurrent.
# Figure out what other feeatures there are
# Doc of course

from __future__ import annotations

from collections.abc import Sequence
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, overload
from typing_extensions import override

import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.general import GeneralBO
from hebo.optimizers.hebo import HEBO

import amltk.randomness
from amltk.optimization.metric import Metric
from amltk.optimization.optimizer import Optimizer
from amltk.optimization.trial import Trial
from amltk.pipeline.parsers.hebo import parser
from amltk.store import PathBucket

if TYPE_CHECKING:
    from typing import Protocol

    from hebo.optimizers.abstract_optimizer import AbstractOptimizer

    from amltk.pipeline import Node
    from amltk.types import Seed

    class HEBOParser(Protocol):
        """A protocol for HEBO design space parser."""

        def __call__(
            self,
            node: Node,
            *,
            flat: bool = False,
            delim: str = ":",
        ) -> DesignSpace:
            """See [`hebo`][amltk.pipeline.parsers.hebo.parser]."""
            ...


HEBOTrial: TypeAlias = pd.DataFrame
"""HEBO uses dataframes internally."""


class HEBOOptimizer(Optimizer[HEBOTrial]):
    """An optimizer that uses HEBO to optimize a HEBO design space."""

    def __init__(
        self,
        optimizer: AbstractOptimizer,
        metrics: Metric | Sequence[Metric],
        bucket: PathBucket | None = None,
        seed: Seed | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            optimizer: The HEBO optimizer.
            metrics: The metrics to optimize.
            bucket: The bucket to store results of individual trials from this
                optimizer.
            seed: The seed to use for trials generated from this optimizer.
        """
        metrics = metrics if isinstance(metrics, Sequence) else [metrics]
        super().__init__(metrics=metrics, bucket=bucket)
        self.optimizer = optimizer
        self.seed = seed

        # TODO: If HEBO does multi-fidelity or some other kinds of optimization,
        # this may not be sufficient
        self.name_generator = iter(f"trial_{i}" for i in count())

    @override
    def tell(self, report: Trial.Report[HEBOTrial]) -> None:
        """Tell the optimizer the report for an asked trial.

        Args:
            report: The report for a trial
        """
        raw_x = report.trial.info
        assert raw_x is not None

        # NOTE: Given a trial fail/crashed, we will have inf/worst or None for each
        # metric. As long as we fill in any missing metrics and maintain metric order,
        # than HEBO should be fine with these reported.
        # Either way, we don't actually have to look at the status of the trial to give
        # the info to hebo.
        # Make sure we have a value for each
        _lookup: dict[str, Metric.Value] = {
            v.metric.name: v for v in report.metric_values
        }
        metric_values = [
            _lookup.get(metric.name, metric.worst) for metric in self.metrics
        ]

        costs = [self.cost(v) for v in metric_values]
        raw_y = np.array([costs])  # Yep, it needs 2d, for single report tells
        self.optimizer.observe(raw_x, raw_y)

    @override
    @overload
    def ask(self, *, n_suggestions: int) -> list[Trial[HEBOTrial]]:
        ...

    @override
    @overload
    def ask(self, *, n_suggestions: None = None) -> Trial[HEBOTrial]:
        ...

    @override
    def ask(
        self,
        *,
        n_suggestions: int | None = None,
        fix_input: dict[str, Any] | None = None,
    ) -> Trial[HEBOTrial] | list[Trial[HEBOTrial]]:
        """Ask the optimizer for a trial to evaluate.

        Returns:
            A Trial
        """
        if fix_input is not None:
            # TODO: Probably fine to implement but not a priority
            # right now.
            raise NotImplementedError(
                "fix_input not yet supported for HEBOOptimizer",
            )

        match n_suggestions:
            # TODO: Allow multiple suggestions per iteration
            case int():
                raise NotImplementedError(
                    "Multiple suggestions per iteration not yet supported",
                )
            case None:
                # NOTE: Assuming for now that if I suggest without
                # anything, i.e. `n_suggestions = 1`, then I get a
                # single row dataframe.
                df_config: pd.DataFrame = self.optimizer.suggest()  # type: ignore
                assert isinstance(df_config, pd.DataFrame)
                assert len(df_config) == 1

                config: dict[str, Any] = df_config.iloc[0].to_dict()

                return Trial(
                    name=next(self.name_generator),
                    config=config,
                    bucket=self.bucket,
                    metrics=self.metrics,
                    info=df_config,
                    seed=amltk.randomness.as_int(self.seed),
                )
            case _:  # type: ignore
                raise ValueError(f"{n_suggestions=} must be `None` or `int > 0`")

    @classmethod
    def create(
        cls,
        *,
        space: Node | DesignSpace,
        metrics: Metric | Sequence[Metric],
        seed: Seed | None = None,
        bucket: PathBucket | str | Path | None = None,
        **optimizer_kwargs: Any,
    ) -> HEBOOptimizer:
        """Create an optimizer from HEBO.

        Args:
            space: The space to search over
            metrics: The metrics to optimize.

                * If `Metric`, then this is a single objective optimization with `HEBO`.
                * If `Sequence[Metric]`, then this is a multi-objective optimization
                    with `GeneralBO`.

            seed: The seed to use for trials generated from this optimizer.
            bucket: The bucket to store results of individual trials from this
                optimizer.
            **optimizer_kwargs: Keyword arguments to pass to the optimizer constructed.

        Returns:
            A HEBOOptimizer.
        """
        # TODO: Since hebo in it's observe will ignore anything that's inf, we can
        # not have metrics that have an unbounded best, as these would get ignored...
        # Probably need to raise an issue.
        # Not really sure how to report or handle that case though as it's only a
        # theoretical problem.
        _check_metrics = [metrics] if isinstance(metrics, Metric) else metrics
        if any(np.isinf(metric.optimal.value) for metric in _check_metrics):
            raise ValueError(
                "HEBO doesn't support metrics with an unbounded optimal value i.e. inf",
            )

        scramble_seed = amltk.randomness.as_int(seed)

        if isinstance(bucket, str | Path):
            bucket = PathBucket(bucket)

        space = space if isinstance(space, DesignSpace) else parser(space)

        match metrics:
            case Metric() | [Metric()]:
                optimizer = HEBO(
                    space=space,
                    scramble_seed=scramble_seed,
                    **optimizer_kwargs,
                )
            case Sequence():
                assert len(metrics) > 1
                optimizer = GeneralBO(
                    space=space,
                    num_obj=len(metrics),
                    # TODO: Not really sure if I should give a ref point or not,
                    # especially if there are unbounded metrics.
                    ref_point=np.array(cls.worst_possible_cost(metrics)),
                    **optimizer_kwargs,
                )

        return cls(optimizer=optimizer, metrics=metrics, bucket=bucket, seed=seed)

    @override
    @classmethod
    def preferred_parser(cls) -> HEBOParser:
        return parser

    @overload
    @classmethod
    def worst_possible_cost(cls, metric: Metric) -> float:
        ...

    @overload
    @classmethod
    def worst_possible_cost(cls, metric: Sequence[Metric]) -> list[float]:
        ...

    @classmethod
    def worst_possible_cost(
        cls,
        metric: Metric | Sequence[Metric],
    ) -> float | list[float]:
        """Get the crash cost for a metric for SMAC."""
        match metric:
            case Metric(bounds=(lower, upper)):  # Bounded metrics
                return abs(upper - lower)
            case Metric():  # Unbounded metric
                return np.inf
            case metrics:
                return [cls.worst_possible_cost(m) for m in metrics]

    @classmethod
    def cost(cls, value: Metric.Value) -> float:
        """Get the cost for a metric value for HEBO."""
        match value.distance_to_optimal:
            case None:  # If we can't compute the distance, use the loss
                return value.loss
            case distance:  # If we can compute the distance, use that
                return distance
