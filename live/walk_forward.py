"""Causal walk-forward evaluation for the paper predictive baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from live.market_data import Bar
from live.predictor import live_features, train_direction_model


@dataclass(frozen=True, slots=True)
class WalkForwardSummary:
    symbol: str
    timeframe: str
    training_bars: int
    evaluation_block_size: int
    evaluation_predictions: int
    retrain_blocks: int
    predictions: int
    accuracy: float
    brier: float
    majority_baseline_accuracy: float
    transaction_cost_bps: float
    turnover_units: float
    gross_return_bps: float
    net_return_bps: float
    max_drawdown_bps: float
    first_prediction: str
    last_prediction: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def walk_forward_evaluate(
    bars: list[Bar],
    *,
    training_bars: int = 120,
    evaluation_bars: int = 120,
    lookback: int = 20,
    timeframe: str = "1Min",
    validation_fraction: float = 0.30,
    learning_rate: float = 0.08,
    iterations: int = 1_500,
    l2: float = 0.02,
    transaction_cost_bps: float = 5.0,
) -> WalkForwardSummary:
    """Evaluate sequential blocks without using any future bar in a fit.

    Each model is fit on ``bars[:origin]`` and predicts labels for bars from
    ``origin`` through the next retraining boundary.  The target for bar t is
    the close movement from t to t+1, so the final bar is never scored.
    """
    if not isinstance(lookback, int) or isinstance(lookback, bool) or lookback < 5:
        raise ValueError("lookback must be an integer of at least 5 bars")
    if not isinstance(training_bars, int) or isinstance(training_bars, bool) or training_bars < lookback + 23:
        raise ValueError("training_bars must leave at least 30 training and 20 validation observations")
    if not isinstance(evaluation_bars, int) or isinstance(evaluation_bars, bool) or evaluation_bars < 1:
        raise ValueError("evaluation_bars must be a positive integer")
    if not isinstance(timeframe, str) or not timeframe.strip():
        raise ValueError("timeframe must be a non-empty string")
    if (isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)) or
            not isfinite(validation_fraction) or not 0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or
            not isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate must be a finite positive number")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if (isinstance(l2, bool) or not isinstance(l2, (int, float)) or
            not isfinite(l2) or l2 < 0):
        raise ValueError("l2 must be a finite non-negative number")
    if (isinstance(transaction_cost_bps, bool) or not isinstance(transaction_cost_bps, (int, float)) or
            not isfinite(transaction_cost_bps) or transaction_cost_bps < 0):
        raise ValueError("transaction_cost_bps must be a finite non-negative number")
    if len(bars) < training_bars + 2:
        raise ValueError("need at least training_bars + 2 bars for walk-forward evaluation")
    sample_count = training_bars - lookback - 1
    split = int(sample_count * (1 - validation_fraction))
    if split < 30 or sample_count - split < 20:
        raise ValueError("training_bars and validation_fraction do not leave enough fit/validation observations")

    # train_direction_model/causal_training_matrix enforce one symbol and
    # strict chronology before any model can be fit.
    origins = range(training_bars, len(bars) - 1, evaluation_bars)
    probabilities: list[float] = []
    targets: list[float] = []
    prediction_indices: list[int] = []
    for origin in origins:
        model = train_direction_model(
            bars[:origin], lookback=lookback, validation_fraction=validation_fraction,
            learning_rate=learning_rate, iterations=iterations, l2=l2, timeframe=timeframe,
        )
        stop = min(origin + evaluation_bars, len(bars) - 1)
        for t in range(origin, stop):
            probability = model.predict_probability(live_features(bars[:t + 1], lookback))
            probabilities.append(probability)
            targets.append(float(bars[t + 1].close > bars[t].close))
            prediction_indices.append(t)

    if not probabilities or not all(isfinite(value) for value in probabilities):
        raise ValueError("walk-forward evaluation produced no finite predictions")
    target_array = np.asarray(targets, dtype=float)
    probability_array = np.asarray(probabilities, dtype=float)
    majority = max(float(target_array.mean()), 1.0 - float(target_array.mean()))
    positions = np.where(probability_array >= 0.5, 1.0, -1.0)
    close_returns = np.asarray([
        np.log(float(bars[index + 1].close) / float(bars[index].close))
        for index in prediction_indices
    ], dtype=float)
    turnover = float(np.abs(np.diff(np.concatenate(([0.0], positions)))).sum())
    gross_bps = float(np.sum(positions * close_returns) * 10_000.0)
    net_bps = gross_bps - turnover * float(transaction_cost_bps)
    cumulative_net = np.cumsum(positions * close_returns) * 10_000.0
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative_net)))
    max_drawdown = float(np.max(running_peak[1:] - cumulative_net))
    return WalkForwardSummary(
        symbol=bars[0].symbol.upper(), timeframe=timeframe,
        training_bars=training_bars, evaluation_block_size=evaluation_bars,
        evaluation_predictions=len(probabilities),
        retrain_blocks=len(list(origins)), predictions=len(probabilities),
        accuracy=float(np.mean((probability_array >= 0.5) == target_array)),
        brier=float(np.mean((probability_array - target_array) ** 2)),
        majority_baseline_accuracy=majority,
        transaction_cost_bps=float(transaction_cost_bps),
        turnover_units=turnover,
        gross_return_bps=gross_bps,
        net_return_bps=net_bps,
        max_drawdown_bps=max_drawdown,
        first_prediction=bars[training_bars].timestamp.isoformat(),
        last_prediction=bars[training_bars + len(probabilities) - 1].timestamp.isoformat(),
    )
