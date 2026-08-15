"""Strictly causal, trainable next-bar direction baseline for paper research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import json
import hashlib
import tempfile
from datetime import datetime, timezone

import numpy as np

from live.market_data import Bar


@dataclass(frozen=True, slots=True)
class ModelReport:
    train_observations: int
    validation_observations: int
    validation_accuracy: float
    validation_brier: float
    deployable_for_paper: bool
    training_symbol: str | None = None
    training_timeframe: str | None = None
    training_start: str | None = None
    training_end: str | None = None
    training_data_sha256: str | None = None
    training_config_sha256: str | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class LogisticDirectionModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    intercept: float
    report: ModelReport

    def __post_init__(self) -> None:
        if type(self.intercept) not in (int, float):
            raise ValueError("model intercept must be a numeric primitive")
        if (type(self.report.train_observations) is not int or
                type(self.report.validation_observations) is not int or
                type(self.report.validation_accuracy) not in (int, float) or
                type(self.report.validation_brier) not in (int, float) or
                type(self.report.deployable_for_paper) is not bool):
            raise ValueError("model report fields have invalid primitive types")
        optional_text = (self.report.training_symbol, self.report.training_timeframe,
                         self.report.training_start, self.report.training_end,
                         self.report.training_data_sha256, self.report.training_config_sha256,
                         self.report.created_at)
        if any(value is not None and type(value) is not str for value in optional_text):
            raise ValueError("model provenance fields must be strings or null")
        if self.report.training_symbol is not None and not self.report.training_symbol.strip():
            raise ValueError("training_symbol must not be empty")
        if self.report.training_timeframe is not None and not self.report.training_timeframe.strip():
            raise ValueError("training_timeframe must not be empty")
        for name, digest in (("training_data_sha256", self.report.training_data_sha256),
                             ("training_config_sha256", self.report.training_config_sha256)):
            if digest is not None and (len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        parsed_times: dict[str, datetime] = {}
        for name, value in (("training_start", self.report.training_start),
                            ("training_end", self.report.training_end),
                            ("created_at", self.report.created_at)):
            if value is not None:
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
                if parsed.tzinfo is None:
                    raise ValueError(f"{name} must include a timezone")
                parsed_times[name] = parsed
        if ("training_start" in parsed_times and "training_end" in parsed_times and
                parsed_times["training_start"] > parsed_times["training_end"]):
            raise ValueError("training_start must not be later than training_end")
        arrays = (self.feature_mean, self.feature_scale, self.weights)
        if any(not isinstance(array, np.ndarray) or array.shape != (4,) for array in arrays):
            raise ValueError("model parameters must be three four-element arrays")
        if any(not np.isfinite(array).all() for array in arrays) or not np.isfinite(self.intercept):
            raise ValueError("model parameters must be finite")
        if np.any(self.feature_scale <= 0):
            raise ValueError("model feature scales must be positive")
        if (not isinstance(self.report.train_observations, int) or self.report.train_observations < 1 or
                not isinstance(self.report.validation_observations, int) or self.report.validation_observations < 1):
            raise ValueError("model observation counts must be positive integers")
        if (not np.isfinite(self.report.validation_accuracy) or not 0 <= self.report.validation_accuracy <= 1 or
                not np.isfinite(self.report.validation_brier) or not 0 <= self.report.validation_brier <= 1):
            raise ValueError("model validation metrics must be finite probabilities")
        if self.report.deployable_for_paper and (
                self.report.validation_observations < 100 or
                self.report.validation_accuracy < 0.52 or
                self.report.validation_brier >= 0.25):
            raise ValueError("deployable model does not satisfy the chronological paper gate")

    def predict_probability(self, features: np.ndarray) -> float:
        features = np.asarray(features, dtype=float)
        if features.shape != self.feature_mean.shape or not np.isfinite(features).all():
            raise ValueError("feature vector has invalid shape or non-finite values")
        score = float(np.clip(((features - self.feature_mean) / self.feature_scale) @ self.weights + self.intercept, -30, 30))
        return float(1.0 / (1.0 + np.exp(-score)))

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"feature_mean": self.feature_mean.tolist(), "feature_scale": self.feature_scale.tolist(),
                   "weights": self.weights.tolist(), "intercept": self.intercept, "report": asdict(self.report)}
        encoded = json.dumps(payload, indent=2)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
                temporary_path = handle.name
                handle.write(encoded)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
            # Persist the directory entry as well as the file contents.  This
            # matters when a host loses power immediately after model rotation.
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    @classmethod
    def from_json(cls, path: Path) -> "LogisticDirectionModel":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("model payload must be an object")
        arrays = [raw.get(name) for name in ("feature_mean", "feature_scale", "weights")]
        if any(not isinstance(values, list) or any(type(value) not in (int, float) for value in values)
               for values in arrays):
            raise ValueError("model parameter arrays must contain only numeric primitives")
        intercept = raw.get("intercept")
        report = raw.get("report")
        if type(intercept) not in (int, float) or not isinstance(report, dict):
            raise ValueError("model intercept or report has an invalid type")
        for name in ("train_observations", "validation_observations"):
            if type(report.get(name)) is not int:
                raise ValueError("model observation counts must be integers")
        for name in ("validation_accuracy", "validation_brier"):
            if type(report.get(name)) not in (int, float):
                raise ValueError("model validation metrics must be numeric")
        if type(report.get("deployable_for_paper")) is not bool:
            raise ValueError("model deployment flag must be boolean")
        return cls(np.asarray(arrays[0], dtype=float), np.asarray(arrays[1], dtype=float),
                   np.asarray(arrays[2], dtype=float), intercept, ModelReport(**report))


def causal_training_matrix(bars: list[Bar], lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Features at bar t; label is the return from t to t+1. No future values leak into X."""
    if not isinstance(lookback, int) or isinstance(lookback, bool) or lookback < 5:
        raise ValueError("lookback must be an integer of at least 5 bars")
    if len(bars) < lookback + 3:
        raise ValueError(f"need at least {lookback + 3} bars")
    symbols = {bar.symbol.upper() for bar in bars}
    if len(symbols) != 1:
        raise ValueError("training bars must contain exactly one symbol")
    timestamps = [bar.timestamp.astimezone(timezone.utc) for bar in bars]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("training bars must be strictly chronological with unique timestamps")
    closes = np.asarray([bar.close for bar in bars], dtype=float)
    volumes = np.asarray([bar.volume for bar in bars], dtype=float)
    if np.any(closes <= 0) or np.any(volumes < 0):
        raise ValueError("bars contain invalid close or volume")
    log_returns = np.diff(np.log(closes))
    rows, targets = [], []
    for t in range(lookback, len(bars) - 1):
        history = log_returns[t - lookback:t]
        vol_history = volumes[t - lookback:t]
        scale = max(float(np.std(vol_history, ddof=1)), 1.0)
        rows.append([log_returns[t - 1], float(np.sum(log_returns[t - 5:t])), float(np.std(history, ddof=1)),
                     (volumes[t] - float(np.mean(vol_history))) / scale])
        targets.append(float(closes[t + 1] > closes[t]))
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def training_data_sha256(bars: list[Bar]) -> str:
    """Hash the ordered normalized bars used to fit a model."""
    digest = hashlib.sha256()
    for bar in bars:
        record = (bar.symbol.upper(), bar.timestamp.astimezone(timezone.utc).isoformat(),
                  bar.open, bar.high, bar.low, bar.close, bar.volume)
        digest.update(json.dumps(record, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def training_config_sha256(*, lookback: int, validation_fraction: float, learning_rate: float,
                           iterations: int, l2: float, timeframe: str) -> str:
    """Hash the feature and fitting configuration bound to a model artifact."""
    config = {"schema": "causal-logistic-v1", "lookback": lookback,
              "validation_fraction": validation_fraction, "learning_rate": learning_rate,
              "iterations": iterations, "l2": l2, "timeframe": timeframe}
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
                         .encode("utf-8")).hexdigest()


def validate_paper_model(model: LogisticDirectionModel, symbol: str, timeframe: str = "1Min") -> None:
    """Require a deployable artifact whose provenance matches the live paper feed."""
    report = model.report
    if not report.deployable_for_paper:
        raise ValueError("model did not pass the minimum chronological paper-research quality gate")
    expected_symbol = symbol.upper()
    if report.training_symbol != expected_symbol:
        raise ValueError("model training symbol does not match the requested paper symbol")
    if report.training_timeframe != timeframe:
        raise ValueError("model training timeframe does not match the live paper feed")
    if any(value is None for value in (report.training_start, report.training_end,
                                       report.training_data_sha256, report.training_config_sha256,
                                       report.created_at)):
        raise ValueError("model lacks complete training provenance for paper evaluation")


def validate_model_data_alignment(model: LogisticDirectionModel, bars: list[Bar],
                                  max_training_gap_seconds: float | None = None) -> None:
    """Reject a model trained after live data or beyond an optional freshness SLA."""
    if not bars:
        raise ValueError("live bars are required for model-data alignment")
    if max_training_gap_seconds is not None and (
            isinstance(max_training_gap_seconds, bool) or not isinstance(max_training_gap_seconds, (int, float))
            or not np.isfinite(max_training_gap_seconds) or max_training_gap_seconds < 0):
        raise ValueError("max_training_gap_seconds must be a finite non-negative number")
    timestamps = [bar.timestamp.astimezone(timezone.utc) for bar in bars]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("live bars must be strictly chronological")
    if model.report.training_end is None:
        raise ValueError("model lacks training_end provenance")
    training_end = datetime.fromisoformat(model.report.training_end.replace("Z", "+00:00"))
    if training_end.tzinfo is None:
        raise ValueError("model training_end must include a timezone")
    latest = timestamps[-1]
    if training_end > latest:
        raise ValueError("model training_end is later than the latest live bar")
    if (max_training_gap_seconds is not None and
            (latest - training_end).total_seconds() > float(max_training_gap_seconds)):
        raise ValueError("model training data is older than the configured freshness window")


def train_direction_model(bars: list[Bar], lookback: int = 20, validation_fraction: float = 0.30,
                          learning_rate: float = 0.08, iterations: int = 1_500, l2: float = 0.02,
                          timeframe: str = "1Min") -> LogisticDirectionModel:
    """Chronological train/validation split with deterministic full-batch logistic fitting."""
    if not isinstance(timeframe, str) or not timeframe.strip():
        raise ValueError("timeframe must be a non-empty string")
    if (isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)) or
            not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or
            not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate must be a finite positive number")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if (isinstance(l2, bool) or not isinstance(l2, (int, float)) or
            not np.isfinite(l2) or l2 < 0):
        raise ValueError("l2 must be a finite non-negative number")
    x, y = causal_training_matrix(bars, lookback)
    split = int(len(x) * (1 - validation_fraction))
    if split < 30 or len(x) - split < 20:
        raise ValueError("need at least 30 training and 20 chronologically later validation observations")
    x_train, y_train, x_val, y_val = x[:split], y[:split], x[split:], y[split:]
    mean = x_train.mean(axis=0)
    scale = np.maximum(x_train.std(axis=0, ddof=1), 1e-8)
    design = (x_train - mean) / scale
    weights, intercept = np.zeros(design.shape[1]), 0.0
    for _ in range(iterations):
        scores = np.clip(design @ weights + intercept, -30, 30)
        probabilities = 1 / (1 + np.exp(-scores))
        residual = probabilities - y_train
        weights -= learning_rate * ((design.T @ residual) / len(design) + l2 * weights)
        intercept -= learning_rate * float(residual.mean())
    val_scores = np.clip(((x_val - mean) / scale) @ weights + intercept, -30, 30)
    probability = 1 / (1 + np.exp(-val_scores))
    accuracy = float(((probability >= .5) == y_val).mean())
    brier = float(np.mean((probability - y_val) ** 2))
    report = ModelReport(
        len(x_train), len(x_val), accuracy, brier,
        bool(len(x_val) >= 100 and accuracy >= .52 and brier < .25),
        training_symbol=bars[0].symbol.upper(),
        training_timeframe=timeframe,
        training_start=bars[0].timestamp.astimezone(timezone.utc).isoformat(),
        training_end=bars[-1].timestamp.astimezone(timezone.utc).isoformat(),
        training_data_sha256=training_data_sha256(bars),
        training_config_sha256=training_config_sha256(
            lookback=lookback, validation_fraction=validation_fraction, learning_rate=learning_rate,
            iterations=iterations, l2=l2, timeframe=timeframe,
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return LogisticDirectionModel(mean, scale, weights, float(intercept), report)


def live_features(bars: list[Bar], lookback: int = 20) -> np.ndarray:
    """Build the exact four causal features using completed bars and the current quote."""
    if len(bars) < lookback + 1:
        raise ValueError(f"need at least {lookback + 1} completed bars")
    history = bars[-(lookback + 1):]
    closes = np.asarray([bar.close for bar in history], dtype=float)
    volumes = np.asarray([bar.volume for bar in history], dtype=float)
    returns = np.diff(np.log(closes))
    scale = max(float(np.std(volumes[:-1], ddof=1)), 1.0)
    return np.asarray([returns[-1], float(np.sum(returns[-5:])), float(np.std(returns, ddof=1)),
                       (volumes[-1] - float(np.mean(volumes[:-1]))) / scale], dtype=float)
