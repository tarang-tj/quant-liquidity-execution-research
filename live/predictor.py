"""Strictly causal, trainable next-bar direction baseline for paper research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np

from live.market_data import Bar


@dataclass(frozen=True, slots=True)
class ModelReport:
    train_observations: int
    validation_observations: int
    validation_accuracy: float
    validation_brier: float
    deployable_for_paper: bool


@dataclass(frozen=True, slots=True)
class LogisticDirectionModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    intercept: float
    report: ModelReport

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
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "LogisticDirectionModel":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(np.asarray(raw["feature_mean"], dtype=float), np.asarray(raw["feature_scale"], dtype=float),
                   np.asarray(raw["weights"], dtype=float), float(raw["intercept"]), ModelReport(**raw["report"]))


def causal_training_matrix(bars: list[Bar], lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Features at bar t; label is the return from t to t+1. No future values leak into X."""
    if len(bars) < lookback + 3:
        raise ValueError(f"need at least {lookback + 3} bars")
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


def train_direction_model(bars: list[Bar], lookback: int = 20, validation_fraction: float = 0.30,
                          learning_rate: float = 0.08, iterations: int = 1_500, l2: float = 0.02) -> LogisticDirectionModel:
    """Chronological train/validation split with deterministic full-batch logistic fitting."""
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
    report = ModelReport(len(x_train), len(x_val), accuracy, brier,
                         bool(len(x_val) >= 100 and accuracy >= .52 and brier < .25))
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
