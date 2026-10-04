"""Evaluate a saved car-price regression pipeline on held-out data.

The script reconstructs the same deterministic train/test split used during
training, loads the fitted joblib artifact, and calculates metrics without
calling ``fit``. It also compares the model with a train-median baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

try:
	from .data_preprocessing import DEFAULT_INPUT_PATH, create_train_test_split
except ImportError:
	from data_preprocessing import DEFAULT_INPUT_PATH, create_train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "car_price_model.joblib"


def calculate_regression_metrics(
	target: pd.Series | np.ndarray,
	predictions: np.ndarray,
) -> dict[str, float]:
	"""Calculate MAE, RMSE, and R-squared for model predictions."""
	return {
		"mae": float(mean_absolute_error(target, predictions)),
		"rmse": float(root_mean_squared_error(target, predictions)),
		"r2": float(r2_score(target, predictions)),
	}


def evaluate_model(
	model: Pipeline,
	features: pd.DataFrame,
	target: pd.Series,
) -> tuple[dict[str, float], np.ndarray]:
	"""Generate predictions and metrics without modifying or fitting the model."""
	predictions = model.predict(features)
	metrics = calculate_regression_metrics(target, predictions)
	return metrics, predictions


def print_evaluation_report(
	metrics: dict[str, float],
	baseline_metrics: dict[str, float],
	test_rows: int,
) -> None:
	"""Print metrics and a concise interpretation in Serbian."""
	mae_improvement = 1 - metrics["mae"] / baseline_metrics["mae"]

	print(f"Evaluacioni redovi: {test_rows:,}")
	print(f"MAE: {metrics['mae']:,.2f} USD")
	print(f"RMSE: {metrics['rmse']:,.2f} USD")
	print(f"R²: {metrics['r2']:.4f}")
	print(f"Baseline MAE: {baseline_metrics['mae']:,.2f} USD")
	print("\nTumačenje:")
	print(f"- Model prosečno promašuje cenu za oko {metrics['mae']:,.0f} USD (MAE).")
	print(
		f"- RMSE od {metrics['rmse']:,.0f} USD je veći od MAE, što pokazuje da "
		"pojedina vozila imaju znatno veće greške."
	)
	print(f"- Model objašnjava {metrics['r2']:.1%} varijanse cena na test skupu (R²).")
	print(
		f"- U odnosu na predviđanje medijane trening skupa, MAE je "
		f"{'smanjen' if mae_improvement >= 0 else 'povećan'} za {abs(mae_improvement):.1%}."
	)


def parse_args() -> argparse.Namespace:
	"""Parse command-line options."""
	parser = argparse.ArgumentParser(description="Evaluiraj sačuvani model cene automobila.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
	parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--random-state", type=int, default=42)
	return parser.parse_args()


def main() -> None:
	"""Load a fitted model and evaluate it on the reproducible test partition."""
	args = parse_args()
	data = pd.read_csv(args.input)
	_, features_test, target_train, target_test = create_train_test_split(
		data,
		test_size=args.test_size,
		random_state=args.random_state,
	)

	model = joblib.load(args.model)
	metrics, _ = evaluate_model(model, features_test, target_test)
	baseline_predictions = np.full(len(target_test), target_train.median())
	baseline_metrics = calculate_regression_metrics(target_test, baseline_predictions)

	print(f"Model: {args.model}")
	print_evaluation_report(metrics, baseline_metrics, len(target_test))


if __name__ == "__main__":
	main()
