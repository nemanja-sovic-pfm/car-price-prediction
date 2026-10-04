"""Train, evaluate, and persist a baseline car-price regression pipeline.

The saved artifact contains both preprocessing and a Ridge regressor, allowing
predictions directly from rows that use the original engineered-data schema.
Evaluation uses a held-out test partition that is never used to fit the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

try:
	from .data_preprocessing import (
		DEFAULT_INPUT_PATH,
		build_preprocessor,
		create_train_test_split,
	)
	from .model_evaluation import calculate_regression_metrics, print_evaluation_report
except ImportError:
	from data_preprocessing import (
		DEFAULT_INPUT_PATH,
		build_preprocessor,
		create_train_test_split,
	)
	from model_evaluation import calculate_regression_metrics, print_evaluation_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "car_price_model.joblib"


def build_model_pipeline(alpha: float = 1.0) -> Pipeline:
	"""Create an unfitted preprocessing and Ridge regression pipeline.

	Args:
		alpha: Non-negative Ridge regularization strength.

	Returns:
		An unfitted scikit-learn pipeline.

	Raises:
		ValueError: If ``alpha`` is negative.
	"""
	if alpha < 0:
		raise ValueError("Ridge alpha mora biti nenegativan.")

	# Ridge is a linear baseline that remains stable with correlated numeric
	# features and the high-dimensional sparse output from OneHotEncoder.
	return Pipeline(
		steps=[
			("preprocessor", build_preprocessor()),
			("regressor", Ridge(alpha=alpha)),
		]
	)


def parse_args() -> argparse.Namespace:
	"""Parse command-line options."""
	parser = argparse.ArgumentParser(description="Treniraj osnovni model za cenu automobila.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
	parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--alpha", type=float, default=1.0)
	return parser.parse_args()


def main() -> None:
	"""Train the baseline pipeline, evaluate it, and save the fitted artifact."""
	args = parse_args()
	data = pd.read_csv(args.input)
	features_train, features_test, target_train, target_test = create_train_test_split(
		data,
		test_size=args.test_size,
		random_state=args.random_state,
	)

	model = build_model_pipeline(alpha=args.alpha)
	model.fit(features_train, target_train)
	predictions = model.predict(features_test)
	metrics = calculate_regression_metrics(target_test, predictions)

	# A train-median predictor provides context for whether the learned model is
	# useful while keeping all baseline information out of the test partition.
	baseline_predictions = np.full(len(target_test), target_train.median())
	baseline_metrics = calculate_regression_metrics(target_test, baseline_predictions)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, args.output)

	print("Model: Ridge regresija")
	print(f"Trening redovi: {len(target_train):,}")
	print_evaluation_report(metrics, baseline_metrics, len(target_test))
	print(f"\nModel je sačuvan u: {args.output}")


if __name__ == "__main__":
	main()
