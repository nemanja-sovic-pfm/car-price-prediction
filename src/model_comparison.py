"""Compare tree-based car-price regression algorithms consistently.

Every candidate uses the same train/test split and the same preprocessing
matrix fitted only on training data. The holdout metrics therefore compare
models under identical conditions without leaking test data into training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

try:
	from .data_preprocessing import (
		DEFAULT_INPUT_PATH,
		build_preprocessor,
		create_train_test_split,
	)
	from .model_evaluation import calculate_regression_metrics
except ImportError:
	from data_preprocessing import (
		DEFAULT_INPUT_PATH,
		build_preprocessor,
		create_train_test_split,
	)
	from model_evaluation import calculate_regression_metrics


COMPARISON_MAX_CATEGORIES = 50
COMPARISON_MIN_FREQUENCY = 10


def build_candidate_models(random_state: int = 42) -> dict[str, RegressorMixin]:
	"""Create the three unfitted regressors used in the comparison."""
	return {
		"Decision Tree": DecisionTreeRegressor(
			max_depth=20,
			min_samples_leaf=3,
			random_state=random_state,
		),
		"Random Forest": RandomForestRegressor(
			n_estimators=10,
			max_depth=10,
			min_samples_leaf=10,
			max_features="sqrt",
			n_jobs=1,
			random_state=random_state,
		),
		"Gradient Boosting": GradientBoostingRegressor(
			n_estimators=20,
			learning_rate=0.1,
			max_depth=2,
			min_samples_leaf=10,
			max_features="sqrt",
			random_state=random_state,
		),
	}


def compare_models(
	features_train: pd.DataFrame,
	features_test: pd.DataFrame,
	target_train: pd.Series,
	target_test: pd.Series,
	models: dict[str, RegressorMixin] | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
	"""Fit and evaluate regressors under identical conditions.

	Returns:
		A metrics table sorted by RMSE and the fitted pipeline for each model.
	"""
	if models is None:
		models = build_candidate_models()
	if len(models) < 3:
		raise ValueError("Poređenje zahteva najmanje tri regresiona modela.")

	results: list[dict[str, float | str]] = []
	fitted_models: dict[str, Pipeline] = {}
	preprocessor = build_preprocessor(
		max_categories=COMPARISON_MAX_CATEGORIES,
		min_frequency=COMPARISON_MIN_FREQUENCY,
	)
	print("Fitovanje zajedničkog preprocessinga...", flush=True)
	transformed_train = preprocessor.fit_transform(features_train)
	transformed_test = preprocessor.transform(features_test)
	# With bounded One-Hot cardinality the dense matrix remains modest in size
	# and avoids unstable sparse handling in tree ensembles on Windows.
	if hasattr(transformed_train, "toarray"):
		transformed_train = transformed_train.toarray()
		transformed_test = transformed_test.toarray()
	print(f"Broj transformisanih karakteristika: {transformed_train.shape[1]:,}", flush=True)

	for model_name, regressor in models.items():
		print(f"Treniranje modela: {model_name}...", flush=True)
		start_time = perf_counter()
		regressor.fit(transformed_train, target_train)
		training_seconds = perf_counter() - start_time
		predictions = regressor.predict(transformed_test)
		metrics = calculate_regression_metrics(target_test, predictions)

		results.append(
			{
				"model": model_name,
				"mae": metrics["mae"],
				"rmse": metrics["rmse"],
				"r2": metrics["r2"],
				"training_seconds": training_seconds,
			}
		)
		# The already fitted steps form a reusable prediction pipeline without
		# refitting or exposing the held-out data to preprocessing.
		fitted_models[model_name] = Pipeline(
			steps=[("preprocessor", preprocessor), ("regressor", regressor)]
		)
		print(
			f"Završeno: {model_name} (RMSE={metrics['rmse']:,.2f} USD, "
			f"vreme={training_seconds:.2f} s)",
			flush=True,
		)

	results_table = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
	return results_table, fitted_models


def print_comparison(results: pd.DataFrame) -> None:
	"""Print the metrics table and identify the best holdout model."""
	display_table = results.copy()
	display_table["mae"] = display_table["mae"].map(lambda value: f"{value:,.2f}")
	display_table["rmse"] = display_table["rmse"].map(lambda value: f"{value:,.2f}")
	display_table["r2"] = display_table["r2"].map(lambda value: f"{value:.4f}")
	display_table["training_seconds"] = display_table["training_seconds"].map(
		lambda value: f"{value:.2f}"
	)

	print("\nRezultati poređenja (niži RMSE i MAE, viši R² su bolji):")
	print(display_table.to_string(index=False))

	best_model = results.iloc[0]
	print(
		f"\nNajbolji model po RMSE metrici je {best_model['model']} "
		f"(RMSE={best_model['rmse']:,.2f} USD, MAE={best_model['mae']:,.2f} USD, "
		f"R²={best_model['r2']:.4f})."
	)


def parse_args() -> argparse.Namespace:
	"""Parse command-line options."""
	parser = argparse.ArgumentParser(description="Uporedi regresione modele za cenu automobila.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--random-state", type=int, default=42)
	return parser.parse_args()


def main() -> None:
	"""Load data, train all candidates, and print comparable holdout metrics."""
	args = parse_args()
	data = pd.read_csv(args.input)
	features_train, features_test, target_train, target_test = create_train_test_split(
		data,
		test_size=args.test_size,
		random_state=args.random_state,
	)
	models = build_candidate_models(random_state=args.random_state)
	results, _ = compare_models(
		features_train,
		features_test,
		target_train,
		target_test,
		models,
	)

	print(f"Trening redovi: {len(target_train):,}")
	print(f"Test redovi: {len(target_test):,}")
	print_comparison(results)


if __name__ == "__main__":
	main()
