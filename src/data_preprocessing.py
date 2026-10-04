"""Build preprocessing components for model-ready car data.

The module defines the regression target, feature groups, train/test splitting,
and a scikit-learn ``ColumnTransformer``. Numeric values are imputed and scaled;
categorical values are imputed and one-hot encoded. Transformers are fitted on
training data only to prevent data leakage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "cars_features.csv"
TARGET_COLUMN = "price_usd"

NUMERIC_COLUMNS = [
	"year",
	"mileage_kilometers",
	"volume_cm3",
	"car_age",
	"mileage_per_year",
	"engine_volume_liters",
	"is_newer_car",
	"is_high_mileage",
]
CATEGORICAL_COLUMNS = [
	"make",
	"model",
	"condition",
	"fuel_type",
	"color",
	"transmission",
	"drive_unit",
	"segment",
	"brand_model",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
	"""Separate predictors from the regression target.

	Args:
		data: Feature-engineered car records.

	Returns:
		A tuple containing an independent feature copy and target copy.

	Raises:
		ValueError: If required columns are missing or the target contains missing
			values.
	"""
	required_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
	missing_columns = required_columns.difference(data.columns)
	if missing_columns:
		missing = ", ".join(sorted(missing_columns))
		raise ValueError(f"Nedostaju obavezne kolone: {missing}")
	if data[TARGET_COLUMN].isna().any():
		raise ValueError(f"Ciljna kolona '{TARGET_COLUMN}' sadrži nedostajuće vrednosti.")

	features = data[FEATURE_COLUMNS].copy()
	target = data[TARGET_COLUMN].copy()
	return features, target


def build_preprocessor() -> ColumnTransformer:
	"""Create an unfitted transformer for numeric and categorical features.

	Returns:
		A ``ColumnTransformer`` ready to be included in a model pipeline.
	"""
	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			(
				"encoder",
				OneHotEncoder(handle_unknown="ignore", sparse_output=True),
			),
		]
	)

	# OrdinalEncoder is intentionally omitted because the current categories do
	# not have a defensible ordering that should be imposed on a model.
	return ColumnTransformer(
		transformers=[
			("numeric", numeric_pipeline, NUMERIC_COLUMNS),
			("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
		],
		remainder="drop",
		verbose_feature_names_out=False,
	)


def create_train_test_split(
	data: pd.DataFrame,
	test_size: float = 0.2,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Create reproducible training and test partitions.

	The returned data remains untransformed so the preprocessor can be fitted on
	the training partition inside a complete model pipeline.
	"""
	if not 0 < test_size < 1:
		raise ValueError("Veličina test skupa mora biti između 0 i 1.")

	features, target = split_features_target(data)
	return train_test_split(
		features,
		target,
		test_size=test_size,
		random_state=random_state,
	)


def parse_args() -> argparse.Namespace:
	"""Parse command-line options."""
	parser = argparse.ArgumentParser(description="Proveri preprocessing pipeline.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--random-state", type=int, default=42)
	return parser.parse_args()


def main() -> None:
	"""Fit preprocessing on training data and validate transformation of test data."""
	args = parse_args()
	data = pd.read_csv(args.input)
	features_train, features_test, target_train, target_test = create_train_test_split(
		data,
		test_size=args.test_size,
		random_state=args.random_state,
	)

	preprocessor = build_preprocessor()
	transformed_train = preprocessor.fit_transform(features_train)
	transformed_test = preprocessor.transform(features_test)

	print(f"Ciljna promenljiva: {TARGET_COLUMN}")
	print(f"Numeričke kolone ({len(NUMERIC_COLUMNS)}): {', '.join(NUMERIC_COLUMNS)}")
	print(f"Kategorijske kolone ({len(CATEGORICAL_COLUMNS)}): {', '.join(CATEGORICAL_COLUMNS)}")
	print(f"Trening skup: {len(target_train):,} redova")
	print(f"Test skup: {len(target_test):,} redova")
	print(f"Broj transformisanih karakteristika: {transformed_train.shape[1]:,}")
	print(f"Dimenzije trening matrice: {transformed_train.shape}")
	print(f"Dimenzije test matrice: {transformed_test.shape}")


if __name__ == "__main__":
	main()
