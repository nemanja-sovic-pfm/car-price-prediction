"""Derive predictive features from the cleaned car dataset.

The module preserves all input columns and appends age, annual mileage, engine
volume, threshold indicators, and a combined brand/model category. It can be
imported or run as a CLI that creates ``data/cars_features.csv``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "cars_cleaned.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "cars_features.csv"
REQUIRED_COLUMNS = {"make", "model", "year", "mileage_kilometers", "volume_cm3"}
NEWER_CAR_MAX_AGE = 10
HIGH_MILEAGE_THRESHOLD = 300_000


def engineer_car_features(
	data: pd.DataFrame,
	reference_year: int | None = None,
) -> pd.DataFrame:
	"""Return a copy of the car data with additional model features.

	Args:
		data: Cleaned car records containing all required source columns.
		reference_year: Year used to calculate vehicle age. The current year is
			used when this argument is omitted.

	Returns:
		A new DataFrame containing the source data and six derived features.

	Raises:
		ValueError: If required columns are missing or the reference year is not
			positive.
	"""
	featured = data.copy()

	missing_columns = REQUIRED_COLUMNS.difference(featured.columns)
	if missing_columns:
		missing = ", ".join(sorted(missing_columns))
		raise ValueError(f"Nedostaju obavezne kolone: {missing}")

	if reference_year is None:
		reference_year = datetime.now().year
	if reference_year <= 0:
		raise ValueError("Referentna godina mora biti pozitivan ceo broj.")

	year = pd.to_numeric(featured["year"], errors="raise")
	mileage = pd.to_numeric(featured["mileage_kilometers"], errors="raise")
	engine_volume = pd.to_numeric(featured["volume_cm3"], errors="raise")

	# Future model-year vehicles are treated as new instead of receiving
	# negative ages. A denominator of one also prevents division by zero.
	featured["car_age"] = (reference_year - year).clip(lower=0).astype("int64")
	age_for_calculation = featured["car_age"].clip(lower=1)
	featured["mileage_per_year"] = (mileage / age_for_calculation).round(2)
	featured["engine_volume_liters"] = (engine_volume / 1_000).round(3)
	featured["is_newer_car"] = featured["car_age"].le(NEWER_CAR_MAX_AGE).astype("int8")
	featured["is_high_mileage"] = mileage.gt(HIGH_MILEAGE_THRESHOLD).astype("int8")
	# Combining brand and model distinguishes names shared by manufacturers.
	featured["brand_model"] = (
		featured["make"].astype("string").str.strip()
		+ "_"
		+ featured["model"].astype("string").str.strip()
	)

	return featured


def parse_args() -> argparse.Namespace:
	"""Parse command-line options."""
	parser = argparse.ArgumentParser(description="Dodaj karakteristike podacima o automobilima.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--reference-year", type=int, default=datetime.now().year)
	return parser.parse_args()


def main() -> None:
	"""Load cleaned data, add features, and save the result."""
	args = parse_args()
	cleaned_data = pd.read_csv(args.input)
	featured_data = engineer_car_features(cleaned_data, reference_year=args.reference_year)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	featured_data.to_csv(args.output, index=False)

	new_columns = [column for column in featured_data.columns if column not in cleaned_data.columns]
	print(f"Obrađeni redovi: {len(featured_data):,}")
	print(f"Nove karakteristike: {', '.join(new_columns)}")
	print(f"Sačuvano u: {args.output}")


if __name__ == "__main__":
	main()
