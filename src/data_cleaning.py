"""Clean raw car data and create a consistent intermediate dataset.

The module can be imported for its reusable cleaning functions or executed as
a command-line script. The CLI reads ``data/cars.csv`` by default and writes
``data/cars_cleaned.csv`` without modifying the raw source file.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "cars.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "cars_cleaned.csv"

REQUIRED_COLUMNS = {
	"make",
	"model",
	"price_usd",
	"year",
	"condition",
	"mileage_kilometers",
	"fuel_type",
	"volume_cm3",
	"color",
	"transmission",
	"drive_unit",
	"segment",
}
NUMERIC_COLUMNS = ["price_usd", "year", "mileage_kilometers", "volume_cm3"]
TEXT_COLUMNS = [
	"make",
	"model",
	"condition",
	"fuel_type",
	"color",
	"transmission",
	"drive_unit",
	"segment",
]
MISSING_VALUE_MARKERS = ["", " ", "NA", "N/A", "nan", "null", "none", "None", "NULL"]


def standardize_missing_values(data: pd.DataFrame) -> pd.DataFrame:
	"""Replace common textual missing-value markers with ``pd.NA``.

	Args:
		data: Source DataFrame whose values should be normalized.

	Returns:
		A new DataFrame; the source object is never modified.
	"""
	cleaned = data.copy()
	return cleaned.replace(MISSING_VALUE_MARKERS, pd.NA)


def normalize_column_name(column_name: str) -> str:
	"""Convert one column name to lowercase ``snake_case``.

	CamelCase boundaries and all non-alphanumeric separators are converted to
	underscores. Leading and trailing separators are removed.
	"""
	normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", column_name.strip())
	normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized)
	return normalized.strip("_").lower()


def clean_car_data(data: pd.DataFrame) -> pd.DataFrame:
	"""Validate and clean the complete car dataset.

	Args:
		data: Raw car records using the expected source columns.

	Returns:
		A new DataFrame with normalized names and values, imputed missing data,
		valid numeric ranges, no duplicate rows, and a consecutive index.

	Raises:
		ValueError: If one or more required columns are missing.
	"""
	cleaned = standardize_missing_values(data)
	cleaned.columns = [normalize_column_name(column) for column in cleaned.columns]

	missing_columns = REQUIRED_COLUMNS.difference(cleaned.columns)
	if missing_columns:
		missing = ", ".join(sorted(missing_columns))
		raise ValueError(f"Nedostaju obavezne kolone: {missing}")

	cleaned = cleaned.drop_duplicates().copy()

	for column in NUMERIC_COLUMNS:
		cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

	for column in TEXT_COLUMNS:
		cleaned[column] = (
			cleaned[column]
			.astype("string")
			.str.strip()
			.str.lower()
			.str.replace(r"\s+", " ", regex=True)
			.replace("", pd.NA)
		)

	# A one-year allowance supports vehicles marketed as the next model year.
	maximum_year = datetime.now().year + 1
	# These broad physical limits remove placeholder values while retaining
	# plausible rare and luxury vehicles that statistical rules may flag.
	valid_rows = (
		cleaned["price_usd"].gt(0)
		& cleaned["year"].between(1886, maximum_year)
		& cleaned["mileage_kilometers"].between(0, 2_000_000)
		& (cleaned["volume_cm3"].isna() | cleaned["volume_cm3"].between(500, 10_000))
		& cleaned["make"].notna()
		& cleaned["model"].notna()
	)
	cleaned = cleaned.loc[valid_rows].copy()

	# Median imputation is resistant to the right-skewed engine-size distribution.
	cleaned["volume_cm3"] = cleaned["volume_cm3"].fillna(cleaned["volume_cm3"].median())
	cleaned[TEXT_COLUMNS] = cleaned[TEXT_COLUMNS].fillna("unknown")

	cleaned["price_usd"] = cleaned["price_usd"].astype("int64")
	cleaned["year"] = cleaned["year"].astype("int64")
	cleaned["mileage_kilometers"] = cleaned["mileage_kilometers"].astype("int64")
	cleaned["volume_cm3"] = cleaned["volume_cm3"].round().astype("int64")

	return cleaned.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
	"""Parse command-line paths."""
	parser = argparse.ArgumentParser(description="Očisti podatke o automobilima.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
	return parser.parse_args()


def main() -> None:
	"""Load, clean, and save the car dataset."""
	args = parse_args()
	raw_data = pd.read_csv(args.input)
	cleaned_data = clean_car_data(raw_data)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	cleaned_data.to_csv(args.output, index=False)

	removed_rows = len(raw_data) - len(cleaned_data)
	print(f"Ulazni redovi: {len(raw_data):,}")
	print(f"Očišćeni redovi: {len(cleaned_data):,}")
	print(f"Uklonjeni redovi: {removed_rows:,}")
	print(f"Sačuvano u: {args.output}")


if __name__ == "__main__":
	main()
