#!/usr/bin/env -S uv run
"""
Convert an aeon/sktime .ts dataset into one CSV file per time series.

Each output CSV contains:

    time_index, dimension_0, dimension_1, ...

There is no header row. The time index is written as a float.

Example:

    uv run ts_to_csv.py ./Phoneme_TRAIN.ts ./output

Possible output filename:

    Phoneme_123_c1.csv


!IMPORTANT!: Initial version created with GPT-5.6, manually checked for correctness.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path


def sanitize_filename_component(value: str) -> str:
    """Make a metadata value safe to use as part of a filename."""
    value = value.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._") or "unknown"


def parse_metadata_line(line: str) -> tuple[str, str]:
    """Split a metadata line into its lower-case tag and value."""
    parts = line.split(maxsplit=1)
    tag = parts[0].lower()
    value = parts[1].strip() if len(parts) > 1 else ""
    return tag, value


def parse_dimension(raw_dimension: str, line_number: int) -> list[float]:
    """Parse one comma-separated time-series dimension."""
    raw_dimension = raw_dimension.strip()

    if not raw_dimension:
        return []

    values: list[float] = []

    for raw_value in raw_dimension.split(","):
        raw_value = raw_value.strip()

        if raw_value == "?":
            values.append(math.nan)
            continue

        if not raw_value:
            raise ValueError(
                f"Empty value encountered on input line {line_number}."
            )

        try:
            values.append(float(raw_value))
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric value {raw_value!r} on input line "
                f"{line_number}."
            ) from exc

    return values


def parse_ts_file(
    input_path: Path,
) -> tuple[str, list[tuple[list[list[float]], str | None]]]:
    """
    Parse an aeon/sktime .ts file.

    Returns:
        problem_name
        list of (dimensions, class_or_target_label)
    """
    problem_name = input_path.stem
    has_label = False
    in_data_section = False
    series: list[tuple[list[list[float]], str | None]] = []

    with input_path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if not in_data_section:
                if not line.startswith("@"):
                    raise ValueError(
                        f"Unexpected content before @data on line "
                        f"{line_number}: {line!r}"
                    )

                tag, value = parse_metadata_line(line)

                if tag == "@problemname" and value:
                    problem_name = value

                elif tag in {"@classlabel", "@targetlabel"}:
                    first_value = value.split(maxsplit=1)[0].lower()
                    has_label = first_value == "true"

                elif tag == "@data":
                    in_data_section = True

                continue

            fields = [field.strip() for field in line.split(":")]

            if has_label:
                if len(fields) < 2:
                    raise ValueError(
                        f"Expected a label on input line {line_number}."
                    )

                raw_dimensions = fields[:-1]
                label = fields[-1]

                if not label:
                    raise ValueError(
                        f"Empty label on input line {line_number}."
                    )
            else:
                raw_dimensions = fields
                label = None

            dimensions = [
                parse_dimension(dimension, line_number)
                for dimension in raw_dimensions
            ]

            if not dimensions:
                raise ValueError(
                    f"No time-series dimensions found on input line "
                    f"{line_number}."
                )

            lengths = {len(dimension) for dimension in dimensions}
            if len(lengths) != 1:
                raise ValueError(
                    f"Dimensions on input line {line_number} have different "
                    f"lengths: {sorted(lengths)}"
                )

            series.append((dimensions, label))

    if not in_data_section:
        raise ValueError("The input file does not contain an @data tag.")

    if not series:
        raise ValueError("The input file does not contain any time series.")

    return problem_name, series


def format_value(value: float) -> str:
    """Format a numeric value for CSV output."""
    if math.isnan(value):
        return "nan"
    return repr(value)


def write_series_csv(
    output_path: Path,
    dimensions: list[list[float]],
) -> None:
    """Write one multivariate time series as a headerless CSV."""
    series_length = len(dimensions[0])

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        for row_index in range(series_length):
            row = [float(row_index)]
            row.extend(dimensions[dimension][row_index] for dimension in range(len(dimensions)))
            writer.writerow(format_value(value) for value in row)


def convert_ts_file(
    input_path: Path,
    output_directory: Path,
    start_index: int,
) -> int:
    """Convert all cases in a .ts file and return the number written."""
    problem_name, series = parse_ts_file(input_path)
    safe_problem_name = sanitize_filename_component(problem_name)

    output_directory.mkdir(parents=True, exist_ok=True)

    for offset, (dimensions, label) in enumerate(series):
        series_index = start_index + offset

        filename_parts = [
            safe_problem_name,
            str(series_index),
        ]

        if label is not None:
            filename_parts.append(sanitize_filename_component(label))

        output_path = output_directory / ("_".join(filename_parts) + ".csv")
        write_series_csv(output_path, dimensions)

    return len(series)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert every time series in an aeon/sktime .ts file into "
            "a separate headerless CSV file."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input .ts file.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Directory in which the CSV files will be created.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index assigned to the first time series. Default: 0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if not args.input.is_file():
        print(
            f"Error: input file does not exist: {args.input}",
            file=sys.stderr,
        )
        return 1

    if args.input.suffix.lower() != ".ts":
        print(
            f"Warning: expected a .ts file, received {args.input.name!r}.",
            file=sys.stderr,
        )

    try:
        count = convert_ts_file(
            input_path=args.input,
            output_directory=args.output,
            start_index=args.start_index,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {count} CSV files to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())