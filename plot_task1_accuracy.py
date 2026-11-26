"""Regenerate a Flower test-accuracy plot from a per-round CSV file."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "quickstart-pytorch"))

from pytorchexample.metrics import plot_accuracy_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="CSV file containing Round and Test Accuracy columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path (defaults to the CSV path with a .png extension)",
    )
    args = parser.parse_args()
    plot_path = args.output or args.csv_path.with_suffix(".png")
    plot_accuracy_csv(args.csv_path, plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
