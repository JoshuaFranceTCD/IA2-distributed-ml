"""Generate accuracy and loss plots from a Task 1 per-round CSV file."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "quickstart-pytorch"))

from pytorchexample.metrics import (  # noqa: E402
    plot_accuracy_csv,
    plot_loss_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="CSV file containing Round, Test Accuracy, and Test Loss columns",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Output filename prefix (defaults to the CSV filename without extension)",
    )
    args = parser.parse_args()

    prefix = args.output_prefix or args.csv_path.with_suffix("")
    accuracy_path = prefix.with_name(f"{prefix.name}.png")
    loss_path = prefix.with_name(f"{prefix.name}_loss.png")
    plot_accuracy_csv(args.csv_path, accuracy_path)
    plot_loss_csv(args.csv_path, loss_path)
    print(f"Accuracy plot saved to {accuracy_path}")
    print(f"Loss plot saved to {loss_path}")


if __name__ == "__main__":
    main()
