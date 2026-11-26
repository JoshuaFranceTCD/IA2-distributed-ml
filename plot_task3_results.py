"""Compare Task 3 IID and Dirichlet CIFAR-10 results in one figure."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parent


def read_metrics(csv_path: Path) -> tuple[list[int], list[float], list[float]]:
    """Read round, centralized test accuracy, and loss from a run CSV."""
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"Round", "Test Accuracy", "Test Loss"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")
        rows = sorted(reader, key=lambda row: int(row["Round"]))

    if not rows:
        raise ValueError(f"No metric rows found in {csv_path}")
    return (
        [int(row["Round"]) for row in rows],
        [float(row["Test Accuracy"]) for row in rows],
        [float(row["Test Loss"]) for row in rows],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "iid_csv",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "quickstart-pytorch" / "output" / "IID Data.csv",
        help="Task 1 IID run CSV (defaults to quickstart-pytorch/output/IID Data.csv)",
    )
    parser.add_argument(
        "non_iid_csv",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "quickstart-pytorch" / "output" / "non-iid.csv",
        help="Task 3 Dirichlet run CSV (defaults to quickstart-pytorch/output/non-iid.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "task3_data_distribution_comparison.png",
        help="Output image path",
    )
    args = parser.parse_args()

    runs = [
        ("IID", args.iid_csv, "#2563eb"),
        ("Non-IID (Dirichlet, alpha=0.5)", args.non_iid_csv, "#dc2626"),
    ]
    metrics = [(label, read_metrics(path), color) for label, path, color in runs]

    fig, (accuracy_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for label, (rounds, accuracy, loss), color in metrics:
        accuracy_ax.plot(rounds, accuracy, linewidth=2, marker=".", label=label, color=color)
        loss_ax.plot(rounds, loss, linewidth=2, marker=".", label=label, color=color)

    accuracy_ax.set_title("CIFAR-10 test accuracy")
    accuracy_ax.set_xlabel("Round")
    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    accuracy_ax.set_xlim(left=0)
    accuracy_ax.grid(True, alpha=0.3)

    loss_ax.set_title("CIFAR-10 test loss")
    loss_ax.set_xlabel("Round")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_xlim(left=0)
    loss_ax.set_ylim(bottom=0)
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    for label, (_, accuracy, loss), _ in metrics:
        print(f"{label}: final accuracy={accuracy[-1]:.4f}, final loss={loss[-1]:.4f}")
    print(f"Comparison plot saved to {args.output}")


if __name__ == "__main__":
    main()
