"""Export and plot centralized evaluation results from a Flower run."""

import csv
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
from flwr.app import MetricRecord
from matplotlib.ticker import PercentFormatter


def save_global_accuracy_csv(
    server_metrics: Mapping[int, MetricRecord], csv_path: Path
) -> None:
    """Write per-round centralized accuracy from Flower's result object."""
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Round", "Test Accuracy"])
        for server_round, metrics in sorted(server_metrics.items()):
            writer.writerow([server_round, metrics["accuracy"]])


def plot_accuracy_csv(csv_path: Path, plot_path: Path) -> None:
    """Create the CIFAR-10 test-accuracy plot from a saved CSV file."""
    rounds: list[int] = []
    accuracies: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            rounds.append(int(row["Round"]))
            accuracies.append(float(row["Test Accuracy"]))

    if not rounds:
        raise ValueError(f"No accuracy values found in {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(rounds, accuracies, color="#2563eb", linewidth=2, marker="o", markersize=3)
    ax.set_title("CIFAR-10")
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlim(left=0)
    ax.set_ylim(0, min(1.0, max(0.65, (max(accuracies) // 0.1 + 1) / 10 + 0.05)))
    ax.grid(True, alpha=0.3)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
