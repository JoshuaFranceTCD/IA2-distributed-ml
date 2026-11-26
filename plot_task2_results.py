import csv
from pathlib import Path

import matplotlib.pyplot as plt
runs = {
    2: {
        "csv": Path("quickstart-pytorch\output\2 Clients Draft.csv"),
        "seconds": 444.7,  # replace with measured elapsed seconds
    },
    5: {
        "csv": Path("quickstart-pytorch\output\5 Clients Draft.csv"),
        "seconds":  416.58,
    },
    10: {
        "csv": Path("quickstart-pytorch\output\10 Clients Draft.csv"),
        "seconds": 416.31,
    },
}

# Accuracy over training rounds
for clients, run in runs.items():
    with run["csv"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rounds = [int(row["Round"]) for row in rows]
    accuracy = [float(row["Test Accuracy"]) for row in rows]
    run["final_accuracy"] = accuracy[-1]

    plt.plot(rounds, accuracy, marker=".", label=f"{clients} clients")

plt.xlabel("Round")
plt.ylabel("Test accuracy")
plt.title("CIFAR-10 accuracy by client count")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("task2_accuracy_by_round.png", dpi=200)
plt.close()

clients = list(runs)
runtimes = [runs[client]["seconds"] / 60 for client in clients]

plt.bar([str(client) for client in clients], runtimes)
plt.xlabel("Number of clients")
plt.ylabel("Runtime (minutes)")
plt.title("Runtime by client count")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("task2_runtime_by_clients.png", dpi=200)
plt.close()
