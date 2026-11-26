# IA2-distributed-ml

## Run the Task 1 baseline on Windows

Use Python 3.12 (recommended) or 3.11.Lab 10 instructions caution against Python 3.13 or newer.
From the repository root, create and activate a virtual environment, install the
Flower app and its dependencies, set the runtime variable, then start the run:

```powershell
cd quickstart-pytorch
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = "1"
flwr run . local --stream `
    --run-config "num-server-rounds=50 local-epochs=2 learning-rate=0.001 fraction-evaluate=0.5" `
    --federation-config "num-supernodes=10"
```

`pip install -e .` installs the dependencies listed in `quickstart-pytorch/pyproject.toml`
and the `pytorchexample` app. The environment variable tells Flower not to reinstall
those dependencies at run time; set it again in each new PowerShell window. The run
configuration uses 50 server rounds, 2 local epochs, learning rate 0.001, evaluation
fraction 0.5, and 10 simulated clients for the IID baseline.

## Run outputs

After training finishes, the server saves its centralized CIFAR-10 test accuracy
for round 0 and every completed round to a run-specific CSV, then creates a PNG
plot from that CSV. The output paths are printed in the Flower run output. To
regenerate a plot later, pass the CSV path to the helper script from the repository
root:

```powershell
python plot_task1_accuracy.py quickstart-pytorch\output\cifar10_global_accuracy_run_<run-id>.csv
```
