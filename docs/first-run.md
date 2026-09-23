# First successful decision

This guide uses a Linux NVIDIA server, Python 3.10+, and the public JevK5 v0.2.0
release. You do not need an API key to download the public weights.

## 1. Check the machine

```bash
nvidia-smi
python3 --version
```

Use an NVIDIA GPU with bf16 support (Ampere or newer). Allow at least 16 GB of
GPU memory, with 12 GiB free, for this starting configuration; weights alone are about 9 GB and
CUDA graphs need additional space. Longer inputs need more memory. Leave roughly
25 GB of disk space for the environment and model cache.

On a minimal Ubuntu image, install Git and Python's virtual-environment support:

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-dev build-essential
```

## 2. Install in a new environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install "jevk5[fast] @ git+https://github.com/allebee/jevk5@v0.2.0"
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

The explicit PyTorch wheel uses CUDA 12.6. This avoids having pip select a newer
CUDA build than the server's driver supports. For other drivers, select a
compatible wheel from [PyTorch's installation instructions](https://pytorch.org/get-started/locally/).

## 3. Make a decision

On a shared or preconfigured server, use cache directories owned by your user:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
```

Run the Python example at the top of the [README](../README.md). The first run
downloads the weights, loads the model, compiles attention kernels, and captures
CUDA graphs. This startup is substantially slower than an individual decision;
keep the same model object loaded for subsequent calls.

For the full support-ticket example and warm latency measurement, from a checkout
containing this guide:

```bash
python examples/quickstart.py
```

Success means a `billing` decision, a probability for each of the three teams,
and measured latency. Probabilities can differ slightly between hardware and
dependency versions. Change the ticket to a software bug and check how the
distribution changes; this is a useful experiment, not an accuracy guarantee.

## 4. Optional HTTP server

```bash
jevk5-serve --model alibiserikbay/JevK5 --port 8090
```

After loading completes, in another terminal:

```bash
curl --fail http://localhost:8090/health
curl --fail http://localhost:8090/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice. Please refund the duplicate.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for money back?"}}}'
```

The HTTP server is a local development interface without authentication. Keep
port 8090 private. A yes/no question returns its true probability in `noul`.

## If the first run fails

| Symptom | Next step |
|---|---|
| `ensurepip is not available` | Install `python3-venv` (or `python3.10-venv` for Python 3.10), then recreate the environment. |
| `Python.h: No such file or directory` / Triton falls back to CPU | Install matching Python headers (`python3-dev`, or `python3.10-dev`) and `build-essential`, then restart Python. |
| `No module named pip` | Activate the new environment and use `python -m pip`; the system Python may not include pip. |
| Permission denied in the model cache | Set the user-owned cache paths above; a server image may point at another user's cache. |
| CUDA unavailable / driver mismatch | Check `nvidia-smi`, then reinstall a PyTorch CUDA wheel compatible with that driver. |
| Out of GPU memory during startup | Free other GPU allocations. For a smaller memory footprint try `JEVK5_GRAPHS=0 python examples/quickstart.py`; this changes latency. |
| Long pause before first output | Weight download, kernel compilation, and graph capture happen once. Watch download progress and `nvidia-smi`. |
| Slow first call / slow long documents | Measure after warm-up. Record input token count, hardware, and whether CUDA graphs and the `fast` extra are enabled. |

When reporting a problem, include the failing command, complete error, Python
version, `nvidia-smi`, and `python -m pip show jevk5 torch transformers flash-linear-attention`.
Do not include access tokens or private ticket contents.
