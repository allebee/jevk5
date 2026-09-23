# NVIDIA L40S demo — 2026-09-23

[Watch the 27-second video](jevk5-nvidia-demo.mp4) · [Actual run data](l40s-run.json)

The video is a readable playback of a real timestamped GPU run, with pauses for
reading. It is not a screen capture. Outputs and event times come from
[`examples/quickstart.py`](../../examples/quickstart.py); no probabilities or
latencies were invented. The model was loaded before the recorded segment.

## Recorded environment

| Item | Value |
|---|---|
| GPU | NVIDIA L40S, 44.4 GiB visible to PyTorch |
| Driver / CUDA wheel | 565.57.01 / CUDA 12.6 |
| Python | 3.10.12 |
| JevK5 | v0.2.0, code commit `00234df80e6012f4bfffa484eeb655baf0b070e4` |
| Model revision | `3c673298eb7f2dc7cb98019262c55b1fac01a0bc` |
| PyTorch / Transformers | 2.9.1+cu126 / 5.17.0 |
| Accelerate / flash-linear-attention | 1.15.0 / 0.5.2 |
| Execution | bf16, batch 1, default CUDA graph lengths, temperature 1.532 |
| Input / output tokens | 152 / 0 |
| Single displayed decision | 20.60 ms |
| 30 warm calls, same ticket | p50 20.49 ms; p95 20.89 ms |
| Peak PyTorch memory | 8.48 GiB allocated; 10.86 GiB reserved |

Timing wraps the complete in-process `model.decide` call with CUDA synchronization,
including tokenization. There are three warm-up calls before recording. The p95
uses the nearest-rank method. Loading, network transport, and reading pauses are
excluded. Recorded model loading took 19.61 seconds with weights already cached;
this is not a cold-download startup measurement. PyTorch memory counters exclude
some CUDA/runtime allocations and are not a universal minimum-VRAM guarantee.

`causal_conv1d` was not installed; Transformers used its reference convolution
implementation. This is the tested installation, not a claim of optimal performance.

## Reproduce

Follow the [first-run guide](../first-run.md), then run from this checkout:

```bash
python examples/quickstart.py --record demo-run.json
```

The model reference defaults to the public Hugging Face repository and may change
later. To reproduce these exact weights, download the model revision above and
pass its local directory with `--model`. The JSON records the resolved revision,
package versions, all 30 samples, memory use, and timestamped display events.

## Additional onboarding validation

[`server-check.json`](server-check.json) records successful `/health` and
`/v1/systemone` calls with the real model. A refund question returned 0.8575 true
probability; a changed software-bug ticket routed to `technical` with 0.99495
probability. These functional checks used eager execution without CUDA graphs;
their HTTP timings are not the demo's warm latency measurements.

These are maintainer checks. The [three developer trials](../setup-trials.md)
remain pending actual participants.
