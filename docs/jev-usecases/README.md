# JevK5 on public Jev use cases

Run on 2026-09-23 with `alibiserikbay/JevK5` v0.2.0 on an NVIDIA L40S. These are **36 newly written, small contract-level cases**, adapted from public integrations that call Jev. They are not JevBench questions, real customer traffic, a live browser run, or a direct comparison with Jev. The full inputs and expected decisions are in [the runner](../../examples/jev_usecase_eval.py); every output and probability is in [the result file](l40s-results.json).

| Public integration contract | Expected decisions | Important operational result |
| --- | ---: | --- |
| [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast/blob/1231850a0bf1a0c0341fe408ef1668dbbfdfac46/jev_ultrafast/model.py): next operation and observed target | 11/12 | One loading page was marked `BLOCKED` instead of `WAIT`. Median 52.8 ms for the full action; actions requiring a target used two sequential model calls. |
| [TypeSafeAI playground tool router](https://github.com/TypeSafeAI/typesafe-playground/blob/e5fc697f4a7b995b045000c0e3f915df5a7df198/docs/tool-router.md): next graph node, fixed blocks, 0.85 confidence gate | 10/12 raw, including two requests blocked before the model; 8/10 model decisions | The gate let only **1/7 actionable requests** route onward. The other six stopped for clarification, including correct low-confidence choices. A configuration change failed to reach its approval checkpoint. |
| [TypeSafeAI playground extraction](https://github.com/TypeSafeAI/typesafe-playground/blob/e5fc697f4a7b995b045000c0e3f915df5a7df198/src/extraction/extraction.ts): select a source span or `null` | 12/12 | Correct on these short, clearly labeled documents, including three ambiguous `null` cases and one instruction-in-document case. Median 26.7 ms per decision. |

## What happened

The browser-action miss was a loading screen with a disabled Settings link: JevK5 chose `BLOCKED` (probability 0.643), while `WAIT` had probability 0.308. The support-router request about a dashboard 502 chose `needs_clarification` instead of `support_agent`. A request to set a production timeout chose `needs_clarification` instead of `modify_config_tool`; because the router requires a human approval checkpoint after selecting that tool, the request never reached approval.

The routing threshold is the clearest integration problem. Of seven requests that should have advanced to a specialist/tool or approval, five had the right raw choice and two were wrong. Only the instruction-attack research case scored above 0.85 and advanced. This favors safe refusal over useful routing. The two sensitive-keyword cases were blocked by deterministic policy **without calling JevK5**, so they are not model successes. One was a benign request about password hashing, showing the policy's known overblocking behavior.

## Method and limits

- The runner uses JevK5's public `decide` API, the integrations' choice sets and task instructions, and independently authored short scenarios. Browser operation/target heads are sent sequentially because JevK5 exposes a single-question API; the original browser integration sends multiple heads in one Jev request. The browser was not operated and the separate text-writing helper was not tested.
- Extraction candidates were supplied as grounded spans, matching the model-facing stage. The playground's regular-expression candidate finder, full document handling, and user interface were not tested. JevK5 currently supports at most 16 choices per question; the extraction prototype can offer up to 100, so that wider contract would need batching or a larger answer head.
- Router scores use the playground's 0.85 selected-probability and confidence rule. JevK5's `confidence` equals its highest option probability. The exact threshold may need validation/calibration for this model; lowering it without measuring unsafe routes would be premature.
- Cases were written for this evaluation and are simple and few. They are not a random or held-out sample of deployments. The model was not tuned on them. No TypeSafe API key was used, so there is **no measured Jev baseline on these cases**. The #2 JevBench result cannot establish performance here.
- Latencies are local end-to-end `decide` calls after model load and graph capture, including tokenization, GPU work and synchronization. They exclude browser/network/tool time. PyTorch was 2.9.1+cu126.

To reproduce on a CUDA machine with JevK5 installed:

```sh
python examples/jev_usecase_eval.py --output docs/jev-usecases/l40s-results.json
```

## Next useful test

Run multi-step browser tasks with an independent final-state verifier, then measure completion, wrong clicks and recovery rather than isolated action accuracy. For tool routing, create an unseen request set and sweep thresholds for both safe-stop rate and incorrect-route rate. For extraction, include long documents and candidate sets exceeding 16 options before positioning it as a drop-in replacement.
