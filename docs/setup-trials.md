# Three developer setup trials

**Status: awaiting three real developer participants.** Automated checks and the
maintainer's GPU run do not count as independent developer trials.

## Task for each participant

1. Start with a fresh environment on an NVIDIA GPU machine.
2. Follow [First successful decision](first-run.md) without live coaching initially.
3. Run the README ticket example and save the result.
4. Change the ticket and make one more successful call.
5. Send the feedback below. If blocked, preserve the exact error before trying a fix.

```text
Developer / date:
OS / Python / GPU / driver:
Time from starting the guide to first successful decision:
Install command used:
Result from the supplied ticket:
Result after changing the ticket:
First confusing or broken step:
Exact error (if any):
Did the documented fix work in a fresh environment?
```

## Completion ledger

| Participant | First decision | Changed ticket | Blocker / fix | Retest |
|---|---|---|---|---|
| 1 — awaiting participant | Pending | Pending | Pending | Pending |
| 2 — awaiting participant | Pending | Pending | Pending | Pending |
| 3 — awaiting participant | Pending | Pending | Pending | Pending |

Count a trial complete only after the participant confirms both calls. Record
failures even when the maintainer can reproduce a successful run elsewhere.

## Issues found during maintainer preparation

| Environment | Issue | Resolution |
|---|---|---|
| NVIDIA Brev, Ubuntu, Python 3.10 | System Python had neither pip nor working `ensurepip` | Install `python3.10-venv`, create a fresh environment, use its `python -m pip`. |
| Same NVIDIA server | Preconfigured Hugging Face cache was not writable | Set `HF_HOME` and `HF_HUB_CACHE` to a user-owned directory. |
| Same NVIDIA server | Triton could not compile because `Python.h` was missing | Install `python3.10-dev` and `build-essential`, then restart the run. |

Outreach draft for selected participants:

> Could you try the JevK5 first-run guide on an NVIDIA GPU? The task is to install
> it, route one support ticket, then change the ticket and run it again. I need
> honest setup feedback, especially the first confusing step or error. Please
> send your GPU/Python versions and time to the first successful decision.
