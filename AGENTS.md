# AGENTS.md

## Project constraints
- Do NOT commit any data from `data/pool/` or `data/versions/`.
- Dataset versions must be reproducible from manifests (pool -> manifest -> versions).
- Validation/test must be real-only. No synthetic/catalog leakage into val/test.

## Workflow rules
- Keep changes small: 1 feature = 1 commit.
- After changes, ensure smoke tests:
  - `python -m runners.train --model dummy --version v0`
  - (later) `python -m datasets.build_version --version v0`
- After each session, append 3–7 bullets to `notes/_local_log.md` (ignored by git):
  - what was done 
  - how to run/verify
  - next TODO
