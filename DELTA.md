# Fork delta vs upstream (open-compass/VLMEvalKit)

Delta shape at 2026-08-11: **19 additive files, 40 invasive** (`scripts/delta_audit.sh`).
Audit this file at each upstream sync: every invasive entry must still justify
itself or be dropped. Next sync should re-cut the branch as a thematic patch
series (see `third_party/lmms-eval/DELTA.md` for the pattern).

## Additive (no sync cost)

- `vlmeval/vlm/apertus_1p5.py` — Apertus 1.5 vLLM wrapper (image-token cache,
  Emu3.5 vision tokenizer, thinking mode).
- Suite benchmarks not yet upstream (Spatial-DISE glue, MMLongBench fixes
  landed upstream since).

## Invasive (rebase tax — keep small)

- `vlmeval/config.py` — env-driven Apertus registry entry (`APERTUS_RUN_NAME`)
  + collision guard (refuses to shadow registered models); model-path override
  mechanism (`VLMEVAL_MODEL_PATH_OVERRIDES`).
- `run.py`, `inference.py`, `inference_video.py` — response-cache hooks,
  run-summary report, distributed timeout knob.
- `vlmeval/vlm/gemma.py` — transformers-5 compat (parse_response dict unwrap),
  pan-and-scan handling. Candidate for upstream PR.
- `vlmeval/smp/file.py`, API clients — judge retry/base-URL fixes; check against
  upstream before next sync, several were fixed there independently.
- `vlmeval/dataset/*` — scoring fixes (mmifeval verifier import, mmlongbenchdoc
  adaptive packing + return value, charxiv judge calls, lazy rouge import).
  MMLongBench items were upstreamed 2026-08; drop our copies at next sync if
  upstream's are equivalent.

## Upstream-PR queue

1. gemma transformers-5 parse_response compat.
2. mmifeval verifier import fix.
3. Lazy rouge import (import-time hardening).
