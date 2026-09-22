# Results

One directory per run, created from Phase 1 onward:

```
results/<phase>_<method>_<domain>_<timestamp>/
  config.yaml      verbatim copy of the config the run used
  env.json         GPU, driver, library versions, commit hash, dirty flag
  timings.json     raw per-prompt, per-repeat wall-clock timings
  outputs.jsonl    generated token ids (for the losslessness diff)
  metrics.json     aggregated metrics
```

`scripts/build_readme.py` reads only from here. Nothing in the README is written by
hand, and anything not measured reads `TBD` (Hard Rule 1).
