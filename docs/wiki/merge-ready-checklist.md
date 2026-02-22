# Merge-ready checklist (mandatory)

Before marking any PR as merge-ready, complete all of the following:

- [ ] CI is green.
- [ ] `hobbit-7layer-acceptance-gate` workflow check is green.
- [ ] `gates/hobbit_7layer_acceptance_gate.json` exists and reports:
  - `gate = "hobbit-7layer"`
  - `status = "PASS"`
- [ ] If acceptance status is `FAIL`, PR is **not merge-ready**.

## Local verification

```bash
python scripts/check_hobbit_7layer_gate.py
```

## CI enforcement

The workflow `.github/workflows/hobbit-7layer-acceptance-gate.yml` writes a machine-readable
artifact and then hard-fails when `status != PASS`.
