# Smarfly2 v0 status

Date: 2026-09-27

## What exists

v0 is implemented as an artificial-ethology observer bench rather than a smarter artificial-life fly.

- A fixed-rule fly samples five cheap local features from a frame and carries three hidden states with different time constants: fast sensory trace, slow context and adaptation.
- The public observer input contains the visible world evidence and visible fly behavior but excludes those hidden variables.
- Sessions are replayable and use one 8-frame target definition with circular heading deltas.
- Three primary observer representations are implemented: present-only, a 12-frame explicit window, and resident fast-minus-slow / slow / phase state.
- Two destructive controls attack resident history while preserving current evidence: reset and shuffled-prefix history.
- A matched-present audit searches only visible variables, then reveals hidden-state distance and future divergence after matching.
- The live webcam bench runs without fitted observers and can optionally fit prediction overlays from a prior saved session. Hidden fly state is behind the `AUDIT` toggle.

## Verification receipt

Full local suite:

```text
25 passed
```

Frozen headless smoke command (the build sandbox cannot reach PyPI, so this execution used `PYTHONPATH=src` rather than an editable install):

```bash
PYTHONPATH=src python scripts/fit_observers.py --synthetic 800 --seed 7 --out results/synthetic_v0.json
```

Held-out rows: **317**.

| observer | endpoint RMSE | angular MAE |
|---|---:|---:|
| present | **102.7701** | 0.033109 |
| window | 186.8121 | **0.029504** |
| resident | 632.8767 | 0.311939 |
| resident reset | 104.2672 | 0.034743 |
| resident shuffle | 166.6283 | 0.036114 |

These numbers are descriptive, not a scientific gate. On this deterministic synthetic fixture the resident representation is plainly not the best predictor: present-only has the lowest endpoint error and the explicit window has the lowest angular error. The result is intentionally kept rather than tuned away.

The synthetic world exists to exercise the software and controls. It does not answer the project's motivating question about externally messy histories in a real webcam stream.

## What has *not* been shown

v0 has not shown that resident/windowed state improves prediction in a real scene. It has not established active probing, GAx generation inference, biological identification, consciousness, or artificial life. There is no pass/fail threshold yet for any of those claims.

## Next manual action

Run:

```bash
python scripts/run_live.py --save session.npz
```

Then move through the camera view, move a hand/light/object through the fly's cone, leave and return, and let the scene run long enough to create repeated-looking situations with different histories. Judge the fly and the prediction traces by eye first.

After collection:

```bash
python scripts/fit_observers.py session.npz
python scripts/replay_session.py session.npz
python scripts/matched_present_report.py session.npz
```

The first interesting receipt is a replayable matched-present episode where the current visible evidence is close, the futures diverge, and a history-sensitive observer predicts that divergence better than the present-only baseline. Only after seeing whether such episodes actually exist should we define a formal gate or add active pings / GAx generations.
