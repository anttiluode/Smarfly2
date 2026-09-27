# SequencingObserver v1 status

Date: 2026-09-27

## What changed

v1 replaces the generic "low-pass every visible coordinate" resident idea with a selective sequencing observer:

- six visible-only innovations are extracted from sensory/behavioral changes;
- only those events are deposited into fast (`0.30 s`) and slow (`3.0 s`) residues;
- four fixed inherited continuation channels advance in a `0.8 s` cycle with four active windows separated by four dead intervals;
- slow context modulates channel susceptibility (tuft analogue);
- prediction conflict raises route-specific closure that decays with `1.2 s` time constant (SST/Martinotti analogue);
- internal prediction is always computed, while publication is a separate gate (AIS/chandelier analogue);
- the trajectory decoder remains the same small ridge readout.

These are computational analogies, not biological identification claims.

## Geometry correction

The v0 fly wraps around the image boundaries, but the original trajectory target used ordinary subtraction. v1 stores world width/height and uses shortest toroidal `(dx, dy)` displacement. Legacy recordings can still be loaded but require explicit dimensions for geometry-aware evaluation.

## Fixed parameters

No parameter was tuned on the recorded webcam session.

| parameter | value |
| --- | ---: |
| fast tau | 0.30 s |
| slow tau | 3.0 s |
| sequence period | 0.8 s |
| continuation channels | 4 |
| recurrence | 0.80 |
| predecessor transfer | 0.35 |
| event gain | 0.50 |
| tuft gain | 0.50 |
| closure decay tau | 1.2 s |

The inherited event projections are fixed Hadamard-sign coordinates rather than learned recurrent weights.

## Frozen synthetic receipt

Command:

```bash
python scripts/fit_observers.py --synthetic 800 --seed 7 --out results/sequencing_synthetic_v1.json
```

Held-out target count: **317**. Publication rate: **0.6845**. Publication-block invariance: **PASS**.

| observer/control | endpoint RMSE ↓ | angular MAE ↓ |
| --- | ---: | ---: |
| present | 0.7829 | 0.0361 |
| 12-frame window | **0.4189** | **0.0330** |
| legacy resident | 2.4860 | 0.3194 |
| sequencing | 1.2464 | 0.1614 |
| sequencing event shuffle | 6.3199 | 0.8330 |
| sequencing no tuft | 1.2712 | 0.1724 |
| sequencing no route closure | 1.2584 | 0.1673 |

The fixed window remains clearly better than SequencingObserver on this synthetic fixture. Event-order destruction hurts heavily, showing that the sequencing state is using temporal order, but the special circuit has **not earned its added complexity** on this benchmark.

## Recorded webcam session

The uploaded legacy `session.npz` contains **4,432 frames**. It was evaluated with explicit `640 x 480` geometry:

```bash
python scripts/fit_observers.py session.npz --world-width 640 --world-height 480
```

Held-out target count: **1,770**. Publication rate: **0.3661**. Publication-block invariance: **PASS**.

| observer/control | endpoint RMSE ↓ | angular MAE ↓ |
| --- | ---: | ---: |
| present | 0.8779 | **0.0662** |
| 12-frame window | **0.8602** | 0.1910 |
| legacy resident | 11.4953 | 0.9211 |
| sequencing | 1.9314 | 0.1652 |
| sequencing event shuffle | 3.9172 | 0.1859 |
| sequencing no tuft | 3.5262 | 0.1932 |
| sequencing no route closure | 1.9189 | 0.1670 |

This is a mixed but informative negative result:

- SequencingObserver does **not** beat the simple present or fixed-window baseline on endpoint RMSE.
- It predicts heading better than the 12-frame window, but worse than present-only.
- Destroying event order degrades endpoint prediction substantially (`1.93 -> 3.92`).
- Removing tuft/context modulation degrades endpoint prediction even more (`1.93 -> 3.53`).
- Removing route closure changes little and is slightly better on endpoint RMSE (`1.93 -> 1.92`), so route closure has not earned itself yet.

Thus the useful observation is not "the neural-inspired system wins." It is narrower: **selective ordered history and contextual susceptibility carry real predictive signal in this implementation, but the complete v1 circuit does not yet convert that signal into a better overall predictor than the simple controls.**

## Legacy control performance fix

The old v0 `resident_shuffle` recomputed a full shuffled prefix for every row, which is quadratic and timed out on this 4,432-frame recording. It is now a causal single-pass randomized-history control: the visible current row remains unchanged, persistent state receives only randomly selected prior samples, and no future sample is used. This preserves the control's purpose while making long-session evaluation practical.

## Claim boundary

v1 demonstrates a working software architecture with selective writes, rhythmic sequencing, tuft-style susceptibility, route-specific closure, publication separation, and destructive controls. It does not establish that this architecture exists in cortex, nor that it is superior to ordinary short-window prediction.

The next scientific move should **not** be parameter tuning on this session. The two mechanisms that currently show the strongest ablation effects are event order and tuft/context modulation. Route closure should remain under suspicion until a task is constructed where local sequence failure actually needs route-specific suppression.

Only after one-observer sequencing earns clearer predictive value should the project move to the two-observer resonance experiment.
