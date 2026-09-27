# SequencingObserver v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the generic full-vector resident memory with a selective event-driven sequencing observer whose history changes continuation dynamics through rhythmic windows, tuft-like context, route-specific closure, and a separate publication gate.

**Architecture:** Keep Smarfly2's present-only and fixed-window controls, but add a new visible-only event path. Events feed selective fast/slow residues and a four-channel ring-like continuation core whose transitions occur in four active windows separated by dead intervals; slow context modulates channel susceptibility, mismatch closes only the offending route, and publication can be silenced without altering internal state. The trajectory readout remains a small NumPy ridge model so any gain must come from state construction rather than decoder power.

**Tech Stack:** Python 3.10+, NumPy, existing OpenCV GUI, pytest. No new ML framework or pretrained model.

**Spec:** `docs/superpowers/specs/2026-09-27-sequencing-observer-v1-design.md`

## Global Constraints

- Hidden fly state must never enter event extraction, sequencing state, normalization, training, or prediction.
- Raw absolute `(x, y)`, clock time, and `dt` are present information only; they are never written into event residues.
- Event normalization is fitted only on the contiguous training period.
- Default dynamics are fixed before evaluation: fast tau `0.30 s`, slow tau `3.0 s`, sequence period `0.8 s`, four continuation channels.
- The `0.8 s` cycle has eight equal bins: even bins activate channels `0..3` in order; odd bins are dead intervals and update no channel.
- Continuation state uses deterministic fixed inherited projections, not learned recurrent weights and not a search over the uploaded session.
- Present/window/legacy-resident/sequencing arms use identical held-out target indices.
- Publication blocking may change only external emission, never internal state or internal prediction.
- No GAx, active ping, or second observer in this implementation.
- Results are descriptive; no threshold may be invented after looking at the recorded-session result.

## Review Focus

1. **Legacy sessions without world dimensions:** loading must remain possible, but wrapped displacement must require explicit dimensions rather than silently infer them from observed coordinates.
2. **Boundary-crossing geometry:** both positive and negative wrap crossings must yield shortest toroidal displacement while ordinary motion remains unchanged.
3. **Train/test normalization leakage:** event mean/scale must be determined from training positions only even though state is rolled chronologically through test inputs.
4. **Irregular or large `dt`:** phase/window advancement and exponential residues must remain finite and deterministic; non-positive `dt` must be rejected.
5. **Publication invariance:** toggling publication block on the same input tape must leave events, residues, channel states, tuft gains, closures, and internal predictions bit-identical.

---

### Task 1: World geometry metadata and toroidal targets

**Files:**
- Modify: `src/smarfly2/session.py`
- Modify: `src/smarfly2/replay.py`
- Modify: `src/smarfly2/gui.py`
- Modify: `src/smarfly2/evaluate.py`
- Modify: `scripts/fit_observers.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_replay.py`
- Test: `tests/test_gui_smoke.py`

**Interfaces:**
- `Session(records, horizon=8, world_width: int | None = None, world_height: int | None = None)`.
- `make_targets(session: Session, horizon: int = 8, world_size: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]`.
- New sessions produced by replay/live collection populate world dimensions; legacy NPZ files load with `None` dimensions.
- A caller may supply `world_size=(width, height)` for a legacy session.

- [ ] **Step 1: Write failing wrap tests**

Add tests asserting `x: 638 -> 3` in width `640` produces `dx=+5`, `x: 2 -> 637` produces `dx=-5`, equivalent `y` cases wrap in height `480`, and an ordinary non-boundary displacement is unchanged.

- [ ] **Step 2: Run the target tests and verify RED**

Run: `pytest tests/test_metrics.py tests/test_replay.py -v`
Expected: at least one new wrap test FAILS because current targets use raw subtraction.

- [ ] **Step 3: Implement geometry-aware session metadata and shortest wrapped displacement**

Save/load `world_width` and `world_height` when present. `make_targets` uses explicit `world_size` first, otherwise session metadata. If neither exists, raise `ValueError` only when geometry-aware targets are requested by evaluation; do not guess dimensions.

- [ ] **Step 4: Add backward-compatibility and live-session tests**

Assert a legacy NPZ without dimension keys still loads, explicit `world_size` evaluates it, and a `LiveBench` record session receives dimensions from the first valid frame.

- [ ] **Step 5: Run Task 1 tests**

Run: `pytest tests/test_metrics.py tests/test_replay.py tests/test_gui_smoke.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/smarfly2/session.py src/smarfly2/replay.py src/smarfly2/gui.py src/smarfly2/evaluate.py scripts/fit_observers.py tests/test_metrics.py tests/test_replay.py tests/test_gui_smoke.py
git commit -m "fix: use toroidal trajectory targets"
```

### Task 2: Visible-only event extraction and train-only normalization

**Files:**
- Create: `src/smarfly2/events.py`
- Create: `tests/test_events.py`

**Interfaces:**
- `raw_events(records: list[FrameRecord]) -> np.ndarray` returns six columns in this exact order: `d_brightness_lr`, `d_motion`, `d_motion_lr`, `d_contrast`, `turn_innovation`, `speed_innovation`.
- `EventScaler.fit(events: np.ndarray) -> EventScaler` and `.transform(events: np.ndarray) -> np.ndarray`.
- `turn_innovation` uses wrapped heading difference; `speed_innovation` uses visible speed magnitude only.

- [ ] **Step 1: Write failing event tests**

Assert a constant visible stream becomes zero after the first row, signed left/right changes retain sign, heading crossing `+pi/-pi` produces a small wrapped innovation, and changing absolute `x/y` alone leaves all six event channels zero.

- [ ] **Step 2: Run event tests and verify RED**

Run: `pytest tests/test_events.py -v`
Expected: FAIL because `smarfly2.events` does not exist.

- [ ] **Step 3: Implement raw visible-only events**

The first row is exactly zeros. Do not read `FrameRecord.hidden`, `t`, or absolute position when constructing the six event channels.

- [ ] **Step 4: Write failing scaler leakage test**

Fit on the first half of an event matrix, mutate only the held-out half by a large offset, and assert fitted `mean_`/`scale_` are unchanged. Reject empty training data and non-finite input.

- [ ] **Step 5: Implement `EventScaler`**

Use per-column mean/std with floor `1e-12 -> 1.0`; no clipping and no test-period refit.

- [ ] **Step 6: Run Task 2 tests**

Run: `pytest tests/test_events.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/smarfly2/events.py tests/test_events.py
git commit -m "feat: add selective visible event stream"
```

### Task 3: Sequencing core — residues, rhythmic windows, continuation channels, and tuft context

**Files:**
- Create: `src/smarfly2/observers/sequencing.py`
- Create: `tests/test_sequencing_core.py`

**Interfaces:**
- `SequencingParams(fast_tau=0.30, slow_tau=3.0, period_s=0.8, n_channels=4, recurrence=0.80, transfer=0.35, event_gain=0.50, tuft_gain=0.50)`.
- `SequencingControls(no_tuft=False, no_route_closure=False, publication_block=False, flatten_phase=False)`.
- `SequencingTrace` exposes `features`, `events`, `change_residue`, `context_residue`, `channels`, `tuft_gain`, `closure`, `phase_bin`, `active_channel`, `publish_mask`.
- `SequencingFeatureBuilder.fit(event_train: np.ndarray) -> SequencingFeatureBuilder` stores only event scaling statistics.
- `SequencingFeatureBuilder.transform(events: np.ndarray, dt: np.ndarray, controls: SequencingControls | None = None) -> SequencingTrace`.

Fixed inherited channel projections are the first four rows of an 8x8 Hadamard sign basis truncated to six event columns and normalized by `sqrt(6)`. They are constants, not fitted parameters.

- [ ] **Step 1: Write failing selective-residue tests**

Assert fast/slow traces respond to event channels, remain finite under irregular positive `dt`, and reject non-positive `dt`. Assert absolute present coordinates cannot enter the residue API because `transform` consumes only the six-event matrix plus `dt`.

- [ ] **Step 2: Run core tests and verify RED**

Run: `pytest tests/test_sequencing_core.py -v`
Expected: FAIL because sequencing observer does not exist.

- [ ] **Step 3: Implement residue updates and the eight-bin scheduler**

Phase advances by actual `dt`. Even bins `0,2,4,6` activate channels `0,1,2,3`; odd bins are dead time and return active channel `-1`. `flatten_phase=True` updates all channels every frame and is used only as a destructive control/test.

- [ ] **Step 4: Add failing ordered-transition tests**

Feed a deterministic event pulse tape and assert only the scheduled channel changes in active bins, no channel changes during dead bins apart from slow closure decay added later, and `flatten_phase` destroys that one-channel-at-a-time property.

- [ ] **Step 5: Implement continuation-ring update and tuft susceptibility**

For active channel `k`, update from its own previous state, predecessor `(k-1) mod 4`, and the fixed signed projection of the current event. Slow context produces bounded channel gain `1 + tuft_gain * tanh(projection_k dot slow_context)`. `no_tuft=True` forces every gain to exactly `1.0` without altering residues.

- [ ] **Step 6: Add same-present/different-history tuft test**

Two tapes ending in the identical event row but carrying opposite earlier context must yield different channel susceptibility with tuft enabled and identical susceptibility with `no_tuft=True`.

- [ ] **Step 7: Run Task 3 tests**

Run: `pytest tests/test_sequencing_core.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/smarfly2/observers/sequencing.py tests/test_sequencing_core.py
git commit -m "feat: add phase-gated sequencing core"
```

### Task 4: Route closure and AIS-style publication separation

**Files:**
- Modify: `src/smarfly2/observers/sequencing.py`
- Create: `tests/test_sequencing_gates.py`

**Interfaces:**
- Each channel has closure scalar in `[0,1]` with default decay tau `1.2 s`.
- Each channel stores its most recent event prediction `tanh(channel_state)`; when new evidence arrives, normalized squared mismatch drives only that channel's closure toward `clip(mismatch / 3.0, 0, 1)`.
- Effective route gain is `1 - closure`; `no_route_closure=True` forces route gain `1` while preserving every other state update.
- Publication uses normalized channel activity. Gate opens when the largest surviving channel share exceeds the second largest by at least `0.05`; all-zero activity is silent.

- [ ] **Step 1: Write failing route-closure tests**

Construct a tape where one channel's prior continuation conflicts with the next event. Assert only that channel's closure rises, other channels' stored states remain intact, and closure decays toward zero on later compatible/quiet input.

- [ ] **Step 2: Run gate tests and verify RED**

Run: `pytest tests/test_sequencing_gates.py -v`
Expected: FAIL because route closure/publication behavior is not implemented.

- [ ] **Step 3: Implement route-specific mismatch and closure decay**

Do not globally reset residues or channel state after mismatch. `no_route_closure` bypasses only the multiplicative route gain.

- [ ] **Step 4: Write failing publication-invariance test**

Run the same tape twice with identical parameters, once normal and once with `publication_block=True`. Assert `events`, residues, channels, tuft gains, closure, and concatenated internal feature matrix are exactly equal while the blocked trace's `publish_mask` is all false.

- [ ] **Step 5: Implement publication mask as read-only output gating**

Publication logic may inspect internal state but must not feed back into it.

- [ ] **Step 6: Run Task 4 tests**

Run: `pytest tests/test_sequencing_core.py tests/test_sequencing_gates.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/smarfly2/observers/sequencing.py tests/test_sequencing_gates.py
git commit -m "feat: add route closure and publication gate"
```

### Task 5: SequencingObserver readout, destructive controls, and matched evaluation

**Files:**
- Modify: `src/smarfly2/observers/sequencing.py`
- Modify: `src/smarfly2/evaluate.py`
- Modify: `src/smarfly2/controls.py`
- Modify: `tests/test_end_to_end.py`
- Create: `tests/test_sequencing_evaluate.py`

**Interfaces:**
- `SequencingObserver(params: SequencingParams | None = None, alpha: float = 1e-3)`.
- `fit(records, target_indices, y, train_pos) -> SequencingObserver` fits event normalization and ridge readout using training positions only while rolling state chronologically.
- `predict_internal(records, target_indices) -> tuple[np.ndarray, SequencingTrace]` always returns internal prediction.
- `predict_published(...) -> tuple[np.ndarray, np.ndarray]` returns predictions plus publication mask; silent rows are represented by `NaN` in external prediction only.
- `shuffle_events_preserve_present(events, seed=1729) -> np.ndarray` permutes only prior event order for the sequencing control.

- [ ] **Step 1: Write failing fit/leakage test**

Mutate only test-period events by a huge constant and assert the fitted event scaler statistics and readout training rows do not change. Assert hidden audit fields may be changed arbitrarily without changing sequencing input/features.

- [ ] **Step 2: Run evaluation tests and verify RED**

Run: `pytest tests/test_sequencing_evaluate.py -v`
Expected: FAIL because the fitted SequencingObserver/evaluation arm does not exist.

- [ ] **Step 3: Implement fitted sequencing observer**

Readout feature order is fixed: present visible vector, change residue, context residue, flattened four-channel state, four tuft gains, four closures, `sin(phase)`, `cos(phase)`. The existing `RidgeReadout` remains the only trained decoder.

- [ ] **Step 4: Add destructive-control evaluation tests**

Assert `sequencing`, `sequencing_event_shuffle`, `sequencing_no_tuft`, and `sequencing_no_route_closure` produce predictions for exactly the same test indices and targets as `present`, `window`, and legacy `resident`. Event shuffle must preserve each current present sample and target.

- [ ] **Step 5: Extend `EvaluationResult` and `evaluate_session`**

Include sequencing arms plus publication rate. `sequencing_publication_block` is recorded as an invariance check, not ranked by endpoint error.

- [ ] **Step 6: Run Task 5 tests and full existing suite**

Run: `pytest tests/test_end_to_end.py tests/test_sequencing_evaluate.py -v && pytest -q`
Expected: PASS with no existing regression.

- [ ] **Step 7: Commit**

```bash
git add src/smarfly2/observers/sequencing.py src/smarfly2/evaluate.py src/smarfly2/controls.py tests/test_end_to_end.py tests/test_sequencing_evaluate.py
git commit -m "feat: evaluate sequencing observer with controls"
```

### Task 6: Matched-present sequencing audit and live/replay inspector

**Files:**
- Modify: `src/smarfly2/matched_present.py`
- Modify: `scripts/matched_present_report.py`
- Modify: `src/smarfly2/gui.py`
- Modify: `scripts/replay_session.py`
- Modify: `tests/test_matched_present.py`
- Modify: `tests/test_gui_smoke.py`

**Interfaces:**
- `MatchedPair` gains optional sequencing-audit fields populated only after visible-only pair selection: event-residue distance, tuft-gain distance, active-channel summaries, sequencing prediction divergence.
- GUI/replay inspector can display four channel activities, phase bin/dead interval, tuft gains, closures, publication state, internal trajectory, and published trajectory.
- `AUDIT` continues to control hidden fly-state visibility independently of sequencing inspector visibility.

- [ ] **Step 1: Write failing matched-present audit test**

Construct two externally identical moments with different earlier event histories. Assert pair selection remains identical whether sequencing/hidden audit data is present or absent, then assert post-selection sequencing diagnostics expose different internal state.

- [ ] **Step 2: Implement post-selection sequencing audit**

Do not use sequencing state, hidden state, or future divergence to select candidate pairs; those fields are appended only after visible nearest-neighbor selection.

- [ ] **Step 3: Write failing GUI smoke test**

Headless replay with a SequencingObserver must expose inspector state without requiring a camera or opening a window; publication-blocked frames must still expose internal trajectory state.

- [ ] **Step 4: Implement compact sequencing inspector/replay integration**

Keep GUI computation nonblocking and preserve no-model collection mode.

- [ ] **Step 5: Run Task 6 tests**

Run: `pytest tests/test_matched_present.py tests/test_gui_smoke.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/smarfly2/matched_present.py scripts/matched_present_report.py src/smarfly2/gui.py scripts/replay_session.py tests/test_matched_present.py tests/test_gui_smoke.py
git commit -m "feat: expose sequencing state in ethology audit"
```

### Task 7: Frozen synthetic receipt, recorded-session evaluation, and status documentation

**Files:**
- Modify: `scripts/fit_observers.py`
- Modify: `README.md`
- Create: `docs/V1_STATUS.md`
- Create: `results/sequencing_synthetic_v1.json`

**Interfaces:**
- `scripts/fit_observers.py` supports `--world-width/--world-height` for legacy recordings that lack stored dimensions.
- Output JSON contains every observer/control metric, publication rate, target count, world dimensions, and fixed sequencing parameter values.

- [ ] **Step 1: Run full verification before looking at scientific comparisons**

Run:

```bash
pytest -q
python -m compileall -q src scripts
```

Expected: all tests pass and compileall exits 0.

- [ ] **Step 2: Run the frozen synthetic receipt**

Run:

```bash
python scripts/fit_observers.py --synthetic 800 --seed 7 --out results/sequencing_synthetic_v1.json
```

Expected: finite internal metrics for present/window/resident/sequencing and destructive controls; publication invariance check passes.

- [ ] **Step 3: Run the existing recorded session if available in the execution workspace**

For the known legacy `session.npz`, use explicit `--world-width 640 --world-height 480`; if the file is unavailable, record that limitation rather than substituting another recording. Do not tune parameters based on this result.

- [ ] **Step 4: Write `docs/V1_STATUS.md` and README update**

Record exact fixed parameters, synthetic metrics, recorded-session metrics when available, destructive-control effects, publication rate, and whether fixed-window remains as good or better. State explicitly that a negative result means the sequencing circuit has not earned its complexity.

- [ ] **Step 5: Commit the unchanged receipt and status**

```bash
git add scripts/fit_observers.py README.md docs/V1_STATUS.md results/sequencing_synthetic_v1.json
git commit -m "docs: record SequencingObserver v1 results"
```

### Task 8: Whole-branch verification and review

**Files:**
- No planned production changes unless review finds a defect.

**Interfaces:**
- Consumes the complete v1 branch and approved spec.
- Produces review findings, fixes for Critical/Important issues via RED→GREEN, and the integration handoff.

- [ ] **Step 1: Run fresh full verification**

Run: `pytest -q && python -m compileall -q src scripts`
Expected: PASS.

- [ ] **Step 2: Review the whole branch against the spec and Review Focus**

Pay special attention to geometry metadata, hidden-state leakage, event-normalization leakage, chronological state roll, publication invariance, and whether controls alter only their intended mechanism.

- [ ] **Step 3: Fix any Critical/Important findings with a failing regression test first**

Run the full suite after the single fix pass. Defer only genuine Minor findings and report them explicitly.

- [ ] **Step 4: Hand off with finishing-a-development-branch**

Do not merge automatically; present the standard integration choices after fresh green verification.
