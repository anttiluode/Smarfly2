# Smarfly2 v0 — Artificial Ethology Observer Bench

Date: 2026-09-27

## Purpose

Smarfly2 is not a smarter artificial-life fly. It is a research bench for studying whether an observer with persistent temporal state can infer hidden causes from visible behavior in a live, messy world.

The fly is deliberately simple and partially hidden. The observer is the scientific object.

The first version asks one narrow question:

> Can a history-sensitive observer predict a fly's short future trajectory better than a present-only observer when externally similar presents can arise from different hidden histories?

The webcam supplies external complexity the simulator does not control. The observer receives world evidence plus visible fly behavior, but never the fly's hidden state or hidden rule parameters.

## Scope

### In v0

- One to a few fixed-rule flies overlaid on a live webcam feed.
- Each fly has a small hidden dynamical state with distinct time scales.
- The observer sees:
  - the webcam evidence available around the fly,
  - fly position and heading,
  - recent visible trajectory,
  - time.
- The observer does not see the fly's hidden internal variables or authored rule coefficients.
- The observer predicts a short future trajectory.
- A memoryless baseline receives the same current visible evidence but no history.
- A simple history baseline receives a fixed recent window.
- The primary observer uses persistent/windowed resident state.
- Hidden fly state is logged only for audit after predictions have been made.
- The program records episodes where visible present state is closely matched but hidden history differs, so history dependence can be inspected directly.
- The live GUI is first-class: we should be able to run it on a webcam and judge the behavior visually before any formal gate is introduced.

### Explicitly not in v0

- Genetic evolution / GAx generations.
- Active perturbation or experimental pings chosen by the observer.
- Multiple observers communicating with each other.
- LLM, diffusion, DINO, or other heavyweight foundation models in the inner loop.
- Claims about consciousness, emotion, mirror neurons, or biological identification.
- Large neural networks whose capacity can hide the mechanism.

These are later stages only after v0 demonstrates a real history-sensitive prediction problem.

## Architecture

```text
LIVE WEBCAM WORLD
      |
      +--> fly local visual cone
      |       |
      |       v
      |   hidden-state fly dynamics
      |       |
      |       v
      |   visible movement
      |
      +------------------------------+
                                     v
                              OBSERVER INPUT
                     world evidence + fly behavior
                                     |
                     +---------------+---------------+
                     |               |               |
                     v               v               v
                present-only     fixed-window    resident/windowed
                  baseline         baseline         observer
                     |               |               |
                     +---------------+---------------+
                                     v
                          short-horizon trajectory
                              prediction + error
                                     |
                                     v
                           audit against hidden state
```

The important asymmetry is intentional:

```text
simple hidden fly  <  complex observer
```

The fly generates a partially observable process. The observer must model it.

## Fly model

The fly should be simple enough to understand completely and rich enough that current visible state is not always sufficient.

### Visible state

- position `(x, y)` on the webcam frame,
- heading angle,
- velocity,
- local visual cone geometry.

### Hidden state

Use three low-dimensional variables with distinct roles and time constants:

1. **fast sensory trace** `s_t`
   - integrates recent local brightness / motion asymmetry,
   - decays quickly,
   - affects near-term turning.

2. **slow contextual bias** `c_t`
   - integrates a longer-lived directional/environmental tendency,
   - decays slowly,
   - biases which of otherwise similar visual inputs wins.

3. **adaptation / refractory state** `a_t`
   - rises after strong turns or bursts of movement,
   - suppresses repeating the same response immediately,
   - creates same-present/different-future cases after different histories.

The exact equations should be compact, deterministic given a random tape, and logged. A small amount of seeded process noise is allowed, but v0 must not depend on noise for its main history effect.

### Environment coupling

The fly samples a cheap local visual cone from the webcam at video rate. Extract only simple features in v0, e.g.:

- mean brightness,
- left/right brightness imbalance,
- frame-to-frame local motion energy,
- left/right motion imbalance,
- coarse contrast/variance.

No pretrained vision model is needed.

### Action law

Movement must depend on both current visual evidence and hidden state. It should not be a direct threshold-to-action table. A compact continuous policy is preferred, for example:

```text
turn_drive = visual_asymmetry
           + k_s * fast_trace
           + k_c * slow_context
           - k_a * signed_adaptation

speed = base_speed + motion_gain * local_motion - adaptation_cost
```

Hidden state is updated before or after action according to one documented convention and tested for determinism.

## Observer input

At each frame the observer receives only externally measurable values:

- fly `(x, y)`, heading, velocity,
- current local visual feature vector,
- optional globally cheap frame statistics shared by all methods,
- elapsed time / `dt`.

It never receives `s_t`, `c_t`, `a_t`, hidden coefficients, or a rule label.

All observer arms receive identical visible inputs and identical train/test episodes.

## Prediction task

Predict the fly's displacement and heading change over a short fixed horizon, initially 5–10 frames.

Targets:

```text
Delta x_H
Delta y_H
Delta heading_H
```

The task is continuous regression, not a hand-authored behavior label.

Primary metrics:

- trajectory endpoint RMSE,
- angular error,
- optional full-horizon path RMSE.

The GUI should visualize predicted short trajectories as faint forward traces for each observer.

## Observer arms

### 1. Present-only baseline

A small linear/ridge model from current visible features to future trajectory.

Purpose: measure how predictable the fly is without memory.

### 2. Fixed-window baseline

The same style of model on a flattened recent window or compact hand-made recent statistics.

Purpose: distinguish generic short memory from persistent resident state.

### 3. Resident/windowed observer

A compact stateful model whose internal coordinates persist across frames and update in temporally separated channels.

Initial design:

```text
fast residue      <- current innovation, short time constant
slow residue      <- compressed history, long time constant
phase/window state <- cyclic temporal coordinate
readout           <- current visible evidence + fast + slow + phase
```

The observer should remain small and interpretable. v0 does not need a deep RNN.

The point is not to prove a particular cortical mechanism. The point is to ask whether factorized temporal state earns predictive information on a partially observed behavioral process.

## Matched-present audit

This is the most important qualitative analysis in v0.

From recorded episodes, search for pairs of moments with close visible state:

```text
same / near-same position
same / near-same heading
similar current local visual features
similar current speed
```

but substantially different hidden fly state or different recent history.

For each matched pair, compare:

- true future trajectory,
- present-only prediction,
- fixed-window prediction,
- resident observer prediction,
- hidden-state difference revealed only after scoring.

The GUI/report should expose a few of these pairs visually. This turns "history matters" into something a person can inspect rather than only a scalar score.

## Data collection and training

The live app should support two modes:

### Observe / collect

- webcam runs,
- fly moves online,
- visible input and hidden audit state are logged,
- observer predictions may run if fitted models already exist.

### Fit / replay

- train observers on recorded sessions,
- evaluate on later held-out contiguous segments or separate sessions,
- replay the exact same video / fly tape for all observer arms.

Avoid random frame splitting because adjacent video frames would leak temporal context.

A synthetic deterministic frame source should also exist for tests and headless CI. It is a software test fixture, not the scientific world.

## GUI

Keep the old Dumbflies/Smartfly immediacy rather than building a research dashboard first.

Main view:

- live webcam,
- one or a few visible flies,
- each fly's visual cone,
- short predicted path overlays for the observer arms,
- a compact strip showing prediction error over time.

Optional inspector panel:

- visible feature values,
- observer resident coordinates,
- hidden fly state behind an explicit `AUDIT` toggle.

The hidden state should be off by default so watching the fly feels like an ethology experiment rather than a simulator debugger.

## Scientific discipline

v0 should make no claim merely because the windowed observer looks sophisticated.

Important controls:

- present-only baseline,
- fixed-window baseline,
- history-shuffled observer input,
- observer-state reset during replay,
- same model/readout budget where practical,
- seeded replay so every arm sees identical fly and webcam-derived inputs.

A positive v0 result would mean only:

> On this partially observed fixed-rule fly system, visible history contains predictive information not available in the current visible state, and the resident/windowed observer captures some of it better than the registered simpler baselines.

A negative result is equally useful: if a fixed recent window matches the resident observer, the special temporal state has not earned complexity yet.

## Testing

### Unit tests

- hidden fly state update determinism under a fixed tape,
- state variables remain bounded,
- observer cannot access hidden fly fields through its public input object,
- present-only arm receives exactly one frame of visible evidence,
- reset and shuffle controls actually destroy the intended history while preserving current visible inputs,
- angular metrics wrap correctly,
- replay reproduces recorded fly trajectories from the same tape when expected.

### Integration tests

- headless synthetic source can collect a session, fit all observers, and evaluate them end to end,
- live mode can start without fitted observer weights,
- saved session can be replayed without a webcam,
- all observer arms score the exact same held-out frames.

### Visual/manual tests

- move a hand/light through the fly's visual field and verify the fly responds continuously rather than by obvious binary triggers,
- produce at least one matched-present episode with visibly diverging futures,
- verify observer path overlays update without blocking camera capture.

## Repository layout

Proposed structure:

```text
Smarfly2/
  README.md
  pyproject.toml
  src/smarfly2/
    fly.py
    vision.py
    world.py
    records.py
    observers/
      base.py
      present.py
      window.py
      resident.py
    metrics.py
    replay.py
    gui.py
  scripts/
    run_live.py
    fit_observers.py
    replay_session.py
    matched_present_report.py
  tests/
    test_fly.py
    test_observer_inputs.py
    test_controls.py
    test_replay.py
    test_end_to_end.py
  docs/
    superpowers/specs/
```

Keep files small and responsibilities explicit.

## Stages after v0

### Stage 1 — active ping / experimental intervention

The observer maintains competing predictions and can choose a cheap perturbation that should maximally separate them. Candidate perturbations could include localized screen overlay flashes or controlled visual stimuli rendered into the fly's view, while leaving the webcam world otherwise intact.

This is where the Ping/Listener, ReadWrite, active-observation, and tomography line enters directly.

### Stage 2 — GAx generations

Evolve a small set of fly parameters across generations. The observer watches phenotype trajectories and generational distributions without access to genotype/fitness internals.

Questions:

- Can generational history forecast future population behavior?
- Can the observer infer when the underlying rule/operator has changed?
- Can different latent computational/behavioral strategies produce the same current visible behavior but diverge under an intervention?

GAx supplies the mathematical reference case for when fixed mutation-selection dynamics are projectively linear and when generational history exposes a hidden operator on the excited subspace.

### Stage 3 — observer-of-observer

Only after the first two stages work: a second observer watches the first observer's probes and predictions. This opens the higher-order loop without pretending it exists in v0.

## Success criterion for v0

The project is worth continuing if, during real or replayed webcam sessions, we can find robust episodes where:

1. current visible evidence is closely matched,
2. different histories lead to different fly futures,
3. the present-only predictor cannot resolve them,
4. at least one history-sensitive observer predicts the divergence better,
5. the effect survives replay and history-shuffle/reset controls.

The GUI should make at least one such episode understandable by eye.

That is the first earned bridge from the old Dumbflies/Smartfly demos into the recent work on resident state, temporal windows, observability, and active probing.