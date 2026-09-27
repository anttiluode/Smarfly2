# Smarfly2

**A webcam-driven artificial ethology bench: keep the fly simple; put the difficult computation in the observer.**

Smarfly2 is the continuation of the old Dumbflies / SmartFly line after reversing the emphasis. The fly is a deliberately small partially observed dynamical system. It moves over a live webcam image using five cheap visual measurements plus three hidden states with different time constants. The research object is an outside observer that sees the same measurable world evidence and the fly's visible behavior, but never those hidden states.

The current v1 question is narrower and harder:

> Can history become useful when it is deposited selectively as events into an ordered continuation system, rather than stored as generic low-pass copies of the whole visible state?

## What is running

```text
physical webcam world
        |
        +--> fly local cone --> hidden fast/slow/adaptation state --> movement
        |
        +--> visible world evidence + visible fly behavior
                             |
                +-------------+----------------+------------------+
                |             |                |                  |
            present       12-frame       legacy resident     sequencing
             ridge          ridge          fast/slow        event residues
                                                                |
                                                  phase windows -> routes
                                                  tuft gain -> closure
                                                  internal -> publication
                +-------------+----------------+------------------+
                                      |
                              8-frame trajectory
                                  prediction
```

The observer input boundary is explicit in code. Hidden fly state is stored only beside the public record for later audit; it is not part of `FrameRecord.observer_vector()`.

## Install and run

Python 3.10+:

```bash
python -m pip install -e ".[test]"
pytest -q
```

Live webcam bench:

```bash
python scripts/run_live.py
```

Close the window to save `session.npz`. Use `--camera 1` for another camera, `--save my-session.npz` for a different output, or use a prior session to fit the three live path overlays:

```bash
python scripts/run_live.py --model-session old-session.npz --world-width 640 --world-height 480
```

Fit/evaluate observers on a recorded session:

```bash
python scripts/fit_observers.py session.npz
```

New v1 sessions store image dimensions. For an old recording without geometry metadata, supply them explicitly:

```bash
python scripts/fit_observers.py session.npz --world-width 640 --world-height 480
```

Replay the recorded fly trajectory without a camera, optionally exposing sequencing state:

```bash
python scripts/replay_session.py session.npz --inspect-sequencing --world-width 640 --world-height 480
```

Find externally similar moments separated in time, then reveal hidden-state and future differences only after matching:

```bash
python scripts/matched_present_report.py session.npz --sequencing --world-width 640 --world-height 480
```

Headless deterministic smoke run:

```bash
python scripts/fit_observers.py --synthetic 800 --seed 7 --out results/sequencing_synthetic_v1.json
```

## What the fly actually is

The five measured visual features are local brightness, left-right brightness imbalance, local motion energy, left-right motion imbalance, and contrast. They drive a fast sensory trace (~0.25 s), slow contextual trace (~3 s), and signed adaptation/refractory state (~0.8 s). The action law is continuous; there is no `if person then chase` or mating/food story.

The fly is intentionally not the clever part.

## What the observer is

All observer arms predict the same `(dx, dy, dheading)` eight frames ahead on one contiguous held-out segment.

- `present`: ridge regression from the current visible sample only.
- `window`: the same readout over the previous 12 visible samples.
- `resident`: current evidence plus fast-minus-slow residue, slow state, and a cyclic temporal coordinate.
- `resident_reset`: destroys persistent state on every sample while keeping current evidence/targets fixed.
- `resident_shuffle`: destroys ordered legacy history while preserving the current visible sample.
- `sequencing`: writes only six visible innovations into fast/slow event residues, advances four continuation routes in rhythmic active/dead windows, modulates them by slow context, applies route-specific closure after mismatch, and separates internal prediction from publication.
- `sequencing_event_shuffle`: destroys event order.
- `sequencing_no_tuft`: removes contextual susceptibility while preserving residues.
- `sequencing_no_route_closure`: leaves routes open while preserving the rest of the state machine.

The biological names used in the design (tuft, SST/Martinotti, AIS/chandelier) are functional analogies, not claims that the program identifies those biological circuits. The decoder is still only ridge regression: the interesting part must be earned by the temporal state construction.

The synthetic source exists to test software and replay. It is not the scientific world. See `docs/V1_STATUS.md` for the frozen synthetic and webcam-session receipts.

## Matched-present audit

The most important qualitative object is not an aggregate score. It is a pair of moments where position, heading, velocity and local visual evidence are very similar, but the prior histories differ and the fly subsequently does different things.

`matched_present_report.py` searches using visible variables only. Hidden-state distance and future divergence are attached after candidate matching. That lets us inspect whether an apparent history effect corresponds to a real hidden distinction without giving the search access to the answer.

## Claim boundary

Smarfly2 v1 is **not** a consciousness model, life simulation, mirror-neuron model, or claim that these time constants identify biology. It is a controlled partially observed behavioral system attached to a real visual stream.

A positive result would require the sequencing architecture to improve history-dependent prediction and for a destructive control to remove a meaningful part of that gain. If the 12-frame window matches or beats it, the special sequencing circuit has not earned its extra complexity.

The frozen v1 receipts are deliberately not tuned away: on both the synthetic fixture and the uploaded webcam recording, simple present/window baselines still win endpoint prediction. Event-order shuffle and removal of tuft context do hurt the sequencing model, so those mechanisms carry signal, while route closure has not yet earned itself.

## Next stages, only if v0 has teeth

1. **Active ping:** the observer chooses a controlled visual perturbation that should distinguish competing hidden-state/rule hypotheses.
2. **GAx generations:** evolve a few fly parameters and ask whether the observer can infer/forecast hidden rule/operator drift across generations.
3. **Observer of observer:** a second system watches which probes the first observer chooses and what those probes reveal.

That is the larger loop:

```text
world -> organism -> behavior -> observer -> model
                                     |
                                     v
                              chosen perturbation
                                     |
                                     v
                         organism response -> update
```
