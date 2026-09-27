# Smarfly2 v1 — Sequencing Observer Design

Date: 2026-09-27

## Purpose

Replace the v0 `ResidentObserver` as the primary history-sensitive model with a compact **SequencingObserver** whose present computation is altered by selectively deposited history.

The design is inspired by the functional roles we have been using as computational analogies for pyramidal/basal drive, apical tuft context, SST/Martinotti route closure, rhythmic inhibition, and AIS/chandelier publication. These names are architectural metaphors, not claims that the software identifies the biological circuit.

The key requirement is:

> Same visible present + different relevant history can select a different continuation operator, while irrelevant absolute coordinates remain ordinary present information rather than being blindly integrated into memory.

This version remains an artificial-ethology observer bench. The fly is still simple; the observer mind is the research object.

## Why v1 exists

The v0 resident model applies generic fast/slow traces to the entire visible vector. That makes history representation depend strongly on arbitrary coordinates such as absolute position, heading, time, and `dt`.

v1 changes the ontology of memory:

```text
v0: every coordinate -> generic low-pass traces
v1: event/innovation -> selective write -> temporal sequence state -> context-dependent continuation
```

The goal is not "more memory." The goal is **history deposited into the right computational coordinates**.

## Required prerequisite fix

Before comparing observers, trajectory targets must respect the fly world's toroidal geometry. `dx` and `dy` are shortest wrapped displacements across the known frame width/height, just as heading already uses a wrapped angular difference.

This fix belongs to the shared evaluation path, not specifically to SequencingObserver.

## Scope

### In v1

- Keep present-only and fixed-window baselines unchanged as controls.
- Add event extraction from visible sensory/behavioral innovations.
- Add a phase/window scheduler that advances internal sequence computation in local temporal windows.
- Add a small population of competing continuation channels.
- Add selective event-to-residue writes rather than integrating all inputs.
- Add a slow contextual/apical state that modulates continuation-channel susceptibility.
- Add route-specific SST-style veto/closure after prediction conflict.
- Add a publication gate separating silent internal continuation from emitted prediction.
- Add destructive controls for event order, tuft context, route closure, and publication.
- Evaluate on the same held-out frames as existing observers.
- Preserve the current live webcam/replay bench.

### Not in v1

- No GAx generations yet.
- No second interacting observer yet.
- No active visual ping chosen by the observer yet.
- No learned deep vision backbone.
- No spiking-neuron simulation.
- No claim that the functional modules are literal biological implementations.

Two-observer resonance, active ping, and GAx become later stages only if one SequencingObserver first demonstrates useful history-dependent computation.

## Architecture

```text
VISIBLE PRESENT
  world features + fly behavior
          |
          +-------------------------------> present readout features
          |
          v
   EVENT / INNOVATION EXTRACTOR
          |
          v
   selective memory writes
          |
          +------------------+
          |                  |
          v                  v
   fast event residue    slow contextual residue
          |                  |
          +--------+---------+
                   v
             TUFT CONTEXT
       channel-specific gain
                   |
                   v
CURRENT EVENT -> COMPETING CONTINUATION CHANNELS
                   |
             rhythmic windows
          step 0 -> 1 -> 2 -> ...
                   |
          prediction mismatch
                   |
                   v
          SST ROUTE CLOSURE
       suppress failed continuation
                   |
                   v
           AIS PUBLICATION GATE
                   |
                   v
        short future trajectory
```

The present path and memory path are intentionally distinct. Absolute `(x, y)`, current heading, velocity, and current visual evidence may be used by the readout as present information. Only selected temporal innovations are written to resident memory.

## 1. Event / innovation extractor

Input is the same visible-only stream available to every observer.

The extractor produces a small continuous event vector from changes rather than semantic labels. Initial channels:

- change in brightness asymmetry,
- change in motion energy,
- change in motion asymmetry,
- change in contrast,
- signed turn innovation from visible heading/velocity,
- acceleration / speed innovation.

No hidden fly variables are used.

Events are normalized using statistics fitted only on the training period.

The implementation may retain signed continuous values. Hard thresholded event symbols are not required for v1.

## 2. Selective resident writes

Maintain per-event-channel fast and slow traces:

\[
f_{t+1}=f_t+\alpha_f(e_t-f_t),
\qquad
s_{t+1}=s_t+\alpha_s(e_t-s_t).
\]

The primary dynamic coordinates are:

\[
r_t^{change}=f_t-s_t,
\qquad
r_t^{context}=s_t.
\]

Only the event vector is written here. Raw absolute position, clock time, and `dt` are excluded from these residues.

Use the same order-of-magnitude time scales already explored in the project as fixed defaults rather than tuning them on the uploaded session:

- fast tau: about `0.30 s`,
- slow tau: about `3.0 s`.

## 3. Rhythmic computation windows

The observer has a cyclic local phase with default period about `0.8 s`.

Phase is not merely appended to a ridge feature vector. It controls **which continuation transition is allowed to update during a frame**.

Divide the cycle into a small number of windows, initially four. Each window advances one stage of internal continuation state; outside that window the corresponding transition is held.

This is the computational analogue of the earlier result where dead time/rhythmic inhibition manufactured ordered traversal from otherwise less directional dynamics.

No claim is made that the chosen period is a biological rhythm.

## 4. Competing continuation channels

Use a small fixed number of continuation channels, initially four.

Each channel maintains a low-dimensional state representing a candidate near-future motion continuation. Channels are not semantic classes such as `avoid` or `approach`; their meanings are learned/fitted only through the output readout and their response to events.

A channel update has the conceptual form:

\[
z_{k,t+1}
= g_{k,t}\,[A_k(\phi_t)z_{k,t}+B_ke_t]\,[1+\beta a_{k,t}],
\]

where:

- `e_t` is current event innovation,
- `A_k(phi_t)` advances only in its permitted rhythmic window,
- `a_{k,t}` is tuft/context susceptibility,
- `g_{k,t}` is the route-open / route-closed factor.

v1 should keep these states small and interpretable. No recurrent neural-network package is needed.

## 5. Tuft-style contextual susceptibility

The slow residue changes the relative gain of continuation channels.

This is the key same-present/different-history mechanism:

\[
a_t = T\,r_t^{context}
\]

and channel gain is a bounded function of `a_t`.

The tuft path does not itself emit the trajectory prediction. It changes which continuation dynamics are easy to recruit.

Control: `no_tuft` sets this contextual modulation to neutral while preserving the rest of the observer.

## 6. SST/Martinotti-style route closure

After a channel has made an internal continuation prediction, compare it with the next visible event when that evidence becomes available.

Large channel-specific mismatch increases a temporary inhibitory/closure state for that channel. Closure decays over time.

The important behavior is:

```text
failed continuation -> that route becomes harder to continue
```

not:

```text
prediction error -> erase all observer memory
```

Control: `no_route_closure` keeps all channels open while leaving event memory and tuft context intact.

## 7. AIS/publication gate

Internal channel state can continue evolving while publication is blocked.

The publication gate computes a confidence/margin from surviving channel states and decides whether the observer emits its internal trajectory estimate to the external prediction interface.

For metric compatibility, evaluation still needs a prediction on every held-out frame. Therefore v1 exposes both:

- `internal_prediction`: always available for scoring and analysis,
- `published_prediction`: emitted only when publication gate is open; otherwise marked silent.

This prevents us from confusing "the model computed a state" with "the model chose to publish that state."

Control: `publication_block` forces external silence while asserting that internal state evolution remains unchanged.

## 8. Readout

The readout receives:

- current visible present features,
- continuation-channel states,
- selective fast-minus-slow event residue,
- slow context residue,
- route-open/closure state,
- current phase encoding.

It predicts the shared 8-frame target `(dx, dy, dheading)`.

Use the existing NumPy ridge readout initially. The interesting computation must live in the temporal state construction, not a powerful decoder.

## 9. Training discipline

All normalizers and readout weights are fitted only on the contiguous training period.

State may be rolled forward through the entire chronological sequence because that is how an online observer behaves, but no future sample may alter an earlier state and no target crossing the train/test boundary may be used for training.

No parameter search is allowed on the uploaded session for the initial v1 implementation. Defaults come from the design; the real session is an evaluation case, not a tuning set.

## 10. Controls

The v1 evaluation table includes:

- `present`,
- `window`,
- legacy `resident`,
- `sequencing`,
- `sequencing_event_shuffle`,
- `sequencing_no_tuft`,
- `sequencing_no_route_closure`,
- `sequencing_publication_block` for internal-state invariance rather than endpoint superiority.

`sequencing_event_shuffle` preserves each present sample and target but destroys the temporal ordering of past event writes.

The purpose of these controls is to identify which part of the sequencing architecture carries any gain.

## 11. Matched-present analysis

Keep the existing visible-only matched-present search.

For selected pairs with close present state and divergent futures, report:

- true future,
- present prediction,
- fixed-window prediction,
- sequencing prediction,
- event residue difference,
- tuft susceptibility difference,
- active continuation-channel distribution,
- hidden fly-state difference revealed only after scoring.

This should become the most human-readable demonstration of the observer mind carrying useful history.

## 12. Live visualization

Add an optional SequencingObserver inspector to the current live/replay GUI.

Show only compact state:

- four continuation-channel activities,
- local phase/window,
- tuft/context gain profile,
- route closures,
- publication open/silent,
- short internal and published trajectory traces.

Do not turn the UI into a neuroscience diagram. The visual purpose is to make the temporal computation legible while the fly moves.

## 13. Testing

### Shared geometry tests

- wrapped `dx` across left/right boundaries,
- wrapped `dy` across top/bottom boundaries,
- non-boundary displacement remains unchanged.

### Event tests

- constant visible stream produces near-zero innovation after initialization,
- controlled left/right or motion changes produce signed event differences,
- absolute position changes alone are not written into sensory event residues unless they alter defined motion/turn innovation.

### Sequence-window tests

- only the permitted transition/window advances at a given phase,
- removing dead intervals or flattening phase removes that ordered-update property,
- identical current event with different phase can produce different continuation evolution when internal state differs.

### Tuft tests

- same present/event with different slow histories changes channel susceptibility,
- `no_tuft` removes that difference without resetting other memory.

### Route-closure tests

- prediction conflict closes only the offending route,
- unrelated channel state remains intact,
- closure decays and the route can reopen.

### Publication tests

- publication block produces no external emission,
- internal state under publication block is bit-identical to an otherwise identical run with publication enabled.

### Leakage/control tests

- hidden fly state never enters SequencingObserver input,
- destructive controls preserve current sample and target indices,
- all observer arms score exactly the same held-out frames.

## 14. Evaluation and claim boundary

v1 is interesting only if the sequencing architecture earns something beyond architectural decoration.

Primary descriptive comparisons:

- endpoint RMSE,
- angular MAE,
- matched-present prediction quality,
- effect of event-order shuffle,
- effect of removing tuft context,
- effect of removing route closure.

A useful positive pattern would be:

```text
sequencing improves over present-only on history-dependent episodes
and
at least one destructive control removes a meaningful part of that gain.
```

But no pass threshold is chosen after seeing the result.

If the fixed 12-frame window remains as good or better, the special circuit has not earned its added complexity.

## 15. Later stage: two observers

Do not implement this in the first v1 branch.

Once one SequencingObserver works, instantiate two copies with the same inherited architecture but different histories. Allow each to observe the other's **published** predictions/probes as ordinary events.

Do not force phase synchronization or hidden-state matching.

The later resonance question is functional:

> Does ordered interaction make corresponding continuation structures become mutually predictive/recruitable, and does scrambling the partner's event order destroy that effect?

That is a separate spec after single-observer v1 has earned continuation.

## Success criterion for this implementation

The implementation is complete when:

1. toroidal trajectory targets are correct,
2. SequencingObserver contains event-selective writes, phase-gated continuation, tuft modulation, route closure, and publication separation,
3. every mechanism has a destructive control/test,
4. all observers use identical held-out samples,
5. synthetic and recorded-session evaluation run without hidden-state leakage,
6. results are reported descriptively without tuning the real session or declaring a biological proof.

The scientific question remains open until the measured comparisons are run.