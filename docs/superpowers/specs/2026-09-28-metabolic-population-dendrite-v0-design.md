# Smarfly2 — Metabolic Population Dendrite v0 Design

Date: 2026-09-28

## Purpose

Build the first neuron-like Smarfly2 observer whose **history changes the material that performs future computation**.

The central move is to stop treating memory as only another hidden vector. A population of temporal listeners carries three distinct state scales:

1. fast resonant/electrical state,
2. medium resource/metabolic state,
3. slow structural support/persistence.

Experience changes the resource and structural state of the listeners that participate in useful global events. The same later input can therefore be processed differently after different histories even when the instantaneous electrical state is matched.

The core requirement is:

> Same present electrical state + different resource/structural history can produce different future computation.

A second requirement is the inverse direction:

> After training, the population state should retain enough information that an outside reader can infer which history shaped it.

This is a computational architecture inspired by neuronal resonance, local dendritic events, somatic thresholding, three-factor plasticity, resource competition and structural persistence. The biological terms are **design analogies**, not claims that the simulator reproduces a specific cortical cell or metabolic pathway.

## Relationship to Smarfly2 v1

This design branches from `feature/sequencing-observer-v1` rather than from `main`.

SequencingObserver v1 established several useful constraints that remain in force:

- hidden simulator state must not leak into ordinary observer inputs,
- temporal order controls must be causal,
- history should be selectively written rather than blindly low-passing every visible coordinate,
- simple controls remain mandatory,
- negative results are preserved rather than tuned away.

However, Metabolic Population Dendrite v0 is a new subsystem rather than another feature vector for the existing SequencingObserver.

The ontology changes again:

```text
v0 resident:       present -> generic traces -> readout
v1 sequencing:     event -> selective residue -> continuation state -> readout
metabolic dendrite: input -> resonant population -> local events -> soma
                                      |                 |
                                      v                 v
                                  resources <------ modulated credit
                                      |
                                      v
                              structural persistence
                                      |
                                      +----> future measuring apparatus
```

The population itself becomes history-bearing computational material.

## Core scientific question

Can local temporal listeners, maintained under a resource budget and reinforced only when their activity participates in a useful somatic event, become a history-shaped observer that is measurably different from both:

- a fixed resonator bank with a trained linear readout, and
- a generic recurrent/history model?

The first build does **not** need to beat every baseline on predictive accuracy. It must first demonstrate a clean causal phenomenon:

```text
history A -> population A
history B -> population B
force/match same fast electrical present
same next input
=> different somatic response because medium/slow state differs
```

If that effect cannot be demonstrated under controlled conditions, later tuft/SST/anatomical layers are not justified.

## Scope

### In v0

- A deterministic synthetic temporal world with several recurring motifs.
- A bank/population of resonant temporal listeners.
- Fast complex resonant state per listener.
- Local nonlinear event threshold per listener.
- One shared soma with a hard threshold.
- Medium resource state per listener with maintenance and activity costs.
- A three-factor credit rule using local activity, soma event and external modulator/reward.
- Slow structural support state per listener driven by sustained resource balance.
- Structural weakening, quiescence and optional recycling after prolonged failure.
- A same-present/different-history causal test.
- A blind population readback test: infer which training history shaped the neuron from the final population state.
- Fixed-bank and destructive controls.
- Deterministic seeds and frozen receipts.
- Small diagnostic visualization only after the headless mechanism is verified.

### Explicitly not in v0

- No apical tuft channel yet.
- No SST/Martinotti route closure yet.
- No PV/basket rhythmic scheduler yet.
- No chandelier/AIS publication layer beyond the single soma threshold.
- No dendritic tree geometry or conduction delay yet.
- No multiple coupled neurons yet.
- No two-observer resonance experiment yet.
- No GAx generations.
- No webcam requirement for the core gate.
- No claim that listener death is literally equivalent to a neuronal activation function.
- No claim that simulated resource is ATP, transmitter, neurotrophin, calcium, or any single biological quantity.

Those layers become later projects only if this substrate earns them.

## Design principle: three clocks, three jobs

The architecture intentionally separates three timescales rather than implementing three interchangeable exponential traces.

### Fast: electrical/resonant state

Carries immediate temporal structure and can cross a local event threshold.

### Medium: resource state

Tracks whether a listener can afford its recent activity and whether its participation has recently been useful.

### Slow: structural support

Changes how strongly that listener contributes in the future and whether the listener remains part of the active measuring population.

The hierarchy is:

```text
fast activity     milliseconds / simulation ticks
resource balance  tens to hundreds of fast updates
structure         hundreds to thousands of fast updates
```

Exact simulation units are arbitrary in v0. What matters is strong separation of time scales and distinct causal roles.

## 1. Synthetic temporal world

The first world is deliberately simpler than the webcam bench so the causal questions are identifiable.

Generate a one-dimensional input stream from a small vocabulary of temporal motifs, for example:

- slow sinusoidal burst,
- fast sinusoidal burst,
- alternating pulse motif,
- aperiodic/noise burst,
- quiet/background interval.

Each episode contains several motifs. Only one motif is associated with positive external modulation/reward in a training condition.

The reward signal is not continuously available as an input feature to the listener bank. It arrives after or around the somatic consequence and acts only as a global modulator.

Two main training histories are required:

- `H_A`: reward motif A,
- `H_B`: reward motif B.

The physical input statistics outside the reward association should be matched as closely as practical so that final population differences cannot be explained merely by motif exposure counts.

## 2. Listener population

Use `N` temporal listeners, initially a small fixed number such as 32 or 64.

Each listener has inherited/fixed intrinsic parameters:

- resonance frequency `omega_i`,
- persistence/window parameter `r_i`,
- local threshold `theta_i`,
- initial structural support `s_i`,
- optional small phase offset.

The initial bank should tile the allowed temporal range rather than be optimized on a test set.

The primary fast state is complex:

\[
z_i(t+1)
= r_i e^{j\omega_i}z_i(t)
+ (1-r_i)\,b_i x_t.
\]

`b_i` is a fixed signed or positive input gain. v0 should avoid learned recurrent weights.

The measurable local response can be one of:

\[
a_i(t)=|z_i(t)|
\]

or a signed projection such as:

\[
a_i(t)=\Re(z_i(t)).
\]

The choice must be fixed in the implementation plan before scientific runs.

## 3. Local nonlinear dendritic event

A listener does not continuously contribute arbitrary precision values to the soma.

It produces a local event when its fast state crosses a threshold while structurally active:

\[
d_i(t)
= \mathbf{1}[a_i(t) > \theta_i]\,\mathbf{1}[s_i(t)>s_{min}].
\]

An optional refractory counter may prevent repeated events on every adjacent tick. If used, it is a fast electrical control and must not be conflated with metabolic recovery.

This local event is the first hard nonlinearity.

Control: a `continuous_listener` arm bypasses the threshold and sends normalized continuous activity to the soma. This tests whether the hard local event is doing anything useful rather than merely making the system harder to fit.

## 4. Shared soma

The soma integrates listener events weighted by structural support:

\[
V(t)=\sum_i s_i(t)\,d_i(t).
\]

The soma fires when:

\[
y(t)=\mathbf{1}[V(t)>\Theta].
\]

The soma is the shared consequence that makes listeners compete and cooperate within one organism-like unit.

No learned output layer is placed after the soma for the primary gate. A separate diagnostic decoder may be trained only for analysis/readback, never to drive the neuron.

The somatic threshold should be chosen before the frozen experiments, preferably from a simple occupancy target on an untrained calibration stream rather than optimized for classification accuracy.

## 5. Medium resource state

Each listener owns a resource variable:

\[
E_i(t)\in[E_{min},E_{max}].
\]

Resources change through three terms:

### Basal maintenance cost

\[
-c_{base}
\]

paid every update.

### Activity cost

\[
-c_{act}\,a_i(t)
\]

or a cost proportional to local events. The implementation plan must freeze which version is used before runs.

### Modulated credit

Useful participation earns resource:

\[
+\eta_E\,e_i(t)\,y(t)\,m(t),
\]

where:

- `e_i(t)` is an eligibility trace from recent local listener activity,
- `y(t)` is the somatic event,
- `m(t)` is the global modulator/reward.

Thus reward is not `reward -> listener strength` directly. Credit requires local eligibility and a shared somatic consequence.

Eligibility decays on a short-to-medium timescale so delayed modulation can still reinforce recently active listeners.

A negative or zero modulator provides no replenishment. v0 does not require explicit punishment beyond ongoing costs.

## 6. Slow structural support

Structural support changes much more slowly than fast state or resource:

\[
s_i(t+1)
= \mathrm{clip}\left(
  s_i(t)+\eta_s\,[E_i(t)-E_*],
  0,
  s_{max}
\right).
\]

Interpretation:

- sustained surplus slowly increases influence,
- sustained deficit slowly weakens influence,
- short bad intervals do not instantly delete a listener.

This makes structure an integrated record of usefulness under the training history.

The soma therefore changes over learning even if the intrinsic resonator bank itself is fixed:

\[
V_H(t)=\sum_i s_i(H)\,d_i(t).
\]

History has changed the measuring apparatus.

## 7. Structural quiescence and recycling

Literal death/replacement is optional in the first scientific gate because it can obscure the simpler continuous structural effect.

The default v0 behavior is:

1. listener support falls toward zero under prolonged deficit,
2. listeners below `s_quiet` cease contributing to the soma,
3. they remain inspectable in the state ledger.

A second `recycling` mode may be added after the core gates pass:

- if `s_i < s_prune` for `T_prune` consecutive updates, recycle that slot,
- new intrinsic tuning is sampled locally around a surviving listener or from the original prior,
- reset fast/resource state,
- assign small initial support.

Recycling must be tested separately because population turnover changes both memory capacity and search dynamics.

The first claim does **not** depend on replacement.

## 8. The central same-present / different-history experiment

Train two identical copies from the same initialization:

```text
Neuron A -> history H_A
Neuron B -> history H_B
```

After training, their resource and structural states differ.

Then construct the causal probe:

1. set/reset both fast resonant states `z_i` to the same agreed value (usually zero),
2. preserve their learned `E_i` and `s_i`,
3. present the exact same probe sequence,
4. record local events, soma potential and soma spikes.

Required observation:

\[
z^A(0)=z^B(0),\quad x^A=x^B,
\]

but

\[
(E^A,s^A)\neq(E^B,s^B)
\]

and the future somatic responses differ in a history-consistent way.

This is stronger than showing two hidden vectors differ. It demonstrates that medium/slow history modifies the later input-output operator.

A destructive control swaps or equalizes only the structural state while keeping the same fast probe. The response difference should disappear or reverse accordingly.

## 9. Blind population readback

After training, save only the population state, excluding the original training stream and labels.

Candidate readback features:

- `s_i`,
- `E_i`,
- intrinsic `omega_i` and `r_i`,
- quiescent/active status.

Do **not** include a stored history label.

Train or fit a deliberately small outside decoder to infer whether the neuron experienced `H_A` or `H_B`.

Primary purpose: test whether the final observer state contains recoverable history.

A stronger optional analysis estimates the rewarded motif frequency/profile from the support distribution over intrinsic listener frequencies.

Controls:

- randomized history labels,
- untrained populations,
- resource-only readback,
- structure-only readback.

The readback decoder is an analysis instrument. It never participates in neuron learning.

## 10. Baselines and destructive controls

The first frozen experiment must include at least these arms.

### Fixed resonator + linear readout

Same intrinsic listener bank, no resource or structure dynamics. Train a small linear/logistic output on the same training examples.

Question: does the metabolic population do anything besides implement a worse linear classifier?

### Metabolic bank without soma contingency

Reward eligible active listeners directly:

\[
\Delta E_i\propto e_i m(t)
\]

without multiplying by the soma event.

Question: is the global consequence/credit term necessary?

### Shuffled modulator

Causally shuffle reward timing while keeping reward count similar.

Question: does learning depend on temporal credit rather than long-run reward abundance?

### Frozen structure

Allow fast and resource states but hold `s_i` fixed.

Question: is the slow structural layer necessary for the same-present history effect?

### Zero-cost control

Remove maintenance/activity cost while preserving rewards.

Question: does resource competition matter, or does everything simply strengthen?

### Generic history baseline

A compact EMA or fixed-window model receives the same input history and predicts the rewarded motif/soma target.

Question: are the metabolic dynamics adding a distinctive causal mechanism rather than just extra memory capacity?

## 11. Success gates

### Gate 0 — deterministic mechanics

With a fixed seed and scripted input, repeated runs produce bit-identical or tolerance-identical state trajectories.

### Gate 1 — three timescales are causally distinct

Fast state reacts immediately; resource state changes more slowly; structure changes slowest. Freezing any one layer affects only its intended downstream role.

### Gate 2 — useful participation maintains structure

Under a training history, listeners eligible around rewarded soma events retain/increase support more than matched listeners that are active but not credited.

### Gate 3 — same-present / different-history operator effect

After matching/resetting fast electrical state and giving identical probes, neurons trained under `H_A` and `H_B` produce measurably different somatic responses attributable to retained medium/slow state.

This is the primary gate.

### Gate 4 — causal destruction

Equalizing/swapping structural state removes or swaps the history-conditioned response effect. Shuffled modulator timing substantially reduces learned selectivity.

### Gate 5 — blind history readback

A small held-out decoder can infer the shaping history from final population state above a randomized-label control.

### Gate 6 — comparison discipline

Report fixed-bank linear-readout and generic-history baselines beside the metabolic system. Do not call the new mechanism superior if it only reproduces their behavior less efficiently.

No numerical superiority threshold is frozen in this design because pilot scale may affect absolute score. The implementation plan must define pre-run effect/seed criteria before the final frozen experiment.

## 12. Data separation and leakage discipline

Use explicit types/records for:

- world input,
- global modulator,
- fast listener state,
- resource state,
- structural state,
- soma state,
- audit labels.

Training-history identity is audit metadata only. It must never enter neuron dynamics.

The same-present probe must reconstruct neurons from saved state rather than reusing a Python object with hidden episode metadata.

Readback datasets must split by independent run/seed, not by individual listener rows from the same trained population, to prevent trivial population-identity leakage.

## 13. Parameter discipline

Initial parameter ranges are chosen structurally, not tuned on the final frozen histories.

Examples to freeze in the implementation plan:

- population size `N`,
- listener frequency grid/range,
- persistence/window range,
- local threshold,
- soma threshold/calibration rule,
- eligibility decay,
- maintenance cost,
- activity cost,
- reward replenishment magnitude,
- resource target `E_*`,
- structural learning rate,
- quiet threshold.

The first frozen run should use several deterministic seeds with the exact same parameter set.

If a pilot reveals numerical instability, fixes must be justified as stability corrections and rerun from scratch; do not search the final benchmark for the prettiest outcome.

## 14. Diagnostics and visualization

Headless receipts come first.

After mechanics pass, add a compact visualization showing one neuron as a population rather than anthropomorphizing hidden states:

- listeners arranged by intrinsic frequency,
- instantaneous fast amplitude,
- resource level,
- structural support,
- local event flashes,
- soma potential/spike,
- current modulator pulse.

The useful visual story is that the same input bank gradually acquires a different **shape of support** under different histories.

No webcam UI is required for v0. A later version may reconnect this neuron to the live artificial-ethology world.

## 15. Error handling and numerical safety

- Clip resource and structural variables to declared ranges.
- Reject NaN/Inf immediately in tests and scripts.
- Keep complex resonator magnitude bounded through `r_i < 1`.
- Avoid structural update rates large enough to change support on the fast timescale.
- Ensure soma threshold does not trivially make the neuron always silent or always firing; calibration rule is frozen before training histories.
- Reward/modulator schedule must be generated independently of internal hidden state except through the explicitly defined training task.

## 16. Claim boundary

A successful v0 would support only this narrow claim:

> A population of temporal listeners with separate fast activity, medium resource and slow structural persistence can retain history in the composition of its measuring apparatus, such that identical later input can produce different somatic computation after different histories.

It would **not** show:

- that neurons literally perform Darwinian selection among subcellular resonators,
- that resource variables correspond one-to-one with ATP, transmitter or neurotrophins,
- that neuronal death/pruning is equivalent to ReLU,
- that the architecture explains cortical cognition,
- that the mechanism outperforms modern sequence models,
- that apical tuft/SST/PV/chandelier motifs have been implemented or validated.

The value of v0 is to create a concrete substrate on which those later questions can be tested one at a time.

## 17. Follow-on staircase

Only if Metabolic Population Dendrite v0 passes its causal gates:

1. **Arbor + delays** — pin listener populations to geometry so place becomes temporal coordinate.
2. **Apical susceptibility** — a second contextual stream modulates which basal listeners/branches can earn somatic influence.
3. **Rhythmic windows** — local inhibitory timing controls when continuation can advance.
4. **SST/Martinotti route suppression** — failed branches become selectively unavailable without erasing the whole state.
5. **Multiple metabolic neurons** — soma outputs become events for downstream populations.
6. **Two observer minds** — two history-shaped systems interact and test functional resonance without forced synchronization.

Each step gets its own design and falsifiers.

## 18. Expected repository shape

The implementation plan may refine file names, but the design expects a small isolated subsystem under the existing package, for example:

```text
src/smarfly2/metabolic/
    world.py
    population.py
    soma.py
    training.py
    readback.py

scripts/
    run_metabolic_dendrite.py
    probe_metabolic_history.py
    readback_metabolic_history.py

tests/
    test_metabolic_population.py
    test_metabolic_credit.py
    test_metabolic_history_probe.py
    test_metabolic_readback.py
```

The existing v0/v1 observers remain available as separate research branches/controls. Metabolic Population Dendrite should not silently replace their code paths.

## Decision summary

The first build is deliberately **not** a complete pyramidal neuron simulation.

It is the smallest system that makes the new idea falsifiable:

```text
temporal input
   -> resonant local activity
   -> local nonlinear events
   -> shared soma
   -> modulated resource credit
   -> slow structural persistence
   -> changed future observer
```

If this substrate fails, adding tuft, arbor geometry and inhibitory cell analogues would only decorate a failed core. If it works, those later anatomical motifs gain a concrete history-bearing material to control.
