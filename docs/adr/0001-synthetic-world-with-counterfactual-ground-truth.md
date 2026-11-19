# ADR 0001: Validate attribution against a synthetic world with counterfactual ground truth

- Status: accepted
- Date: 2026-11-19

## Context

Attribution models are usually compared with each other ("Markov gives webinars 18%,
last-touch gives them 7%"). That tells you the models disagree, not which one is right.
On real CRM data the causal truth is unknown, so no model can be scored.

## Decision

Generate synthetic B2B journeys from an explicit data-generating process (DGP) in which
each channel has a configured per-touch lift on each funnel transition
(lead -> MQL -> SQL -> opportunity -> won). Define ground truth for channel *c* as the
expected number of wins that disappear when every *c* touch is deleted, estimated by
re-running the stage state machine with **common random numbers** (the same uniform draw
per event in the factual and counterfactual run) over `gt_replicates` replicates.
Models are scored on channel *shares* (MAE, Spearman) because models distribute observed
conversions while removal effects need not sum to total wins.

The DGP deliberately includes what breaks attribution in practice: an unobserved intent
confounder that drives both exposure and baseline progression, an ABM factor that
correlates LinkedIn/paid/events, late-journey channels, CRM stage-recording lag, missing
UTMs, anonymous web sessions, unlinked CRM contacts and multi-person buying committees.
A second config (`positive_control.yaml`) removes all of that to check the engine itself.

## Consequences

- Every claim in the README is a measured number from a seeded run.
- Results are only as relevant as the DGP: a model that wins here is not proven on real
  data. The README states this as the main limitation.
- Ground truth is a Monte-Carlo estimate; zero-lift channels can show tiny non-zero
  removal effects from simulation noise (path-wise effects are not monotone because a
  removed touch changes the stage at which later events land). Negative estimates are
  clipped to zero before computing shares.
