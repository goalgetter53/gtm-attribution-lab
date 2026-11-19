# ADR 0003: Attribution model design choices (Markov, Shapley, horizon matching)

- Status: accepted
- Date: 2026-11-19

## Context

Data-driven attribution has several variants that can give very different answers.
The lab needs choices that are standard, explainable and testable.

## Decisions

1. **Markov: canonical removal effect.** Removing a channel redirects every state that
   contains it to `(null)` (Anderl et al.). Higher-order chains use the last *k* channels
   as the state. Absorption probabilities are solved exactly with `numpy.linalg.solve`.
   Removal effects are provably in [0, 1]; credits are proportional to them.
2. **Shapley: two characteristic functions.**
   - `shapley_empirical`: v(S) = smoothed conversion rate of journeys whose channel *set*
     is exactly S (Beta prior centred on journeys with the same number of channels).
     This is what most "data-driven Shapley" tools do.
   - `shapley_model`: v_j(S) = predicted conversion probability of journey *j* keeping only
     the touches of channels in S, from a ridge logistic regression on per-channel touch
     **counts** (a noisy-OR response is log-linear in counts), fit by IRLS in numpy.
   Exact Shapley (bitmask enumeration) is used up to `shapley_exact_max_channels`
   players; above that, permutation-sampling Monte Carlo with standard errors. A
   convergence test checks the MC error shrinks at ~1/sqrt(m).
   Per-journey values are clipped at 0 and normalised, so credits sum to conversions.
3. **Horizon matching for data-driven models.** Converted journeys stop at the win while
   non-converting journeys run to the data cutoff, so naive comparisons punish channels
   that occur late. Markov and Shapley therefore see non-converting journeys truncated at
   a horizon sampled from the converters' lead-to-win durations (risk-set matching).
   Heuristic models only use converting journeys and are unaffected.
4. **'direct' is a channel for the models but not for scoring.** Untracked traffic is
   kept so models can't silently re-assign it, then excluded (and reported) when shares
   are compared with ground truth.

## Consequences

- Each choice is a parameter or a separate model name, so alternatives can be benchmarked
  side by side with `gal validate`.
- The removal effect is known to over-reward high-coverage channels; the lab measures
  that rather than hiding it.
