"""Markov-chain attribution with the removal effect (Anderl et al., 2016).

Journeys become paths ``(start) -> s1 -> ... -> sk -> (conversion | null)`` where a state
is the tuple of the last ``order`` channels (order 1 = classic first-order chain).
The probability of reaching ``(conversion)`` from ``(start)`` is the absorbing-chain
solution ``x = (I - Q)^-1 r``. The *removal effect* of channel c is the relative drop in
that probability when every state containing c is redirected to ``(null)``:

    RE(c) = 1 - P(conv | c removed) / P(conv)

Removing states can only delete paths to conversion, so RE(c) is always in [0, 1].
Credits split the attributable conversions in proportion to the removal effects.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from gal.attribution.base import Credits, Journey, total_attributable

State = tuple[str, ...]


@dataclass(frozen=True)
class MarkovChain:
    order: int
    states: tuple[State, ...]  # transient states; index 0 is the (start) state = ()
    transition: npt.NDArray[np.float64]  # (m, m) transient -> transient probabilities
    to_conversion: npt.NDArray[np.float64]  # (m,) transient -> conversion probabilities

    @property
    def channels(self) -> list[str]:
        return sorted({c for s in self.states for c in s})

    def conversion_probability(self, removed: str | None = None) -> float:
        Q = self.transition.copy()
        r = self.to_conversion.copy()
        if removed is not None:
            dead = [i for i, s in enumerate(self.states) if removed in s]
            Q[dead, :] = 0.0
            r[dead] = 0.0
        m = Q.shape[0]
        x = np.linalg.solve(np.eye(m) - Q, r)
        return float(x[0])


def state_sequence(channels: tuple[str, ...], order: int) -> list[State]:
    return [tuple(channels[max(0, i - order + 1) : i + 1]) for i in range(len(channels))]


def fit_chain(journeys: Iterable[Journey], order: int = 1) -> MarkovChain:
    """Estimate transition probabilities by maximum likelihood (row-normalised counts)."""
    if order < 1:
        raise ValueError("order must be >= 1")
    paths = [(state_sequence(j.channels, order), j.converted) for j in journeys if not j.is_empty]
    states: list[State] = [()]
    index: dict[State, int] = {(): 0}
    for seq, _ in paths:
        for s in seq:
            if s not in index:
                index[s] = len(states)
                states.append(s)
    m = len(states)
    counts = np.zeros((m, m))
    conv = np.zeros(m)
    null = np.zeros(m)
    for seq, converted in paths:
        prev = 0
        for s in seq:
            cur = index[s]
            counts[prev, cur] += 1
            prev = cur
        if converted:
            conv[prev] += 1
        else:
            null[prev] += 1
    out = counts.sum(axis=1) + conv + null
    safe = np.where(out > 0, out, 1.0)
    return MarkovChain(
        order=order,
        states=tuple(states),
        transition=counts / safe[:, None],
        to_conversion=conv / safe,
    )


def removal_effects(journeys: Iterable[Journey], order: int = 1) -> dict[str, float]:
    chain = fit_chain(list(journeys), order)
    base = chain.conversion_probability()
    if base <= 0:
        return dict.fromkeys(chain.channels, 0.0)
    effects = {}
    for c in chain.channels:
        # Clip float noise: removal can never raise the conversion probability.
        effects[c] = min(max(1.0 - chain.conversion_probability(removed=c) / base, 0.0), 1.0)
    return effects


def markov_attribution(journeys: Iterable[Journey], order: int = 1) -> Credits:
    js = list(journeys)
    effects = removal_effects(js, order)
    total_re = sum(effects.values())
    n_conv = total_attributable(js)
    if total_re <= 0 or n_conv == 0:
        return dict.fromkeys(effects, 0.0)
    return {c: n_conv * e / total_re for c, e in effects.items()}
