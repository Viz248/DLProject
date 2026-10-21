"""
Event-level "early warning" evaluation.

The research question is not only "can a forecaster rank windows by risk"
(that is Table B) but "does it warn BEFORE the error-based detectors react".
ADWIN/DDM only ever see the error stream, so on the real Elec2 stream there
is no ground-truth change point.  Instead we use *degradation events*
derived from the same label that the forecasters are trained on:

    onset  = first window centre t of a run of positive labels
             (the error rate over [t, t+horizon) exceeds the past rate by delta)

Every method is then reduced to a list of alarm times and scored identically:

    hit      : an alarm in [onset - lead_max, onset + tol]
    offset   : alarm_time - onset   (negative = warned BEFORE the onset)
    early    : a hit whose first alarm is at or before the onset
    FA/1000  : alarms that match no event, per 1000 stream steps

Forecasters fire on rising edges of P(degradation) >= threshold, ADWIN/DDM on
their drift flags, and a Monte-Carlo "random alarms" reference with the same
number of alarms shows how much of each hit rate is chance.
"""
from typing import Dict, Iterable, List, Optional

import numpy as np


# ------------------------------------------------------------------ #
#  Alarm / event extraction                                            #
# ------------------------------------------------------------------ #
def alarms_from_probs(probs: np.ndarray, centers: np.ndarray,
                      threshold: float, refractory: int = 20) -> np.ndarray:
    """Stream indices where P(degradation) >= `threshold`, keeping at most one
    alarm per `refractory` steps.  (Alarm count is monotone in the threshold,
    so an alarm budget can be matched by sweeping it.)"""
    return _apply_refractory(centers[probs >= threshold], refractory)


def alarms_from_flags(flags: np.ndarray, offset: int = 0,
                      refractory: int = 20) -> np.ndarray:
    """Stream indices of detector alarms (flags is a boolean array)."""
    return _apply_refractory(np.flatnonzero(flags) + offset, refractory)


def _apply_refractory(times: np.ndarray, refractory: int) -> np.ndarray:
    kept: List[int] = []
    last = -10 ** 9
    for t in np.sort(times):
        if t - last >= refractory:
            kept.append(int(t))
            last = t
    return np.array(kept, dtype=np.int64)


def degradation_onsets(labels: np.ndarray, centers: np.ndarray,
                       merge_gap: int = 20) -> np.ndarray:
    """Onset of each degradation episode: the first centre of a run of
    positive labels; runs closer than `merge_gap` steps are merged."""
    pos = centers[labels == 1]
    if len(pos) == 0:
        return np.array([], dtype=np.int64)
    onsets = [int(pos[0])]
    prev = int(pos[0])
    for t in pos[1:]:
        if t - prev > merge_gap:
            onsets.append(int(t))
        prev = int(t)
    return np.array(onsets, dtype=np.int64)


def alarms_from_probs_adaptive(probs: np.ndarray, centers: np.ndarray,
                               quantile: float = 0.95, history: int = 2000,
                               min_history: int = 300,
                               refractory: int = 20) -> np.ndarray:
    """
    Scale-free alarm rule: alarm when P(degradation) exceeds the `quantile`
    of the forecaster's OWN previous `history` outputs (causal - only past
    values are used).  A fixed probability threshold tuned on validation is
    brittle here because the probability scale shifts between periods (the
    GRU's outputs on test are compressed relative to validation); a relative
    rule gives every forecaster a comparable, small alarm budget (~1-q of the
    time, thinned by `refractory`).
    """
    probs = np.asarray(probs, dtype=np.float64)
    fire = np.zeros(len(probs), dtype=bool)
    for t in range(min_history, len(probs)):
        past = probs[max(0, t - history):t]
        fire[t] = probs[t] >= np.quantile(past, quantile)
    return _apply_refractory(centers[fire], refractory)


# ------------------------------------------------------------------ #
#  Scoring                                                             #
# ------------------------------------------------------------------ #
def evaluate_alarms(onsets: np.ndarray, alarms: np.ndarray,
                    span: int, lead_max: int = 50, tol: int = 100) -> Dict:
    """Score one alarm list against the degradation onsets."""
    n_ev, n_al = len(onsets), len(alarms)
    offsets: List[int] = []
    matched = np.zeros(n_al, dtype=bool)

    for o in onsets:
        in_win = np.flatnonzero((alarms >= o - lead_max) & (alarms <= o + tol))
        if len(in_win):
            first = alarms[in_win[0]]
            offsets.append(int(first - o))
            matched[in_win] = True

    n_hit = len(offsets)
    n_early = sum(1 for d in offsets if d <= 0)
    n_false = int((~matched).sum())
    return {
        "n_events":        int(n_ev),
        "n_alarms":        int(n_al),
        "n_hits":          int(n_hit),
        "event_recall":    float(n_hit / n_ev) if n_ev else float("nan"),
        "early_rate":      float(n_early / n_ev) if n_ev else float("nan"),
        "median_offset":   float(np.median(offsets)) if offsets else None,
        "alarm_precision": float(matched.sum() / n_al) if n_al else float("nan"),
        "false_alarms":    n_false,
        "false_alarms_per_1000": float(1000.0 * n_false / span) if span else float("nan"),
    }


def chance_reference(onsets: np.ndarray, n_alarms: int, start: int, end: int,
                     lead_max: int = 50, tol: int = 100,
                     n_draws: int = 300, seed: int = 0) -> Dict:
    """Expected event_recall / alarm_precision if `n_alarms` alarms were placed
    uniformly at random in [start, end) - the "is this better than luck" line."""
    rng = np.random.default_rng(seed)
    rec, prec = [], []
    for _ in range(n_draws):
        al = np.sort(rng.integers(start, end, size=n_alarms))
        r = evaluate_alarms(onsets, al, end - start, lead_max, tol)
        rec.append(r["event_recall"])
        prec.append(r["alarm_precision"])
    prec = [v for v in prec if not np.isnan(v)]     # no alarms -> undefined
    return {"chance_event_recall": float(np.mean(rec)),
            "chance_alarm_precision": float(np.mean(prec)) if prec else float("nan")}


def compare_methods(onsets: np.ndarray, alarm_sets: Dict[str, np.ndarray],
                    start: int, end: int,
                    lead_max: int = 50, tol: int = 100) -> Dict[str, Dict]:
    """Evaluate every method (+ its chance reference) on the same events."""
    out = {}
    for name, alarms in alarm_sets.items():
        alarms = alarms[(alarms >= start) & (alarms < end)]
        res = evaluate_alarms(onsets, alarms, end - start, lead_max, tol)
        res.update(chance_reference(onsets, len(alarms), start, end,
                                    lead_max, tol))
        out[name] = res
    return out
