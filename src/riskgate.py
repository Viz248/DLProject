"""
RiskGate-CUSUM  -  forecast-gated sequential change detection   (project novelty)

Idea
----
Two kinds of evidence about an upcoming degradation of a deployed classifier
exist, and the existing detectors use only one of them each:

  * LABEL-FREE, EARLY, NOISY   : a sequence forecaster (LSTM/GRU) reads the
    classifier's internal signals (confidence, entropy) and outputs a risk
    score r_t.  It is available before any label arrives, but on its own it
    fires too often (see Table D of the report).
  * LABEL-BASED, LATE, RELIABLE: a sequential test (CUSUM) on the error stream.
    Statistically well-founded, but it needs errors to accumulate first.

RiskGate-CUSUM fuses them *inside* the sequential test instead of voting
between two separate alarms.  The forecaster's risk, converted to a scale-free
causal percentile rank u_t in [0, 1], does two things to a one-sided CUSUM:

    S_t = max(0, S_{t-1} + (e_t - mu_t - k) + gamma * max(0, u_t - u0))   # (1) evidence injection
        (cold start: S stays 0 until `min_ref` past errors exist, otherwise the
         reference rate mu_t starts at 0 and the statistic climbs spuriously)
    alarm when  S_t > h_t ,   h_t = h0 * (1 - beta * u_t)                 # (2) threshold gating

  e_t   error indicator (1 if the classifier was wrong)
  mu_t  causal reference error rate (mean of the last `ref_len` errors)
  k     allowance (ignores small, normal fluctuations)
  u_t   percentile rank of r_t among the forecaster's own previous outputs
  (1) risk ABOVE the priming level u0 pushes the statistic up even before
      errors accumulate  -> earlier alarms
  (2) high risk LOWERS the threshold, low risk keeps it at h0 -> the detector
      is sensitive only when the forecaster says trouble is likely, so the
      false-alarm load stays near that of the plain test.

Special cases (all covered by tests):
    beta = gamma = 0   ->  plain one-sided CUSUM on the error stream
    gamma = 0          ->  pure threshold gating
    beta = 0           ->  pure evidence injection

Everything is strictly causal: the decision at time t uses e_0..e_t and
r_0..r_t only.  The class has an online API (`update`) and a batch helper
(`run`).

NOTE ON NOVELTY: CUSUM, forecasting of drift and label-free confidence
detectors are all known.  What is proposed here is this specific coupling
(rank-normalised neural risk driving both the evidence term and the threshold
of the error CUSUM).  A literature check is still the authors' job before
claiming it is new.
"""
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class RiskGateParams:
    h0: float = 4.0          # base alarm threshold
    k: float = 0.05          # CUSUM allowance
    beta: float = 0.5        # threshold gating strength   (0 = off), in [0, 1)
    gamma: float = 0.02      # evidence-injection strength (0 = off)
    u0: float = 0.8          # risk-rank level above which evidence is injected
    ref_len: int = 500       # look-back of the causal reference error rate
    min_ref: int = 100       # cold start: no accumulation until this many past errors exist
    rank_history: int = 2000 # window of past risks used for the percentile rank
    rank_min_history: int = 300  # before this many risks, u_t = 0 (no priming)
    refractory: int = 20     # minimum steps between two alarms

    def validate(self):
        assert self.h0 > 0, "h0 must be positive"
        assert 0.0 <= self.beta < 1.0, "beta must be in [0, 1)"
        assert self.gamma >= 0.0 and self.k >= 0.0
        assert 0.0 <= self.u0 <= 1.0
        assert self.ref_len >= 1 and self.rank_history >= 2
        assert 1 <= self.min_ref <= self.ref_len, "min_ref must be in [1, ref_len]"
        assert self.rank_min_history >= 1 and self.refractory >= 1
        return self

    def to_dict(self):
        return asdict(self)


class RiskRank:
    """Causal percentile rank of a value among the previous `history` values.

    rank(x) = fraction of stored past values strictly below x (ties count half),
    computed BEFORE x is added, so x never ranks against itself.
    """

    def __init__(self, history: int = 2000, min_history: int = 300):
        self.history = history
        self.min_history = min_history
        self._fifo = deque()
        self._sorted: List[float] = []

    def update(self, x: float) -> float:
        n = len(self._sorted)
        if n < self.min_history:
            u = 0.0
        else:
            lo = bisect_left(self._sorted, x)
            hi = lo
            while hi < n and self._sorted[hi] == x:
                hi += 1
            u = (lo + 0.5 * (hi - lo)) / n
        # add x, then evict the oldest value
        self._fifo.append(x)
        insort(self._sorted, x)
        if len(self._fifo) > self.history:
            old = self._fifo.popleft()
            del self._sorted[bisect_left(self._sorted, old)]
        return float(u)

    def warm(self, values: Iterable[float]):
        """Feed past risks (e.g. validation outputs) without using the ranks."""
        for v in values:
            self.update(float(v))


class RiskGateCUSUM:
    """Online forecast-gated CUSUM.  Call update(error, risk) once per step."""

    def __init__(self, params: Optional[RiskGateParams] = None,
                 initial_errors: Optional[Iterable[float]] = None,
                 initial_risks: Optional[Iterable[float]] = None):
        self.p = (params or RiskGateParams()).validate()
        self._rank = RiskRank(self.p.rank_history, self.p.rank_min_history)
        if initial_risks is not None:
            self._rank.warm(initial_risks)
        self._errs = deque(maxlen=self.p.ref_len)
        self._err_sum = 0.0
        if initial_errors is not None:
            for e in list(initial_errors)[-self.p.ref_len:]:
                self._push_err(float(e))
        self.S = 0.0
        self.t = -1
        self._last_alarm = -10 ** 9
        self.last_u = 0.0
        self.last_threshold = self.p.h0

    def _push_err(self, e: float):
        if len(self._errs) == self._errs.maxlen:
            self._err_sum -= self._errs[0]
        self._errs.append(e)
        self._err_sum += e

    def reference_rate(self) -> float:
        return self._err_sum / len(self._errs) if self._errs else 0.0

    def update(self, error: float, risk: float) -> bool:
        """Process one step; returns True if an alarm fires at this step."""
        p = self.p
        self.t += 1
        u = self._rank.update(float(risk))          # uses risks BEFORE this one
        mu = self.reference_rate()                  # uses errors BEFORE this one
        warm = len(self._errs) >= p.min_ref          # reference rate is trustworthy
        inc = (float(error) - mu - p.k) + p.gamma * max(0.0, u - p.u0)
        self.S = max(0.0, self.S + inc) if warm else 0.0
        h = p.h0 * (1.0 - p.beta * u)
        self.last_u, self.last_threshold = u, h
        self._push_err(float(error))                # now add this step's error
        if self.S > h and (self.t - self._last_alarm) >= p.refractory:
            self._last_alarm = self.t
            self.S = 0.0
            return True
        return False

    def run(self, errors: np.ndarray, risks: np.ndarray) -> np.ndarray:
        """Batch helper: returns the step indices (0-based) of the alarms."""
        errors = np.asarray(errors, dtype=np.float64)
        risks = np.asarray(risks, dtype=np.float64)
        assert len(errors) == len(risks), "errors and risks must align"
        return np.array([i for i in range(len(errors))
                         if self.update(errors[i], risks[i])], dtype=np.int64)


def plain_cusum(errors: np.ndarray, h0: float, k: float, ref_len: int = 500,
                refractory: int = 20, initial_errors=None) -> np.ndarray:
    """Reference one-sided CUSUM (no risk input) = RiskGate with beta=gamma=0."""
    det = RiskGateCUSUM(RiskGateParams(h0=h0, k=k, beta=0.0, gamma=0.0,
                                       ref_len=ref_len, refractory=refractory),
                        initial_errors=initial_errors)
    return det.run(errors, np.zeros(len(errors)))
