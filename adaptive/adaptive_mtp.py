"""Acceptance-driven 2..8 proposal controller, C1 experimental deployment.

Never changes verifier acceptance or sampling. Four completed groups provide
hysteresis; output truncation/stale results must be filtered by the caller.
This is a measured-acceptance heuristic, not a throughput-optimality claim.
"""
from collections import deque

class AdaptiveMTP:
    def __init__(self, minimum=2, maximum=8, window=4):
        if not (minimum == 2 and maximum == 8 and window >= 2):
            raise ValueError('Validated experimental range is 2..8')
        self.minimum, self.maximum = minimum, maximum
        self.k = minimum
        self.recent = deque(maxlen=window)
        self.groups = 0

    def observe(self, drafted, accepted):
        if not (isinstance(drafted, int) and isinstance(accepted, int)
                and 1 <= drafted <= self.maximum and 0 <= accepted <= drafted):
            raise ValueError('Invalid verification counters')
        self.groups += 1
        self.recent.append((drafted, accepted))
        old = self.k
        ratio = sum(a for d, a in self.recent) / sum(d for d, a in self.recent)
        if len(self.recent) == self.recent.maxlen:
            if ratio >= .80:
                self.k = min(self.maximum, self.k + 1)
            elif ratio < .50:
                self.k = max(self.minimum, self.k - 1)
            if self.k != old:
                self.recent.clear()
        return {'group': self.groups, 'drafted': drafted, 'accepted': accepted,
                'recent_acceptance': ratio, 'previous_k': old, 'next_k': self.k}
