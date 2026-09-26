"""Experimental C1 throughput-directed MTP 2..8 policy; not yet deployed.

Uses time between completed verifier groups, NOT backend publishable TG/s.
Drops mixed-size transition intervals; examines each K within one live request
before exploiting a measured winner. No verifier/sampling changes. Four steady
groups per K reduce noise; phase changes trigger a new bounded exploration.
"""
import math,time
from collections import deque

class ThroughputAdaptiveMTP:
    def __init__(self, minimum=2, maximum=8, window=4):
        if minimum!=2 or maximum!=8 or window<4:raise ValueError('Validated range 2..8, window>=4')
        self.minimum=minimum;self.maximum=maximum;self.k=2;self.window=window;self.groups=0
        self._last_time=None;self._last_drafted=None;self._samples=[];self._scores={};self._phase='explore';self._held=0
        self.recent=deque(maxlen=window)
    def observe(self,drafted,accepted,now=None):
        if not isinstance(drafted,int)or not isinstance(accepted,int)or not 1<=drafted<=8 or not 0<=accepted<=drafted:raise ValueError('Invalid counters')
        now=time.perf_counter()if now is None else now
        if not math.isfinite(now):raise ValueError('Invalid time')
        self.groups+=1;old=self.k;self.recent.append((drafted,accepted))
        dt=None if self._last_time is None else now-self._last_time
        steady=(self._last_drafted==drafted==self.k and dt is not None and 0<dt<=10)
        self._last_time=now;self._last_drafted=drafted
        rate=None
        if steady:
            self._samples.append((accepted+1,dt));self._held+=1
            if len(self._samples)>=self.window:
                rate=sum(n for n,t in self._samples)/sum(t for n,t in self._samples);self._samples.clear()
                if self._phase=='explore':
                    self._scores[self.k]=rate
                    if self.k<self.maximum:self.k+=1
                    else:
                        self.k=max(self._scores,key=lambda k:(self._scores[k],-k));self._phase='exploit';self._held=0
                else:
                    baseline=self._scores[self.k]
                    # Periodic recheck prevents committing forever to a K
                    # measured during a different output phase.
                    if rate<baseline*.75 or self._held>=48:
                        self._scores={};self._phase='explore';self.k=self.minimum;self._held=0
                    else:self._scores[self.k]=.75*baseline+.25*rate
        elif dt is not None and (dt<=0 or dt>10):
            self._samples.clear()  # stall, resumed work, or invalid clock
        return {'group':self.groups,'drafted':drafted,'accepted':accepted,
                'recent_acceptance':sum(a for d,a in self.recent)/sum(d for d,a in self.recent),
                'previous_k':old,'next_k':self.k,'policy':'throughput-experimental',
                'phase':self._phase,'steady_interval':steady,'group_interval_s':dt,
                'window_step_tps':rate,'scores':dict(self._scores)}
