"""Isolated adaptive2..8 candidate, NOT deployed or a measured speed claim.

Preserves verification/sampling and the initial four-steady-group-per-K search.
Unlike the baseline, one noisy low window cannot reset the complete search.
Rates are controller interval estimates, never publishable per-chat TG/s.
"""
import math,time
from collections import deque

class ThroughputAdaptiveMTP:
    def __init__(self,minimum=2,maximum=8,window=4):
        if minimum!=2 or maximum!=8 or window<4:raise ValueError('Validated range 2..8, window>=4')
        self.minimum=minimum;self.maximum=maximum;self.k=2;self.window=window;self.groups=0
        self._last_time=None;self._last_drafted=None;self._samples=[];self._scores={};self._phase='explore';self._held=0;self._bad_windows=0
        self.recent=deque(maxlen=window)
    def observe(self,drafted,accepted,now=None):
        if not isinstance(drafted,int)or not isinstance(accepted,int)or not 1<=drafted<=8 or not 0<=accepted<=drafted:raise ValueError('Invalid counters')
        now=time.perf_counter()if now is None else now
        if not math.isfinite(now):raise ValueError('Invalid time')
        self.groups+=1;old=self.k;self.recent.append((drafted,accepted))
        dt=None if self._last_time is None else now-self._last_time
        steady=self._last_drafted==drafted==self.k and dt is not None and 0<dt<=10
        self._last_time=now;self._last_drafted=drafted;rate=None;reset_reason=None
        if steady:
            self._samples.append((accepted+1,dt));self._held+=1
            if len(self._samples)>=self.window:
                rate=sum(n for n,t in self._samples)/sum(t for n,t in self._samples);self._samples.clear()
                if self._phase=='explore':
                    self._scores[self.k]=rate
                    if self.k<self.maximum:self.k+=1
                    else:
                        self.k=max(self._scores,key=lambda k:(self._scores[k],-k));self._phase='exploit';self._held=0;self._bad_windows=0
                else:
                    baseline=self._scores[self.k]
                    self._bad_windows=self._bad_windows+1 if rate<baseline*.75 else 0
                    self._scores[self.k]=.8*baseline+.2*rate
                    if self._held>=96 or (self._held>=16 and self._bad_windows>=3):
                        reset_reason='periodic'if self._held>=96 else'sustained-degradation'
                        self._scores={};self._phase='explore';self.k=self.minimum;self._held=0;self._bad_windows=0
        elif dt is not None and (dt<=0 or dt>10):
            self._samples.clear();self._bad_windows=0
        return {'group':self.groups,'drafted':drafted,'accepted':accepted,
                'recent_acceptance':sum(a for d,a in self.recent)/sum(d for d,a in self.recent),
                'previous_k':old,'next_k':self.k,'policy':'throughput-stable-v1-experimental',
                'phase':self._phase,'steady_interval':steady,'group_interval_s':dt,
                'window_step_tps':rate,'scores':dict(self._scores),'bad_windows':self._bad_windows,'reset_reason':reset_reason}
