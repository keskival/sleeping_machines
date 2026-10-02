"""Generator-informed two-trace diagnostic; no learned routes or supremacy claim."""
import math
import torch


class DecayTimingReference:
    def __init__(self):
        self.state={}
        self.last_time=-math.inf
        self.tau=torch.tensor([.7,4.],dtype=torch.float64)

    def consume(self,source,timestamp,mark):
        if not math.isfinite(timestamp) or timestamp<self.last_time:
            raise ValueError('Finite nondecreasing event times required')
        value,query=mark
        if not math.isfinite(value) or query not in (0.,1.):
            raise ValueError('Finite mark and explicit query flag required')
        now=torch.tensor(timestamp,dtype=torch.float64)
        if source in self.state:
            previous,traces=self.state[source]
            traces=traces*torch.exp(-(now-previous)/self.tau)
        else:
            traces=torch.zeros(2,dtype=torch.float64)
        traces=traces+value
        self.state[source]=(now,traces)
        self.last_time=timestamp
        return traces[0]-.6*traces[1] if query else None
