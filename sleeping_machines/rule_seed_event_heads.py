"""Independent rule/embedding sharing diagnostic; inherited event dynamics intact."""
from torch import nn
from .split_event_heads import SplitEventHeads, CommonSourceSeed


class RuleSeedEventHeads(SplitEventHeads):
    def __init__(self,*args,common_source_seed=None,**kwargs):
        super().__init__(*args,**kwargs)
        common=self.shared_maps if common_source_seed is None else bool(common_source_seed)
        if common!=self.shared_maps:
            self.embedding=(CommonSourceSeed(self.sources,self.total_payload) if common else
                            nn.Embedding(self.sources,self.total_payload))
        self.common_source_seed=common
