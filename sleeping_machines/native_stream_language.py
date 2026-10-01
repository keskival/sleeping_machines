"""Token content adapter for the native addressed temporal core; no KV bank."""
import torch
from torch.nn import functional as F

from .addressed_event_heads import AddressedEventHeads, AddressedEventState


class NativeLanguageState(AddressedEventState):
    def packed_storage(self):
        # Compatibility for the frozen accumulator's detach/recovery contract.
        stats=self.storage()
        stats['differentiable_entries']=sum(int(t.requires_grad) for t in self.memories.values())
        return stats


class NativeStreamLanguageModel(AddressedEventHeads):
    def __init__(self,payload=16,depth=8,pool=2,heads=2,vocabulary=27,matching=0,recent=0):
        if matching or recent:
            raise ValueError('Native persistent state has no episodic candidate index')
        super().__init__(sources=1,content_dim=vocabulary,classes=vocabulary,
                         payload=payload,depth=depth,pool=pool,heads=heads)
        self.vocabulary=vocabulary

    def new_state(self):
        return NativeLanguageState()

    def forward_chunk(self,tokens,state=None):
        state=self.new_state() if state is None else state
        indices=torch.as_tensor(tokens,dtype=torch.long,device=self.embedding.weight.device)
        if indices.ndim!=1 or not len(indices):
            raise ValueError('Nonempty observed token stream required')
        outputs=[]
        for token in indices:
            mark=F.one_hot(token,num_classes=self.vocabulary).to(self.embedding.weight.dtype)
            logits,_=self.consume_event(0,float(state.events),mark,state)
            outputs.append(logits)
        return torch.stack(outputs),state
