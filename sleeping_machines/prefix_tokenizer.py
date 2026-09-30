"""Train-only, complete prefix dictionaries for causal language experiments.

Leaves are tokens; every internal node has all alphabet children. A leaf is
released exactly when its final character arrives. No lookahead or unknown
token is needed. The learned language model supplies probabilities: frequency
counts only choose the vocabulary and never act as a prediction expert.
"""
from collections import Counter
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .stream_language import StreamingEventLanguageModel


class CompletePrefixTokenizer:
    def __init__(self, phrases, alphabet=27):
        self.alphabet=int(alphabet)
        self.phrases=tuple(sorted(tuple(map(int,p)) for p in phrases))
        if len(set(self.phrases))!=len(self.phrases) or not self.phrases:
            raise ValueError('Distinct nonempty phrases required')
        self.leaves={p:i for i,p in enumerate(self.phrases)}
        self.descendants={}
        for i,p in enumerate(self.phrases):
            if not p or any(c<0 or c>=self.alphabet for c in p):
                raise ValueError('Invalid phrase')
            for k in range(len(p)+1):self.descendants.setdefault(p[:k],[]).append(i)
        for prefix in self.descendants:
            if prefix in self.leaves:
                if len(self.descendants[prefix])!=1:raise ValueError('Not prefix free')
            elif any(prefix+(c,) not in self.descendants for c in range(self.alphabet)):
                raise ValueError('Incomplete dictionary')

    @classmethod
    def fit(cls, training, expansions=4, alphabet=27, max_length=4):
        raw=tuple(map(int,training))
        if expansions<0 or max_length<1:raise ValueError('Invalid capacity')
        counts=Counter(raw[i:i+k] for i in range(len(raw))
                       for k in range(1,min(max_length,len(raw)-i)+1))
        leaves={(c,) for c in range(alphabet)}
        for _ in range(expansions):
            eligible=[p for p in leaves if len(p)<max_length]
            if not eligible:raise ValueError('Requested more expansions than depth permits')
            selected=min(eligible,key=lambda p:(-counts[p],p))
            leaves.remove(selected)
            leaves.update(selected+(c,) for c in range(alphabet))
        return cls(leaves,alphabet)

    def encode(self, characters):
        prefix=();tokens=[]
        for c in characters:
            prefix=prefix+(int(c),)
            if prefix not in self.descendants:raise ValueError('Unknown character')
            if prefix in self.leaves:tokens.append(self.leaves[prefix]);prefix=()
        return tokens,prefix

    def decode(self,tokens):
        return tuple(c for token in tokens for c in self.phrases[int(token)])

    def log_mass(self,log_probability,prefix):
        indices=torch.tensor(self.descendants[prefix],device=log_probability.device)
        return torch.logsumexp(log_probability.index_select(0,indices),0)


class ByteTimedTokenModel(StreamingEventLanguageModel):
    def observe_token(self,token,raw_arrival,state):
        """Token time is its last observed character's index, for both arms."""
        arrival=float(raw_arrival)
        self.advance(state,arrival,readout=False)
        index=torch.tensor(int(token),device=self.embedding.weight.device)
        self._enqueue(state,0,self.embedding(index),self.embedding.weight.new_tensor(arrival))
        logits=self.advance(state,arrival+self.query_budget)
        state.position+=1
        return logits


@dataclass
class PrefixLanguageState:
    event: object
    log_probability: torch.Tensor
    log_mass: torch.Tensor
    prefix: tuple=()
    raw_position: int=0

    def detach(self):
        return PrefixLanguageState(self.event.detach(),self.log_probability.detach(),
                                   self.log_mass.detach(),self.prefix,self.raw_position)


def new_prefix_state(model,tokenizer):
    event=model.new_state()
    lp=F.log_softmax(model.head(event.output),0)
    return PrefixLanguageState(event,lp,tokenizer.log_mass(lp,()))


def observe_character(model,tokenizer,state,character):
    """Score the unobserved byte, then observe it; score every partial phrase.

    Subtree probability ratios define normalized next-character probabilities.
    Within a completed phrase their log scores telescope to its token log score.
    The final unfinished phrase is scored exactly by subtree marginalization.
    """
    next_prefix=state.prefix+(int(character),)
    mass=tokenizer.log_mass(state.log_probability,next_prefix)
    nll=state.log_mass-mass
    if next_prefix in tokenizer.leaves:
        token=tokenizer.leaves[next_prefix]
        lp=F.log_softmax(model.observe_token(token,state.raw_position,state.event),0)
        state.log_probability=lp;state.log_mass=tokenizer.log_mass(lp,());state.prefix=()
    else:
        state.log_mass=mass;state.prefix=next_prefix
    state.raw_position+=1
    return nll
