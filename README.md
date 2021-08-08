# Sleeping Machines
A manifesto for temporal computing where memory indexing is replaced by temporal delays and relative, causal, temporal referencing.

## A Mind Map
![Alt](sleeping_machines.png)

## Notes

- We believe Sleep Sort exposes something fundamental about computation in the time domain that is critical in understanding sparse asynchronous computing like that happening in biological neural networks.
- Formulating a theory of computation in a Turing machine analogue where the tape read/write is replaced by parallel, asynchronous nodes sleeping, waiting and sending causal signals should offer us ways to understand fast processes in biological neural networks better.
- It is likely that disabling Long-Term Potentiation (LTP) in a human brain would still allow the human to solve a Sudoku. It is therefore putting a carriage before the horse to go on with the hypothesis that neural signaling purpose is to effect LTP such as Spike Timing-Dependent Plasticity (STDP). Instead, LTP happens to support the primary process of fast time-domain distributed, sparse, asynchronous computation.
- In classical computers, the operations are dynamic in terms of space and topology; it is fast to access any memory location and read/write them. Hence sorting algorithms are based on swapping items around across arrays. The time dimension dynamics are synchronous, lockstep and inflexible.
- In biological neural networks, the topological connectivity and arbitrary spatial access are extremely static and limited, but time domain is flexible, asynchronous and rich in arbitrary dynamics. Hence sorting algorithms in such architectures would be conveniently based on delays.
- In quantum computing, the computation can be imagined to be offloaded to parallel universes where all potentials happen. The resulting signal of the computation results need to be communicated back over a shared channel, which limits the information bandwidth. This might be analogous to neural (non-quantum) computing where we can say that there are "enough" parallel units, but their signaling goes over a shared channel with cross-talk. The resulting computational complexity theory might have some shared results and analogies in quantum computation results.

## Causal Logic

- Emit A after delay x if no B before that.

## Computational Challenges

- Find a best alternative out of alternative signals.
- Predict the future signal.D

## Relation to Quantum Computing

In quantum computing the computation formalism is generally defined based on memories and gates. The memories have states which can be represented as complex numbers, and the gates operate on these states to produce new states. The operations are represented by matrix multiplications.

While state superpositions allow computational speed up compared to classical computing, in a way computing all the simultaneous state computational circuits simultaneously, and utilizing entanglement, constructive and destructive interference to make the alternatives fall into measurable consensus state, time dimension is not represented at all.

The formalism is spatial in nature, and defines way different memories are spatially separate and the relations between those spatially discrete states. Interestingly anti-causal phenomena can happen in quantum circuits.

However, complex numbers in principle can represent time-domain phenomena such as frequencies and phases in signals, so maybe analogous formalism could be used in neural computation, where something analogous to constructive and destructive interference can happen in neuronal connectivity in excitatory and inhibitory synaptic connections. The complex algebra might allow us to represent relative temporal characteristics of signals and how they interact.

We still need to define the formalism in a way which relates state to state evolution through operations. Instead of discrete memories I suggest we might consider representing the state as a mixture of signals, forming a reservoir computing -kind of a pool of dynamic interaction. Additionally, the fundamental computing operations should be causal in kind so that it matters which precondition triggers first, and these operations should have a tendency to "collapse" out the future alternatives which were made counterfactual.

Note that the state here is a single global spatial location (or region), and computation happens in the causal time dimension interactions within it. When we have formalism defined for one region, we can look into how to expand it (continuously?) into imperfectly connected sets of regions, which might form either discrete units (unrealistic?) or a continuum of units which blend into each other continuously.

## Relation to Neural Controlled Differential Equations

Controlled differential equations have a natural relationship with time and sparse signals. It is unclear to me if causality can be strictly incorporated into these formalisms so that the control for example always depends only on the past signal values. Additionally, as it currently is, controlled differential equations and even neural controlled differential equations don't seem to have appropriate focus on parallel reservoir computing where multiple branches of computations compete with each others in time, and where the numbers of parallel branches on-going could be abstracted as a continuous distribution over states in a reservoir instead of explicitly defining micro-level functions.

## Some Initial Thoughts

Let's divide the time into causal split, where the past and the present is represented as a state.

Let's represent the future as a pool of potential events, every event having an associated distribution of latency from the current time.

Let's represent the operation of rolling the time forward one infinitesimal step as some iterative operator `W` which weaves the future potential events into the present state representation. In specific:

`s', f' = W(s, f)`, where `s` is the state before weaving, `s'` is the state after weaving, `f` is the pool of future potential events, and `f'` is a next time slice modulated pool of future potential events. `W` operator is stochastic, and samples the latency distributions in `f` so that some subset of `f` is picked to be applied, and that subset applies to `f` to produce `f'` and to `s` to produce `s'`.

When `x in f` applies, it needs to be self-canceling, so that it removes itself from the pool of future potential events. Every time step also needs to modulate the `f` to take an infinitesimal time step forward, we call that a trivial modulation `t`. The trivial modulation simply reduces the expected latencies of all future potential events uniformly. In general, `x in f` can increase or decrease latencies of future potential events in a non-uniform fashion. Causality cannot be broken, so expected latencies cannot be negative.

The beef is in how the iterative weaving operator `W` is defined. It can be defined as follows:

1. Slice the next infinitesimal time slice of all the future potential event probability distributions in `f` so that we get the independent probability for each potential event at this next time slice. Note that event probabilities are causally dependent over successive time slices, but independent within a time slice.
2. Sample the set of modulations to apply, `m subset_of f`, where `x in m`.
3. Combine these modulations with the trivial modulation `t`.
4. Apply these modulations conditioned by `s` to `f` so that latency probability distributions of different events shift to produce `f'`, which also cancels all `m` in `f'` so that `f' intersection m = 0`.
5. Apply the state transformations related to these modulations to `s` to produce `s'`.

The pool of future potential events needs to have the following operators defined:
1. Sample `m` from `f`.
2. Apply `m + t` conditioned by `s` to `f` to produce `f'`.

The state needs to have the following operators defined:
1. Apply `m + t` to `s` to produce `s'`.

## Wild Ideas

- What if neurons which roughly integrate and fire actually integrate confidence in some interpretation of the world, and fire as soon as the confidence for this achieves large enough value? Populations of neurons would then listen to signals from the external world, each increasing and decreasing its own internal confidence, and compete in which neuron fires its own subsymbol first, thus broadcasting this opinion of truth which the other neurons then need to take into account in their own continuously integrated confidences.

## References

- [Sleep Sort](https://rosettacode.org/wiki/Sorting_algorithms/Sleep_sort): "Basically if you transform set L into the time domain, when collecting it back you get it back sorted.", "the time and space complexity for Sleep sort are O(1)."
- [Synapses and Memory Storage](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3367555/): "For example, certain genetic manipulations that disrupt hippocampal LTP do not impair forms of memory believed to require the hippocampus (e.g., Zamanillo et al. 1999)."
- [On the origin of chaos in autonomous Boolean networks](https://royalsocietypublishing.org/doi/10.1098/rsta.2009.0235): "An autonomous Boolean network (ABN) is a set of nodes with binary values coupled by links with associated time delays. Each node is updated continuously according to a designated Boolean function of the values of its inputs at the appropriate previous times."
- [Causal Logic Models](https://sites.stat.washington.edu/tsr/uai-causal-structure-learning-workshop/papers/dash.pdf): "We define Causal Logic Models (CLMs), a new probabilistic, first-order representation which uses causality as a fundamental building block. Rather than merely converting causal rules to first-order logic as various methods in Statistical Relational Learning have done, we treat the causal rules as basic primitives which cannot be altered without changing the system."
- [Learning and Memory](http://michaeldmann.net/mann18.html): Explains how learning works on biology in-depth detail in the synapses.
- [A Memory without a Brain](https://www.tum.de/nc/en/about-tum/news/press-releases/details/36462/)
- [Quantum Parallelism](https://www.sciencedirect.com/topics/mathematics/quantum-parallelism): "Like most factorization algorithms, Shor's algorithm reduces the factorization problem to the problem of finding the period of a function, but uses quantum parallelism to find a superposition of all values of the function in one step.", "In 1996 Grover described a quantum algorithm for searching an unsorted database D containing N items in a time of order N; on a classical computer, the search requires a time of order N."
- [Machine Learning With Neural Controlled Differential Equations](https://www.maths.ox.ac.uk/node/38559): "Neural controlled differential equations are actually the continuous-time limit of recurrent neural networks."
- [Towards Causal Representation Learning](https://arxiv.org/abs/2102.11107): "A central problem for AI and causality is, thus, causal representation learning, the discovery of high-level causal variables from low-level observations."
- [“It’s really important not just how many [neuron activations] occur, but when exactly they occur.”](https://www.quantamagazine.org/a-new-kind-of-information-coding-seen-in-the-human-brain-20210707/)
