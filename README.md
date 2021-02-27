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

## References

- [Sleep Sort](https://rosettacode.org/wiki/Sorting_algorithms/Sleep_sort): "Basically if you transform set L into the time domain, when collecting it back you get it back sorted.", "the time and space complexity for Sleep sort are O(1)."
- [Synapses and Memory Storage](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3367555/): "For example, certain genetic manipulations that disrupt hippocampal LTP do not impair forms of memory believed to require the hippocampus (e.g., Zamanillo et al. 1999)."
