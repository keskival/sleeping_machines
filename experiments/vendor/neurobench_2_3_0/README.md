Unmodified copy of the `neurobench` 2.3.0 package (PyPI wheel neurobench-2.3.0-py3-none-any.whl), Apache-2.0
(see licenses/). Used for the official NeuroBench dataset loaders and metrics in experiments/SOTA_TARGETS.md
work (Mackey-Glass, primate reaching). Not installed into the environment; import by adding this directory to
sys.path. The Mackey-Glass dataset class imports `jitcdde` only for data generation; with the official downloaded
series it only loads, so tests stub that import.
