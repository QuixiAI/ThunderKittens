# Claude Instructions

Follow `AGENTS.md` in this repository. For kernel work, read `perf/perf.md`
before editing and update `perf/optimization_status.md` with measured results.

Do not commit any kernel implementation, routing change, benchmark change, or
performance claim until at least one focused performance optimization run has
been completed on an affected kernel. If the required CUDA runtime is
unavailable, report the blocker instead of committing a claimed optimization.
