# Performance gate

Checks that a branch of HyperFEMKernels.jl is not slower, and does not allocate more, than `origin/main`, and that
the compiled backend is not slower than Gridap (the backend HyperFEM uses without compilation), on HyperFEM's
simulation tests, the examples and the benchmark problems of this repository. It runs by hand on a quiet
machine; it is not part of CI. Its deterministic part (paths of the groups, allocations, work done by `update`) is
in the test suite: `test/test_perf_invariants.jl`.

## When and how to run it

After the implementation and the independent review of a branch, before asking to push it, from the checkout of
the branch (or its worktree), with nothing else running on the machine:

```
benchmark/gate/gate.sh            # the branch against origin/main: every case, 1 and 4 threads (~1–1.5 h)
```

The first run in a work folder also resolves and precompiles two environments (one per checkout).
Other uses:

```
benchmark/gate/gate.sh --quick                                  # subset (simulations, EM, pattern build), to iterate
benchmark/gate/gate.sh --cases sims/Stokes,em_box_q2/jacobian --threads 4 --pairs 6   # repeat some cases
benchmark/gate/gate.sh --aa                                     # origin/main against itself: must PASS
benchmark/gate/gate.sh --threads 1,4,18                         # 18 threads too (informative)
benchmark/gate/gate.sh --inject 0.10                            # acceptance test: must FAIL
```

Output in `<folder containing _work>/_work/<branch>/gate/<date-time>/` (`latest` links to the newest run):
`report.md` (the table for the PR), `samples.csv` (every sample), `summary.toml`, `logs/`. Exit code: 0 PASS,
1 FAIL, 3 INCONCLUSIVE (the report gives the command that repeats the undecided cases with more pairs), 2 could not
run (git, environment, machine never quiet, a process died). A macOS notification tells when it ends.

## What is compared

For each case and thread count, two numbers of the compiled backend and one of memory:

- **branch / main**: time per call of the compiled backend, branch over main. FAIL when the whole 95 % interval
  is above 1.05 (1.25 for calls under 5 ms, where the noise is higher; simulations within 1 ms are not
  regressions);
- **compiled / Gridap**: in the same process (the state of the machine cancels), for the branch and for main.
  FAIL when the whole interval of the branch is above 1.02, or above the limit of an exception;
- **bytes per call**: the minimum over the samples. FAIL when the branch allocates more than main + 256 B for a
  call whose bytes are deterministic (assemblies, products, state updates), 5 % + 64 KiB more for other calls
  (builds, `update`, pattern build: ~2.5 % between processes with the same code) or 10 % + 1 MiB more for a
  simulation (~5 % between processes);
- **results**: a norm of the result of each case, branch against main (relative 1e-10 for calls, 1e-7 for
  simulations; a non-finite result of the branch fails);
- **informative**: cold latency (build and first call, one sample per process), the deterministic facts of the
  kernel operators (nnz, bytes, path of each group) when they change, and the Gridap branch/main ratio of every
  case, which measures the bias of the pairs of processes (the same code runs in both).

Cases (`cases.jl`; groups in `groups.jl`): HyperFEM's nine simulation tests; EM monolithic Q2 box (residual,
Jacobian, build, `update`, matrix-free prepare and apply, SpMV) and P1 plate; visco-hyperelastic residual,
Jacobian and internal-variable update; five micro forms on 40³ hexahedra; sparsity pattern build (elasticity Q1
40³, u–p Q1–Q1 30³); aorta P1–P1 on `TransientModel` (Jacobian, residual, LU of the step); aorta P2–P1
(`examples/aorta_cfd`: assembly of the step, `update`); hMRE + vacuum 2D (`examples/hmre_vacuum_2d`). Both
processes load the cases from the branch, so that they solve the same problems.

## How it measures

Two processes with the same code differ in a stable way (1–4 % on simulations, up to ~30 % on calls under 5 ms:
A/A experiment of 2026-11-21), so the unit of evidence is a pair of processes, not a repetition.

1. Stage 1, per thread count: one process of `origin/main` and one of the branch (built and warmed up one after
   the other, never together), all cases, 7 rounds. In each round every case is sampled by both processes in turn
   (ABBA), and each process times Gridap and the kernel alternately; a sample is the minimum time (and bytes) per
   call over a batch of each backend (at least 0.25 s and 3 calls; 2 calls from 1 s per call, 1 from 5 s). The estimate of a pair is the median over its
   rounds of the per-round ratios (one sample twice as long, seen inside a process, does not move it). A round is
   repeated (up to 3 times) when a fixed calibration computation drifts by more than 10 % across it (a gross
   disturbance; the CPU frequency alone moves it by a few per cent), or when other processes use a core or more
   (not with `--force`).
2. Suspect cases (branch/main above 1.03, or 1.25 under 5 ms; compiled/Gridap above its limit) are measured by 2
   more pairs of processes, which build the groups of those cases and warm up and measure only them. With
   `--cases`, every selected case gets the extra pairs.
3. Verdict per case on the t interval over the pairs (`stats.jl`): FAIL when the whole interval is above the
   limit, PASS when it is below, INCONCLUSIVE otherwise (then repeat with more pairs, as the report says). The
   smallest regression stage 1 catches has not been measured; a regression of a few per cent can pass it when the
   bias of that pair goes the other way.

Before the measurements the gate waits (up to 10 min) for the other processes to use less than one core; `--force`
measures anyway and says so in the report. The two environments must have the same Julia, HyperFEM and Gridap
(checked; the report gives the versions). BLAS threads are left at their default (what users get); the report
gives them.

## Exceptions

`exceptions.toml`: a case that may be slower than Gridap, or than main, up to a limit, with the reason, the
measurement and the decision. Every exception is a decision of Jesús; the limit stays checked, and the report
lists the exceptions in force.

## Files

`gate.sh` (worktree of origin/main, environments, driver, notification), `setup.jl` and `Project.toml` (one
environment per checkout), `driver.jl` (processes, rounds, stages, report; plain Julia), `worker.jl` (measuring
process), `cases.jl`, `groups.jl`, `aorta_transient.jl` (aorta P1–P1 problem), `stats.jl`, `exceptions.toml`.
