# Tables of the cross-code benchmark from the JSON lines in results/crosscode/*.jsonl.
#
#   python benchmark/crosscode/compare.py [results directory]
#
# For each (p, n): assembly times, the ratio to the fastest run, and the relative difference of the
# check values (energy, residual and Jacobian norms) to the first code listed; then the Newton
# runs. The last run of each (code, processes, threads, p, n, assembly/newton) is used.
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results", "crosscode")
runs = {}
for path in sorted(glob.glob(os.path.join(root, "*.jsonl"))):
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                kind = "newton" if "newton_iters" in r and "t_jac" not in r else "assembly"
                runs[(r["code"], r["nprocs"], r["nthreads"], r["p"], r["n"], kind)] = r
if not runs:
    sys.exit(f"no results in {root}")


def fmt(t):
    return "—" if t is None else f"{t:.4f}"


def rel(a, b):
    return "—" if a is None or b is None else f"{abs(a - b) / max(abs(b), 1e-300):.1e}"


def label(r):
    mpi = not r["code"].startswith("hyperfem")
    par = f"{r['nprocs']} proc" if mpi else f"{r['nthreads']} thr"
    return f"{r['code']} ({par})"


for p, n in sorted({(k[3], k[4]) for k in runs}):
    group = [r for k, r in runs.items() if (k[3], k[4]) == (p, n)]
    ndofs = group[0]["ndofs"]
    print(f"\n## Q{p}, mesh {4 * n}×{n}×{n} hexahedra, {ndofs} DOFs\n")
    asm = [r for r in group if "t_jac" in r]
    if asm:
        ref = asm[0]
        tj0 = min(r["t_jac"] for r in asm)
        tr0 = min(r["t_res"] for r in asm)
        print("| code | Jacobian (s) | × fastest | residual (s) | × fastest | first Jacobian (s) | "
              f"Δ energy | Δ ‖R‖ | Δ ‖K‖_F (vs {ref['code']}) |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(asm, key=lambda r: r["t_jac"]):
            print(f"| {label(r)} | {fmt(r['t_jac'])} | {r['t_jac'] / tj0:.1f} | {fmt(r['t_res'])} | "
                  f"{r['t_res'] / tr0:.1f} | {fmt(r.get('t_jac_first'))} | {rel(r['energy'], ref['energy'])} | "
                  f"{rel(r['res_norm'], ref['res_norm'])} | {rel(r['jac_fro'], ref['jac_fro'])} |")
    newton = [r for r in group if "newton_iters" in r]
    if newton:
        ref = newton[0]
        print("\n| code | Newton iterations per step | assembly (s) | linear solve (s) | total (s) | "
              f"linear solver | Δ final energy (vs {ref['code']}) |")
        print("|---|---|---|---|---|---|---|")
        for r in sorted(newton, key=lambda r: r["t_newton_total"]):
            print(f"| {label(r)} | {r['newton_iters']} | {fmt(r['t_newton_assembly'])} | "
                  f"{fmt(r['t_newton_solve'])} | {fmt(r['t_newton_total'])} | {r['linear_solver']} | "
                  f"{rel(r['energy_final'], ref['energy_final'])} |")
