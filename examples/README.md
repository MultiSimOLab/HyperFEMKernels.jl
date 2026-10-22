# Examples

- `aorta_cfd/` — pulsatile blood flow in a patient aorta (incompressible Navier–Stokes, Oseen–BDF2,
  Taylor–Hood P2–P1 with SUPG/PSPG/LSIC, backflow and Windkessel outlets). The same weak form is assembled
  with Gridap and with HyperFEMKernels; `mode=compare` checks matrix and vector at every step and times both.

- `hmre_vacuum_2d/` — hMRE beam with explicit surrounding air, 2D, three staggered problems (magnetic, solid,
  vacuum mesh motion) with HyperFEM's drivers and `backend = :gridap | :kernel` (local copy of
  Adv_Mater_Ultra_Soft_Magnet `hMRE_Vacuum_2D.jl`); `mode=compare` runs both and compares the final solutions.

The EM 3D and hMRE 2D cases run as benchmarks (`benchmark/`) and tests (`test/hyperfem`).
