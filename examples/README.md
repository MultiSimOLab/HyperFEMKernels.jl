# Examples

- `aorta_cfd/` — pulsatile blood flow in a patient aorta (incompressible Navier–Stokes, Oseen–BDF2,
  Taylor–Hood P2–P1 with SUPG/PSPG/LSIC, backflow and Windkessel outlets). The same weak form is assembled
  with Gridap and with HyperFEMKernels; `mode=compare` checks matrix and vector at every step and times both.

HyperFEM examples (electromechanical 3D and hMRE 2D) run with `backend = :kernel | :gridap`
through the HyperFEM adapter (see `test/hyperfem` and `benchmark/`).
