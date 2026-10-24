// Cross-code benchmark, MFEM implementation. Problem and protocol: ../README.md.
//
//   ./bench -p 2 -n 8 -b both
//   mpirun -n 6 ./bench -p 2 -n 13 -b assembly
//
// MFEM's hyperelastic forms act on the current positions x = X + u (as in example 10); the
// residual and Jacobian with respect to x are those with respect to u. One JSON line per run is
// printed and appended to -o.
#include "mfem.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;

static const real_t L = 4.0, LAM = 10.0, MU = 1.0, A0 = 0.05;

// HyperFEM's NeoHookean3D: W = μ/2 (tr FᵀF − 3) − μ ln J + λ/2 (J − 1)²,
// P = μ F + s H with H = J F⁻ᵀ and s = −μ/J + λ (J − 1).
class HyperFEMNeoHookean : public HyperelasticModel
{
   mutable DenseMatrix H, M, T;

public:
   real_t EvalW(const DenseMatrix &F) const override
   {
      const real_t J = F.Det();
      return 0.5 * MU * (F.FNorm2() - 3.0) - MU * std::log(J) + 0.5 * LAM * (J - 1.0) * (J - 1.0);
   }

   void EvalP(const DenseMatrix &F, DenseMatrix &P) const override
   {
      const real_t J = F.Det();
      H.SetSize(3);
      CalcAdjugateTranspose(F, H);                 // H = J F⁻ᵀ
      const real_t s = -MU / J + LAM * (J - 1.0);
      P.SetSize(3);
      for (int i = 0; i < 9; i++) { P.GetData()[i] = MU * F.GetData()[i] + s * H.GetData()[i]; }
   }

   // A(i + a·dof, j + b·dof) += w Σ_{B,D} DS(i,B) C(a,B,b,D) DS(j,D), with
   // C(a,B,b,D) = ∂P_aB/∂F_bD = μ δ_ab δ_BD + s' H_aB H_bD + (s/J)(H_aB H_bD − H_aD H_bB).
   void AssembleH(const DenseMatrix &F, const DenseMatrix &DS, const real_t w,
                  DenseMatrix &A) const override
   {
      const int dof = DS.Height();
      const real_t J = F.Det();
      H.SetSize(3);
      CalcAdjugateTranspose(F, H);
      const real_t s = -MU / J + LAM * (J - 1.0), ds = MU / (J * J) + LAM;
      M.SetSize(3);
      T.SetSize(dof, 3);
      for (int a = 0; a < 3; a++)
      {
         for (int b = 0; b < 3; b++)
         {
            for (int B = 0; B < 3; B++)
            {
               for (int D = 0; D < 3; D++)
               {
                  M(B, D) = (a == b && B == D ? MU : 0.0) + (ds + s / J) * H(a, B) * H(b, D)
                            - (s / J) * H(a, D) * H(b, B);
               }
            }
            Mult(DS, M, T);                        // T = DS · M  (dof × 3)
            for (int i = 0; i < dof; i++)
            {
               for (int j = 0; j < dof; j++)
               {
                  real_t v = 0.0;
                  for (int D = 0; D < 3; D++) { v += T(i, D) * DS(j, D); }
                  A(i + a * dof, j + b * dof) += w * v;
               }
            }
         }
      }
   }
};

static double wtime(MPI_Comm comm, const std::function<void()> &f)
{
   MPI_Barrier(comm);
   double t = MPI_Wtime();
   f();
   MPI_Barrier(comm);
   double dt = MPI_Wtime() - t, dtmax;
   MPI_Allreduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, comm);
   return dtmax;
}

static double best(MPI_Comm comm, int reps, const std::function<void()> &f)
{
   double tb = 1e300;
   for (int r = 0; r < reps; r++) { tb = std::min(tb, wtime(comm, f)); }
   return tb;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;
   const int rank = Mpi::WorldRank(), nprocs = Mpi::WorldSize();

   int p = 2, n = 8, reps = 5, steps = 5;
   real_t delta = 0.5;
   const char *bench = "both";
   const char *out = "../../results/crosscode/mfem.jsonl";
   OptionsParser args(argc, argv);
   args.AddOption(&p, "-p", "--order", "Polynomial order.");
   args.AddOption(&n, "-n", "--n", "Mesh parameter: 4n x n x n hexahedra.");
   args.AddOption(&bench, "-b", "--bench", "assembly, newton or both.");
   args.AddOption(&reps, "-r", "--reps", "Repetitions (best time).");
   args.AddOption(&steps, "-s", "--steps", "Load steps.");
   args.AddOption(&delta, "-d", "--delta", "Prescribed transverse displacement at x = L.");
   args.AddOption(&out, "-o", "--out", "Output JSON-lines file.");
   args.Parse();
   if (!args.Good()) { if (rank == 0) { args.PrintUsage(std::cout); } return 1; }
   const std::string b(bench);
   const bool do_asm = (b == "assembly" || b == "both"), do_newton = (b == "newton" || b == "both");

   // ---- setup: mesh, space, form ---------------------------------------------------------------
   ParMesh *pmesh = nullptr;
   ParFiniteElementSpace *fes = nullptr;
   H1_FECollection fec(p, 3);
   const double t_space = wtime(comm, [&]()
   {
      Mesh mesh = Mesh::MakeCartesian3D(4 * n, n, n, Element::HEXAHEDRON, L, 1.0, 1.0);
      for (int be = 0; be < mesh.GetNBE(); be++)    // boundary attributes: 1 x = 0, 2 x = L, 3 other
      {
         Array<int> v;
         mesh.GetBdrElementVertices(be, v);
         real_t xm = 0.0;
         for (int k = 0; k < v.Size(); k++) { xm += mesh.GetVertex(v[k])[0] / v.Size(); }
         mesh.SetBdrAttribute(be, xm < 1e-10 ? 1 : (xm > L - 1e-10 ? 2 : 3));
      }
      mesh.SetAttributes();
      pmesh = new ParMesh(comm, mesh);
      fes = new ParFiniteElementSpace(pmesh, &fec, 3, Ordering::byNODES);
   });
   const HYPRE_BigInt ndofs = fes->GlobalTrueVSize();

   Array<int> ess_bdr(pmesh->bdr_attributes.Max()), ess_tdofs;
   ess_bdr = 0;
   ess_bdr[0] = ess_bdr[1] = 1;
   fes->GetEssentialTrueDofs(ess_bdr, ess_tdofs);
   int n_bc_loc = ess_tdofs.Size(), n_bc;
   MPI_Allreduce(&n_bc_loc, &n_bc, 1, MPI_INT, MPI_SUM, comm);

   HyperFEMNeoHookean model;
   ParNonlinearForm nlf(fes);
   auto *integ = new HyperelasticNLFIntegrator(&model);
   integ->SetIntRule(&IntRules.Get(Geometry::CUBE, 2 * p));
   nlf.AddDomainIntegrator(integ);
   nlf.SetEssentialTrueDofs(ess_tdofs);

   const int tsize = fes->GetTrueVSize();
   Vector x(tsize), r(tsize);
   auto norm = [&](const Vector &y) { return std::sqrt(InnerProduct(comm, y, y)); };
   auto positions = [&](std::function<void(const Vector &, Vector &)> f, Vector &xt)
   {
      ParGridFunction g(fes);
      VectorFunctionCoefficient c(3, f);
      g.ProjectCoefficient(c);
      g.GetTrueDofs(xt);
   };

   std::ostringstream js;
   js.precision(17);
   js << "{\"code\": \"mfem\", \"version\": \"" << MFEM_VERSION_STRING << "\", \"p\": " << p
      << ", \"n\": " << n << ", \"ndofs\": " << ndofs << ", \"nprocs\": " << nprocs
      << ", \"nthreads\": 1, \"t_setup_space\": " << t_space;

   // ---- benchmark 1: assembly at the fixed state u0 ----------------------------------------------
   if (do_asm)
   {
      positions([](const Vector &X, Vector &y)
      {
         y.SetSize(3);
         y(0) = X(0) + A0 * X(0) * X(1);
         y(1) = X(1) + A0 * X(0) * X(2);
         y(2) = X(2) + A0 * X(0) * X(1) * X(2);
      }, x);
      HypreParMatrix *K = nullptr;
      const double t_res1 = wtime(comm, [&]() { nlf.Mult(x, r); });
      const double t_jac1 = wtime(comm, [&]() { K = dynamic_cast<HypreParMatrix *>(&nlf.GetGradient(x)); });
      const double t_res = best(comm, reps, [&]() { nlf.Mult(x, r); });
      const double t_jac = best(comm, reps, [&]() { K = dynamic_cast<HypreParMatrix *>(&nlf.GetGradient(x)); });
      nlf.Mult(x, r);                             // essential entries set to 0
      const double res_norm = norm(r);
      K = dynamic_cast<HypreParMatrix *>(&nlf.GetGradient(x));   // BC rows/cols eliminated, 1 on the diagonal
      const double fro = K->FNorm();
      const double jac_fro = std::sqrt(fro * fro - n_bc);
      const double energy = nlf.GetEnergy(x);
      js << ", \"t_res_first\": " << t_res1 << ", \"t_jac_first\": " << t_jac1 << ", \"t_res\": " << t_res
         << ", \"t_jac\": " << t_jac << ", \"energy\": " << energy << ", \"res_norm\": " << res_norm
         << ", \"jac_fro\": " << jac_fro;
   }

   // ---- benchmark 2: Newton with Dirichlet load steps ---------------------------------------------
   if (do_newton)
   {
      Vector X(tsize), target(tsize), dx(tsize);
      positions([](const Vector &Xp, Vector &y) { y = Xp; }, X);
      x = X;
      double t_asm = 0.0, t_sol = 0.0;
      std::vector<int> iters;
      std::string solver_name;
      MPI_Barrier(comm);
      const double t0 = MPI_Wtime();
      for (int k = 1; k <= steps; k++)
      {
         const real_t lam = real_t(k) / steps;
         positions([&](const Vector &Xp, Vector &y)
         {
            y = Xp;
            if (Xp(0) > L - 1e-10) { y(2) -= delta * lam; }
         }, target);
         for (int i = 0; i < ess_tdofs.Size(); i++) { x(ess_tdofs[i]) = target(ess_tdofs[i]); }
         int it = 0;
         double r0 = -1.0;
         while (true)
         {
            t_asm += wtime(comm, [&]() { nlf.Mult(x, r); });
            const double rn = norm(r);
            if (r0 < 0.0) { r0 = rn; }
            if (rn <= std::max(1e-8 * r0, 1e-11) || it == 25) { break; }
            HypreParMatrix *K = nullptr;
            t_asm += wtime(comm, [&]() { K = dynamic_cast<HypreParMatrix *>(&nlf.GetGradient(x)); });
            t_sol += wtime(comm, [&]()
            {
#ifdef MFEM_USE_MUMPS
               MUMPSSolver ls(comm);
               ls.SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
               ls.SetOperator(*K);
               ls.Mult(r, dx);
               solver_name = "MUMPS";
#else
               // No sparse direct solver in this MFEM build: GMRES + BoomerAMG (elasticity
               // options) to a tight tolerance, so that the Newton iterations are those of a
               // direct solve.
               HypreBoomerAMG amg(*K);
               amg.SetSystemsOptions(3, true);
               amg.SetPrintLevel(0);
               HypreGMRES gmres(*K);
               gmres.SetTol(1e-12);
               gmres.SetMaxIter(2000);
               gmres.SetKDim(200);
               gmres.SetPreconditioner(amg);
               gmres.SetPrintLevel(0);
               dx = 0.0;
               gmres.Mult(r, dx);
               solver_name = "GMRES+BoomerAMG rtol 1e-12 (hypre)";
#endif
            });
            x -= dx;
            it++;
         }
         iters.push_back(it);
      }
      MPI_Barrier(comm);
      double t_total = MPI_Wtime() - t0, t_total_max;
      MPI_Allreduce(&t_total, &t_total_max, 1, MPI_DOUBLE, MPI_MAX, comm);
      const double energy_final = nlf.GetEnergy(x);
      js << ", \"newton_iters\": [";
      for (size_t i = 0; i < iters.size(); i++) { js << (i ? ", " : "") << iters[i]; }
      js << "], \"t_newton_total\": " << t_total_max << ", \"t_newton_assembly\": " << t_asm
         << ", \"t_newton_solve\": " << t_sol << ", \"linear_solver\": \"" << solver_name
         << "\", \"energy_final\": " << energy_final;
   }
   js << "}";

   if (rank == 0)
   {
      std::cout << js.str() << std::endl;
      std::ofstream f(out, std::ios::app);
      if (f) { f << js.str() << "\n"; }
      else { std::cerr << "warning: cannot open " << out << " (create the directory first)\n"; }
   }
   delete fes;
   delete pmesh;
   return 0;
}
