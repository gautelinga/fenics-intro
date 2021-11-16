import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import ufl
from stokes import Top
from utils import Params
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Solve Stokes flow")
    parser.add_argument("folder", type=str, help="Input/output folder")
    parser.add_argument("-D", type=float, default=1e-6, help="Diffusivity")
    parser.add_argument("-dt", type=float, default=0.005, help="Timestep")
    parser.add_argument("-T", type=float, default=1.0, help="Total time")
    return parser.parse_args()

def get_next_subfolder(folder):
    i = 0
    while os.path.exists(os.path.join(folder, "{}".format(i))):
        i += 1
    return os.path.join(folder, "{}".format(i))

if __name__ == "__main__":
    rank = df.MPI.rank(df.MPI.comm_world)

    args = parse_args()
    D = args.D
    dt = args.dt
    T = args.T

    mesh_params = Params(os.path.join(args.folder, "params.dat"))
    Lx = mesh_params["Lx"]
    Ly = mesh_params["Ly"]

    results_folder = get_next_subfolder(args.folder)
    if rank == 0:
        os.makedirs(results_folder)
    ade_params = Params()
    ade_params["D"] = D
    ade_params["dt"] = dt
    ade_params["T"] = T
    ade_params.save(os.path.join(results_folder, "params.dat"))

    Pe = 2*mesh_params["r"]/D

    if rank == 0:
        print("Pe =", Pe)

    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), os.path.join(args.folder, mesh_params["mesh"]), "r") as h5f:
        h5f.read(mesh, "mesh", False)

    V = df.VectorFunctionSpace(mesh, "Lagrange", 2)
    u_ = df.Function(V)
    with df.HDF5File(mesh.mpi_comm(), os.path.join(args.folder, "velocity.h5"), "r") as h5f:
        h5f.read(u_, "u")

    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    subd.set_all(0)
    Top(Lx, Ly).mark(subd, 2)

    S_DG0 = df.FunctionSpace(mesh, "DG", 0)
    u_proj_x = df.project(u_[0], S_DG0, solver_type="gmres", preconditioner_type="amg")
    u_proj_y = df.project(u_[1], S_DG0, solver_type="gmres", preconditioner_type="amg")
    u_proj_ = df.as_vector((u_proj_x, u_proj_y))

    S = df.FunctionSpace(mesh, "Lagrange", 1)

    c = df.TrialFunction(S)
    psi = df.TestFunction(S)

    c_ = df.Function(S, name="c")
    c_1 = df.Function(S, name="c_prev")

    h = df.CellDiameter(mesh)
    c_mid = 0.5*(c + c_1)

    # residual
    rc = c - c_1 + dt * (
        df.dot(u_, df.grad(c_mid))
        - D * df.div(df.grad(c_mid))
    )

    # Variational form
    Fc = psi * (c - c_1) * df.dx + dt * (
        psi * df.dot(u_, df.grad(c_mid)) * df.dx
        + D * df.dot(df.grad(c), df.grad(psi)) * df.dx
    )

    u_norm = df.sqrt(df.dot(u_, u_))
    Fc += h / (2 * u_norm) * df.dot(u_, df.grad(psi)) * rc * df.dx
    ac = df.lhs(Fc)
    Lc = df.rhs(Fc)

    bc_c = df.DirichletBC(S, df.Expression("1.", degree=2), subd, 2)
    bcs_c = [bc_c]

    problem_c = df.LinearVariationalProblem(ac, Lc, c_, bcs=bcs_c)
    solver_c = df.LinearVariationalSolver(problem_c)
    solver_c.parameters["linear_solver"] = "bicgstab"
    solver_c.parameters["preconditioner"] = "hypre_amg"
    solver_c.parameters["krylov_solver"]["relative_tolerance"] = 1e-6
    solver_c.parameters["krylov_solver"]["monitor_convergence"] = True

    xdmff = df.XDMFFile(mesh.mpi_comm(), os.path.join(results_folder, "c.xdmf"))
    xdmff.parameters["functions_share_mesh"] = True
    xdmff.parameters["rewrite_function_mesh"] = False
    xdmff.parameters["flush_output"] = True

    t = 0.
    while t < T:
        if rank == 0:
            print(t)
        solver_c.solve()
        t += dt
        c_1.assign(c_)

        xdmff.write(c_, t)
    xdmff.close()

    df.plot(c_)
    plt.show()