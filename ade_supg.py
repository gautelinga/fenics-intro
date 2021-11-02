import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import ufl
from stokes import Top
from make_porous_mesh import r

nu = 1.0
D = 1e-6
dt = 0.005
T = 1.0
Pe = 2*r/D
print("Pe =", Pe)

mesh = df.Mesh()
with df.HDF5File(mesh.mpi_comm(), "porous_mesh.h5", "r") as h5f:
    h5f.read(mesh, "mesh", False)

V = df.VectorFunctionSpace(mesh, "Lagrange", 2)
u_ = df.Function(V)
with df.HDF5File(mesh.mpi_comm(), "velocity.h5", "r") as h5f:
    h5f.read(u_, "u")

subd = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
subd.set_all(0)
Top.mark(subd, 2)

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

rc = c - c_1 + dt * (
    df.dot(u_, df.grad(c_mid))
    - D * df.div(df.grad(c_mid))
)
Fc = psi * (c - c_1) * df.dx + dt * (
    psi * df.dot(u_, df.grad(c_mid)) * df.dx
    + D * df.dot(df.grad(c), df.grad(psi)) * df.dx
)

u_norm = df.sqrt(df.dot(u_, u_))
Fc += h / (2 * u_norm) * df.dot(
    u_,df.grad(psi)) * rc * df.dx
ac = df.lhs(Fc)
Lc = df.rhs(Fc)

bc_c = df.DirichletBC(S, df.Expression("1.", degree=2), subd, 2)
bcs_c = [bc_c]

problem_c = df.LinearVariationalProblem(ac, Lc, c_, bcs=bcs_c)
solver_c = df.LinearVariationalSolver(problem_c)
solver_c.parameters["linear_solver"] = "bicgstab"
solver_c.parameters["preconditioner"] = "hypre_amg"

xdmff = df.XDMFFile(mesh.mpi_comm(), "c.xdmf")
xdmff.parameters["functions_share_mesh"] = True
xdmff.parameters["rewrite_function_mesh"] = False

t = 0.
while t < T:
    print(t)
    solver_c.solve()
    t += dt
    c_1.assign(c_)

    xdmff.write(c_, t)
xdmff.close()

df.plot(c_)
plt.show()