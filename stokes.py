import dolfin as df
import matplotlib.pyplot as plt

def wall(x, on_bnd):
    return on_bnd and x[0] < df.DOLFIN_EPS_LARGE or x[0] > 1.-df.DOLFIN_EPS_LARGE
def top(x, on_bnd):
    return on_bnd and x[1] > 1-df.DOLFIN_EPS_LARGE
def btm(x, on_bnd):
    return on_bnd and x[1] < df.DOLFIN_EPS_LARGE
def obst(x, on_bnd):
    return on_bnd and x[0] > df.DOLFIN_EPS_LARGE and x[0] < 1.-df.DOLFIN_EPS_LARGE and \
        x[1] > df.DOLFIN_EPS_LARGE and x[1] < 1.0 - df.DOLFIN_EPS_LARGE

Wall = df.AutoSubDomain(wall)
Top = df.AutoSubDomain(top)
Btm = df.AutoSubDomain(btm)
Obst = df.AutoSubDomain(obst)

if __name__ == "__main__":
    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), "porous_mesh.h5", "r") as h5f:
        h5f.read(mesh, "mesh", False)

    V_el = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W_el = df.MixedElement([V_el, P_el])
    W = df.FunctionSpace(mesh, W_el)

    u, p = df.TrialFunctions(W)
    v, q = df.TestFunctions(W)

    # - nu * div grad u = - grad p + f
    # div u = 0
    a_stokes = df.inner(df.grad(u), df.grad(v)) * df.dx \
        - p * df.div(v)*df.dx - q * df.div(u)*df.dx
    L_stokes = df.dot(df.Constant((0., -1.)), v) * df.dx
    b_stokes = df.inner(df.grad(u), df.grad(v)) * df.dx + q * p * df.dx

    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    subd.set_all(0)

    Wall.mark(subd, 1)
    Top.mark(subd, 2)
    Btm.mark(subd, 3)
    Obst.mark(subd, 4)

    bcu_wall = df.DirichletBC(W.sub(0), df.Constant((0., 0.)), subd, 1)
    bcu_obst = df.DirichletBC(W.sub(0), df.Constant((0., 0.)), subd, 4)
    bcp_top = df.DirichletBC(W.sub(1), df.Constant(0.), subd, 2)
    bcp_btm = df.DirichletBC(W.sub(1), df.Constant(0.), subd, 3)
    bcs_stokes = [bcu_wall, bcu_obst, bcp_top, bcp_btm]

    w_ = df.Function(W)

    A, b = df.assemble_system(a_stokes, L_stokes, bcs=bcs_stokes)
    P, btmp = df.assemble_system(b_stokes, L_stokes, bcs=bcs_stokes)
    solver_stokes = df.PETScKrylovSolver("minres", "amg")
    solver_stokes.parameters["monitor_convergence"] = True
    solver_stokes.set_operators(A, P)
    solver_stokes.solve(w_.vector(), b)

    u_, p_ = w_.split(deepcopy=True)
    u_.rename("u", "tmp")
    df.plot(u_)
    plt.show()

    uy_mean = df.assemble(u_[1] * df.dx) / df.assemble(df.Constant(1.)*df.dx(domain=mesh))
    print("uy_mean =", uy_mean)
    u_.vector()[:] /= abs(uy_mean)

    with df.HDF5File(mesh.mpi_comm(), "velocity.h5", "w") as h5f:
        h5f.write(u_, "u")