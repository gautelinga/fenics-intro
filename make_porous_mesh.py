import dolfin as df
import mshr
import numpy as np
import matplotlib.pyplot as plt

Lx = 1.0
Ly = 1.0
R = 0.04
r = 0.03
N = 100
res = 92

if __name__ == "__main__":
    n_put = 0
    n_tries = 0
    pos = np.zeros((N, 2))
    while n_put < N:
        x = R + (np.array([Lx, Ly]) - 2*R) * np.random.rand(2)
        dr = np.linalg.norm(pos[:n_put, :] - np.ones((n_put, 1)) * x, axis=1)
        if all(dr > 2*R):
            pos[n_put, :] = x
            n_put += 1
            n_tries = 0
        else:
            n_tries += 1

        if n_tries >= 1000:
            break
    pos = pos[:n_put, :]
    #print(pos)

    geom = mshr.Rectangle(df.Point(0., 0.), df.Point(Lx, Ly))
    for x in pos:
        geom -= mshr.Circle(df.Point(x), r)

    mesh = mshr.generate_mesh(geom, res)

    with df.HDF5File(mesh.mpi_comm(), "porous_mesh.h5", "w") as h5f:
        h5f.write(mesh, "mesh")

    df.plot(mesh)
    plt.show()