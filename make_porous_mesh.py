import dolfin as df
import mshr
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from utils import Params


def parse_args():
    parser = argparse.ArgumentParser(description="Make porous mesh")
    parser.add_argument("folder", type=str, help="Output folder")
    parser.add_argument("-Lx", type=float, default=1.0, help="Lx")
    parser.add_argument("-Ly", type=float, default=1.0, help="Ly")
    parser.add_argument("-R", type=float, default=0.04, help="R")
    parser.add_argument("-r", type=float, default=0.03, help="r")
    parser.add_argument("-N", type=int, default=100, help="N")
    parser.add_argument("-res", type=int, default=92, help="res")
    return parser.parse_args()

if __name__ == "__main__":
    size = df.MPI.size(df.MPI.comm_world)
    if size > 1:
        exit("Cannot be run in parallel")

    args = parse_args()
    N = args.N
    Lx, Ly = args.Lx, args.Ly
    R = args.R
    r = args.r
    res = args.res

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

    if not os.path.exists(args.folder):
        os.makedirs(args.folder)

    with df.HDF5File(mesh.mpi_comm(), os.path.join(args.folder, "porous_mesh.h5"), "w") as h5f:
        h5f.write(mesh, "mesh")

    params = Params()
    params["Lx"] = Lx
    params["Ly"] = Ly
    params["R"] = R
    params["r"] = r
    params["mesh"] = "porous_mesh.h5"
    params.save(os.path.join(args.folder, "params.dat"))

    df.plot(mesh)
    plt.show()