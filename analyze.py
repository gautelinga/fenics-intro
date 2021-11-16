from utils import Params
import argparse
import os
import xml.etree.ElementTree as ET
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri
import mpi4py


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze")
    parser.add_argument("folder", type=str, help="Input folder")
    parser.add_argument("--plot", action="store_true", help="Plot front")
    return parser.parse_args()


def pathlength(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))


if __name__ == "__main__":
    if mpi4py.MPI.COMM_WORLD.Get_size() > 1:
        exit("Cannot be run in parallel")

    args = parse_args()
    ade_params = Params(os.path.join(args.folder, "params.dat"))

    topology_path = None
    geometry_path = None

    timeseries = []

    root = ET.parse(os.path.join(args.folder, "c.xdmf")).getroot()
    for grid in root[0][0]:
        assert(grid.tag == "Grid")
        time, c_path = None, None
        for element in grid:
            if element.tag == "Topology":
                topology_path = element[0].text.split(":")
            elif element.tag == "Geometry":
                geometry_path = element[0].text.split(":")
            elif element.tag == "Attribute" and element.attrib["Name"] == "c":
                c_path = element[0].text.split(":")
            elif element.tag == "Time":
                time = float(element.attrib["Value"])
        assert ( time is not None and c_path is not None )
        timeseries.append((time, c_path))

    assert (topology_path[0] == geometry_path[0])
    h5f = h5py.File(os.path.join(args.folder, topology_path[0]), "r")
    
    topology = np.array(h5f[topology_path[1]])
    geometry = np.array(h5f[geometry_path[1]])

    triang = tri.Triangulation(
        geometry[:, 0], geometry[:, 1], triangles=topology)

    imgfolder = os.path.join(args.folder, "images")
    if not os.path.exists(imgfolder):
        os.makedirs(imgfolder)

    tL = np.zeros((len(timeseries), 2))
    for it, (t, c_path) in enumerate(timeseries):
        c = np.array(h5f[c_path[1]]).flatten()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        tcset = ax.tricontour(triang, c, levels=[0.5])

        L = 0
        paths = tcset.collections[0].get_paths()
        for path in paths:
            x = path.vertices
            L += pathlength(x[:, 0], x[:, 1])
            if args.plot:
                plt.scatter(x[:, 0], x[:, 1])
        tL[it, 0] = t
        tL[it, 1] = L
        if args.plot:
            ax.set_title("t = {:04.6f}".format(t))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.savefig(os.path.join(imgfolder, "front_{:06d}.png".format(it)))
            #plt.show()
        plt.close()

    np.savetxt(os.path.join(args.folder, "t_vs_L.dat"), tL)
    plt.plot(tL[:, 0], tL[:, 1])
    plt.xlabel("time")
    plt.ylabel("front length")
    plt.show()