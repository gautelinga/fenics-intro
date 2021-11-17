from ufl.operators import ge
from utils import Params
import argparse
import os
import xml.etree.ElementTree as ET
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri
import mpi4py
import dolfin as df


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze")
    parser.add_argument("folder", type=str, help="Input folder")
    parser.add_argument("--plot", action="store_true", help="Plot front")
    return parser.parse_args()


def pathlength(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))


def get_boundary_edges(cells):
    edge_dict = dict()
    for cell in cells:
        for id1, id2 in [[0, 1], [1, 2], [2, 0]]:
            key = (min(cell[id1], cell[id2]), max(cell[id1], cell[id2]))
            if key in edge_dict:
                edge_dict[key] += 1
            else:
                edge_dict[key] = 1

    edges = []            
    for key, val in edge_dict.items():
        if val == 1:
            edges.append(key)
    return edges


def get_other(vs, v):
    assert(len(vs) == 2)
    if vs[0] == v:
        return vs[1]
    assert vs[1] == v
    return vs[0]

def get_closed_loops(boundary_edges):
    node2node = dict()
    for v1, v2 in boundary_edges:
        if v1 not in node2node:
            node2node[v1] = [v2]
        else:
            node2node[v1].append(v2)
        if v2 not in node2node:
            node2node[v2] = [v1]
        else:
            node2node[v2].append(v1)
    nodes_left = set(np.unique(boundary_edges))
    loops = []
    while len(nodes_left) > 0:
        v0 = nodes_left.pop()
        v1 = node2node[v0][0]
        v1_prev = v0
        loop = [v0]
        while v1 != v0:
            loop.append(v1)
            nodes_left.remove(v1)
            v1, v1_prev = get_other(node2node[v1], v1_prev), v1
        loops.append(loop)
    return loops


if __name__ == "__main__":
    if mpi4py.MPI.COMM_WORLD.Get_size() > 1:
        exit("Cannot be run in parallel")

    args = parse_args()
    ade_params = Params(os.path.join(args.folder, "params.dat"))
    mesh_params = Params(os.path.join(args.folder, "..", "params.dat"))

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

    # Filling in the holes
    boundary_edges = get_boundary_edges(topology)
    loops = get_closed_loops(boundary_edges)

    new_nodes = []
    new_cells = []
    i = len(geometry)
    for loop in loops:
        x = geometry[loop, :]
        if (all(x[:, 0] > df.DOLFIN_EPS_LARGE) 
            and all(x[:, 0] < mesh_params["Lx"] - df.DOLFIN_EPS_LARGE)
            and all(x[:, 1] > df.DOLFIN_EPS_LARGE) 
            and all(x[:, 1] < mesh_params["Ly"] - df.DOLFIN_EPS_LARGE)):
            x_mid = np.mean(x, axis=0)
            new_nodes.append(x_mid)
            for j in range(len(loop)):
                j_next = (j + 1) % len(loop)
                new_cells.append([loop[j], loop[j_next], i])
            i += 1
    
    new_nodes = np.array(new_nodes)
    new_cells = np.array(new_cells)
    Nextra = len(new_nodes)

    topology = np.vstack([topology, new_cells])
    geometry = np.vstack([geometry, new_nodes])

    triang = tri.Triangulation(
        geometry[:, 0], geometry[:, 1], triangles=topology)

    imgfolder = os.path.join(args.folder, "images")
    if not os.path.exists(imgfolder):
        os.makedirs(imgfolder)

    tL = np.zeros((len(timeseries), 2))
    for it, (t, c_path) in enumerate(timeseries):
        c = np.array(h5f[c_path[1]]).flatten()
        c = np.hstack([c, -100.*np.ones(Nextra)])

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        tcset = ax.tricontour(triang, c, levels=[0.5])

        paths = tcset.collections[0].get_paths()

        L = np.zeros(len(paths))
        touching_btm = np.zeros(len(paths), dtype=bool)
        for i, path in enumerate(paths):
            x = path.vertices
            L[i] = pathlength(x[:, 0], x[:, 1])
            touching_btm[i] = any(x[:, 1] < 1e-3)
            if args.plot:
                plt.scatter(x[:, 0], x[:, 1])

        Lm = sum(L[touching_btm]) if any(touching_btm) else max(L)
        
        tL[it, 0] = t
        tL[it, 1] = Lm
        if args.plot:
            ax.set_title("t = {:04.6f}".format(t))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.savefig(os.path.join(imgfolder, "front_{:06d}.png".format(it)))
            #plt.show()
        plt.close()

    np.savetxt(os.path.join(args.folder, "t_vs_L.dat"), tL)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(tL[:, 0], tL[:, 1]-1, label="data")
    ax.plot(tL[:, 0], 1000*tL[:, 0]**2, label='t^2')
    ax.plot(tL[:, 0], 100*tL[:, 0], label="t")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("front length")
    plt.legend()
    plt.show()