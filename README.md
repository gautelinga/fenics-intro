# fenics-intro

To run the ADE problem in three steps:
```
python3 make_porous_mesh.py  # ! Only works in serial for now
mpiexec -n [NUM_CORES] python3 stokes.py
mpiexec -n [NUM_CORES] python3 ade_supg.py 
```
