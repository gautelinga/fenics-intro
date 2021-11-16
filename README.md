# fenics-intro

To run the ADE problem in three steps:
```
python3 make_porous_mesh.py [FOLDER] # ! Only works in serial for now
mpiexec -n [NUM_CORES] python3 stokes.py [FOLDER]
mpiexec -n [NUM_CORES] python3 ade_supg.py [FOLDER]  # Makes a new folder named 0, 1, 2... and so on --> [NUM]
python3 analyze.py [FOLDER] --plot
```
And to make a movie:
```
ffmpeg -f image2 -framerate 10 -i [FOLDER]/[NUM]/front_%06d.png [FOLDER]/[NUM]/front.mp4
```
