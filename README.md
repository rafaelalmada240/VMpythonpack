The code that is ready for use is in the release folder.

The available code is currently being improved. Any suggestions are welcome. 

## Structure of the release file

- vmpack - module where the VM implementation and auxiliary functions are implemented, fully in python. Both a serial and parallel implementation is currently possible through the multiprocessing python library;
- vmvideo -  additional module for video generation and visualization of simulation results with different visualization options;
- makenetwork2 - A file where a new network can be generated using Voronoi tesselation and a wound can be included.
- makenetmovie - A file where a video of a simulation can be made.
- runwh - A file where the simulation is actually implemented.

