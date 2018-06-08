# TreeD

### Visual representation of the branch-and-cut tree of SCIP using spatial dissimilarities of LP solutions

## [Example](https://plot.ly/~mattmilten/103/)

## Usage:
- run `TreeD.py` to get usage information

## Dependencies:
- [PySCIPOpt](https://github.com/SCIP-Interfaces/PySCIPOpt) to solve the instance and generate the necessary tree data
- [Plot.ly](https://plot.ly/) to draw the 3D visualization
- [pandas](https://pandas.pydata.org/) to organize the collected data
- [sklearn](http://scikit-learn.org/stable/) for multi-dimensional scaling
- [pysal](https://github.com/pysal) to compute statistics based on spatial (dis)similarity

## Usage with [Amira](https://amira.zib.de/):
- run `AmiraTreeD.py` to get usage information.

### `AmiraTreeD.py` generates the '.am' file to be loaded by Amira software to draw the tree using LineRaycast.
