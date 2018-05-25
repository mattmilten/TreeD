# TreeD

### Draw a visual representation of the branch-and-cut tree of SCIP using spatial dissimilarities of the node LP solutions.

## [Example](TreeD_for_instance_lseu_generated_with_SCIP_5.0.1.html)

## Usage:
- run `TreeD.py` to get usage information

## Dependencies:
- [PySCIPOpt](https://github.com/SCIP-Interfaces/PySCIPOpt) to solve the instance and generate the necessary tree data
- Plot.ly to draw the 3D visualization
- [pandas](https://pandas.pydata.org/) to organize the collected data
- [sklearn](http://scikit-learn.org/stable/) for multi-dimensional scaling
- [pysal](https://github.com/pysal) to compute statistics based on spatial (dis)similarity
