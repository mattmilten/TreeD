# TreeD

### Visual representation of the branch-and-cut tree of SCIP using spatial dissimilarities of LP solutions -- [Interactive Example](http://www.zib.de/miltenberger/treed-showcase.html)

[![Example](res/treed-example.png)](https://plot.ly/~mattmilten/103/)
## Installation

```
python -m pip install treed
```

## Usage
- run Python script `bin/treed` (will be installed into your PATH on Linux/macOS when using `pip install treed`) to get usage information or use this code snippet in a Jupyter notebook:

```
from treed import TreeD

treed = TreeD(
    probpath="model.mps",
    nodelimit=20,
    transformation='mds',
    showcuts=True
)

treed.solve()
fig = treed.draw()
fig.show(renderer='notebook')
```

## Dependencies
- [PySCIPOpt](https://github.com/scipopt/PySCIPOpt) to solve the instance and generate the necessary tree data
- [Plotly](https://plot.ly/) to draw the 3D visualization
- [pandas](https://pandas.pydata.org/) to organize the collected data
- [sklearn](http://scikit-learn.org/stable/) for multi-dimensional scaling
- [pysal](https://github.com/pysal) to compute statistics based on spatial (dis)similarity; this is optional

## Export to [Amira](https://amira.zib.de/)
- run `AmiraTreeD.py` to get usage information.

`AmiraTreeD.py` generates the '.am' data files to be loaded by Amira software to draw the tree using LineRaycast.

### Settings

![Project View](res/ProjectView.png)

- `DataTree.am`: SpatialGraph data file with tree nodes and edges.
- `LineRaycast`: Module to display the SpatialGraph. Note that is needed to set the colormap according to py code output (For instance 'Color map from 1 to 70' in this picture).
- `DataOpt.am`: SpatialGraph data file with optimun value.
- `Opt Plane`: Display the optimal value as a plane.

### Preview

![Amira preview](res/AmiraTree.gif)
