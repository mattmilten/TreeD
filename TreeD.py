from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE
from sklearn import manifold
import pandas as pd
import plotly.graph_objs as go
import networkx as nx
import sys
import math


class LPstatEventhdlr(Eventhdlr):
    """PySCIPOpt Event handler to collect data on LP events."""

    transvars = {}

    def collectNodeInfo(self, firstlp = True):
        objval = self.model.getSolObjVal(None)
        if abs(objval) >= self.model.infinity():
            return

        LPsol = {}
        if self.transvars == {}:
            self.transvars = self.model.getVars(transformed = True)
        for var in self.transvars:
            solval = self.model.getSolVal(None, var)
            # store only solution values above 1e-6
            if abs(solval) > 1e-6:
                LPsol[var.name] = self.model.getSolVal(None, var)
        node = self.model.getCurrentNode()
        if node.getNumber() != 1:
            parentnode = node.getParent()
            parent = parentnode.getNumber()
        else:
            parent = 1
        depth = node.getDepth()

        age = self.model.getNNodes()
        condition = math.log10(self.model.getCondition())
        iters = self.model.lpiGetIterations()
        self.nodelist.append({'number': node.getNumber(),
                              'LPsol': LPsol,
                              'objval': objval,
                              'parent': parent,
                              'age': age,
                              'depth':depth,
                              'first': firstlp,
                              'condition': condition,
                              'iterations': iters
                             })

    def eventexec(self, event):
        if event.getType() == SCIP_EVENTTYPE.FIRSTLPSOLVED:
            self.collectNodeInfo(True)
        elif event.getType() == SCIP_EVENTTYPE.LPSOLVED:
            self.collectNodeInfo(False)
        else:
            print("unexpected event:" + str(event))
        return {}

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.LPEVENT, self)


class TreeD:
    """
    Draw a visual representation of the branch-and-cut tree of SCIP for
    a particular instance using spatial dissimilarities of the node LP solutions.

    Attributes:
        scip_settings (list of (str, value)): list of optional SCIP settings to use when solving the instance
        transformation (sr): type of transformation to generate 2D data points ('tsne, 'mds')
        showcuts (bool): whether to show nodes/solutions that originate from cutting rounds
        color (str): data to use for colorization of nodes ('age', 'depth', 'condition')
        colorscale (str): type of colorization, e.g. 'Viridis', 'Portland'
        colorbar (bool): whether to show the colorbar
        title (bool): show/hide title of the plot
        showlegend (bool): show/hide logend of the plot
        fontsize (str): fixed fontsize or 'auto'
        nodesize (int): size of tree nodes
        weights (str): type of weights for pysal, e.g. 'knn'
        kernelfunction (str): type of kernelfunction for distance metrics, e.g. 'triangular'
        knn_k (int): number of k-nearest neighbors
        fig (object): handler for generated figure
        df (Dataframe): storage of tree information
        div (str): html for saving a plot.ly object to be included as div

    Dependencies:
     - PySCIPOpt to solve the instance and generate the necessary tree data
     - Plot.ly to draw the 3D visualization
     - pandas to organize the collected data
    """

    def __init__(self):
        self.scip_settings = [('limits/totalnodes', 500)]
        self.transformation = 'mds'
        self.showcuts = True
        self.color = 'age'
        self.colorscale = 'Portland'
        self.colorbar = False
        self.title = True
        self.showlegend = True
        self.fontsize = None
        self.nodesize = 5
        self.weights = 'knn'
        self.kernelfunction = 'triangular'
        self.knn_k = 2
        self.fig = None
        self.df = None
        self.div = None
        self.include_plotlyjs = 'cdn'
        self.nxgraph = nx.Graph()
        self.stress = None
        self._symbol = []

    def transform(self):
        """compute transformations of LP solutions into 2-dimensional space"""
        # df = pd.DataFrame(self.nodelist, columns = ['LPsol'])
        df = self.df['LPsol'].apply(pd.Series).fillna(value=0)
        if self.transformation == 'tsne':
            mf = manifold.TSNE(n_components=2)
        else:
            mf = manifold.MDS(n_components=2) 
        self.xy = mf.fit_transform(df)
        self.stress = mf.stress_

        self.df['x'] = self.xy[:,0]
        self.df['y'] = self.xy[:,1]
        self._generateEdges()

    def performSpatialAnalysis(self):
        """compute spatial correlation between LP solutions and their condition numbers"""
        import pysal
        
        df = pd.DataFrame(self.nodelist, columns = ['LPsol', 'condition'])
        lpsols = df['LPsol'].apply(pd.Series).fillna(value=0)
        if self.weights == 'kernel':
            weights = pysal.Kernel(lpsols, function=self.kernelfunction)
        else:
            weights = pysal.knnW_from_array(lpsols, k=self.knn_k)
        self.moran = pysal.Moran(df['condition'].tolist(), weights)

    def _generateEdges(self, separate_frames=False):
        """Generate edge information corresponding to parent information in df

        :param separate_frames: whether to generate separate edge sets for each node age
                                (used for tree growth animation)
        """

        # 3D edges
        Xe = []
        Ye = []
        Ze = []

        symbol = []

        if not separate_frames:
            if self.showcuts:
                self.nxgraph.add_nodes_from(range(len(self.df)))
                for index, curr in self.df.iterrows():
                    if curr['first']:
                        symbol += ['circle']
                        # skip root node
                        if curr['number'] == 1:
                            continue
                        # found first LP solution of a new child node
                        # parent is last LP of parent node
                        parent = self.df[self.df['number'] == curr['parent']].iloc[-1]
                    else:
                        # found an improving LP solution at the same node as before
                        symbol += ['diamond']
                        parent = self.df.iloc[index - 1]

                    Xe += [float(parent['x']), curr['x'], None]
                    Ye += [float(parent['y']), curr['y'], None]
                    Ze += [float(parent['objval']), curr['objval'], None]
                    self.nxgraph.add_edge(parent.name, curr.name)
            else:
                self.nxgraph.add_nodes_from(list(self.df['number']))
                for index, curr in self.df.iterrows():
                    symbol += ['circle']
                    if curr['number'] == 1:
                        continue
                    parent = self.df[self.df['number'] == curr['parent']]
                    Xe += [float(parent['x']), curr['x'], None]
                    Ye += [float(parent['y']), curr['y'], None]
                    Ze += [float(parent['objval']), curr['objval'], None]
                    self.nxgraph.add_edge(parent.iloc[0]['number'], curr['number'])

        else:
            max_age = self.df['age'].max()
            for i in range(1, max_age + 1):
                tmp = self.df[self.df['age'] == i]
                Xe_ = []
                Ye_ = []
                Ze_ = []
                for index, curr in tmp.iterrows():
                    if curr['first']:
                        symbol += ['circle']
                        # skip root node
                        if curr['number'] == 1:
                            continue
                        # found first LP solution of a new child node
                        # parent is last LP of parent node
                        parent = self.df[self.df['number'] == curr['parent']].iloc[-1]
                    else:
                        # found an improving LP solution at the same node as before
                        symbol += ['diamond']
                        parent = self.df.iloc[index - 1]

                    Xe_ += [float(parent['x']), curr['x'], None]
                    Ye_ += [float(parent['y']), curr['y'], None]
                    Ze_ += [float(parent['objval']), curr['objval'], None]
                Xe.append(Xe_)
                Ye.append(Ye_)
                Ze.append(Ze_)

        self.df['symbol'] = symbol
        self.Xe = Xe
        self.Ye = Ye
        self.Ze = Ze

    def _create_nodes_and_projections(self):
        colorbar = go.scatter3d.marker.ColorBar(title='', thickness=10, x=0)
        marker = go.scatter3d.Marker(symbol = self.df['symbol'],
                        size = self.nodesize,
                        color = self.df['age'],
                        colorscale = self.colorscale,
                        colorbar = colorbar)
        node_object = go.Scatter3d(x = self.df['x'],
                                y = self.df['y'],
                                z = self.df['objval'],
                                mode = 'markers+text',
                                marker = marker,
                                hovertext = self.df['number'],
                                hovertemplate = 'LP obj: %{z}<br>node number: %{hovertext}<br>%{marker.color}',
                                hoverinfo = 'z+text+name',
                                opacity = 0.7,
                                name = 'LP solutions'                                
                                )
        proj_object = go.Scatter3d(x = self.df['x'],
                                y = self.df['y'],
                                z = self.df['objval'],
                                mode = 'markers+text',
                                marker = marker,
                                hovertext = self.df['number'],
                                hoverinfo = 'z+text+name',
                                opacity = 0.0,
                                projection = dict(z = dict(show = True)),
                                name = 'projection of LP solutions',
                                visible=False
                                )
        return node_object, proj_object

    def draw(self):
        """Draw the tree, depending on the mode"""

        self.transform()

        nodes, nodeprojs = self._create_nodes_and_projections()

        edges = go.Scatter3d(x = self.Xe,
                                y = self.Ye,
                                z = self.Ze,
                                mode = 'lines',
                                line = go.scatter3d.Line(color = 'rgb(75,75,75)',
                                                    width = 2
                                                    ),
                                hoverinfo = 'none',
                                name = 'edges'
                                )

        min_x = min(self.df['x'])
        max_x = max(self.df['x'])
        min_y = min(self.df['y'])
        max_y = max(self.df['y'])

        optval = go.Scatter3d(x = [min_x, min_x, max_x, max_x, min_x],
                                    y = [min_y, max_y, max_y, min_y, min_y],
                                    z = [self.optval] * 5,
                                    mode = 'lines',
                                    line = go.scatter3d.Line(color = 'rgb(0,200,50)',
                                                        width = 5
                                                        ),
                                    hoverinfo = 'name+z',
                                    name = 'optimal value',
                                    opacity = 0.5
                                    )

        xaxis = go.layout.scene.XAxis(showticklabels=False, title='X', backgroundcolor='white', gridcolor='lightgray')
        yaxis = go.layout.scene.YAxis(showticklabels=False, title='Y', backgroundcolor='white', gridcolor='lightgray')
        zaxis = go.layout.scene.ZAxis(title='objective value', backgroundcolor='white', gridcolor='lightgray')
        scene = go.layout.Scene(xaxis=xaxis, yaxis=yaxis, zaxis=zaxis)
        title = 'TreeD of '+self.probname+', using '+self.scipversion if self.title else ''

        layout = go.Layout(title=title,
                        font=dict(size=self.fontsize),
                        autosize=True,
                        # width=900,
                        # height=600,
                        showlegend=self.showlegend,
                        hovermode='closest',
                        scene=scene
                       )

        updatemenus=list([
            dict(
                buttons=list([
                    dict(
                        args=['marker.color', [self.df['age']]],
                        label='Node Age',
                        method='restyle'
                    ),
                    dict(
                        args=['marker.color', [self.df['depth']]],
                        label='Tree Depth',
                        method='restyle'
                    ),
                    dict(
                        args=['marker.color', [self.df['condition']]],
                        label='LP Condition (log 10)',
                        method='restyle'
                    ),
                    dict(
                        args=['marker.color', [self.df['iterations']]],
                        label='LP Iterations',
                        method='restyle'
                    )
                ]),
                direction = 'down',
                showactive = True,
                type = 'buttons',
                # x = 1.2,
                # y = 0.6,
            ),
        ])

        layout['updatemenus'] = updatemenus

        self.fig = go.Figure(data = [nodes, nodeprojs, edges, optval], layout = layout)

        nicefilename = layout.title.text.replace(' ', '_')
        nicefilename = nicefilename.replace('"', '')
        nicefilename = nicefilename.replace(',', '')

        self.fig.write_html(file = nicefilename + '.html', include_plotlyjs=self.include_plotlyjs)

        # generate html code to include into a website as <div>
        self.div = self.fig.write_html(file = nicefilename + '.html', include_plotlyjs=self.include_plotlyjs, full_html=False)

        return self.fig


    def main(self):
        """Solve the instance and collect and generate the tree data"""

        self.nodelist = []

        self.probname = self.probpath.split('/')[-1].rstrip('.mps.lp.gz')

        model = Model("TreeD")
        eventhdlr = LPstatEventhdlr()
        eventhdlr.nodelist = self.nodelist
        model.includeEventhdlr(eventhdlr, "LPstat", "generate LP statistics after every LP event")
        model.readProblem(self.probpath)
        model.setIntParam('presolving/maxrestarts', 0)

        for setting in self.scip_settings:
            model.setParam(setting[0], setting[1])

        model.optimize()

        self.scipversion = 'SCIP '+str(model.version())
        # self.scipversion = self.scipversion[:-1]+'.'+self.scipversion[-1]

        if model.getStatus() == 'optimal':
            self.optval = model.getObjVal()
        else:
            self.optval = None


        # print("performing Spatial Analysis on similarity of LP condition numbers")
        # self.performSpatialAnalysis()

        columns = self.nodelist[0].keys()
        self.df = pd.DataFrame(self.nodelist, columns = columns)

        # merge solutions from cutting rounds into one node
        if not self.showcuts:
            self.df = self.df[self.df['first'] == False].drop_duplicates(subset='age', keep='last').reset_index()

    def compute_distances(self):
        """compute all pairwise distances between the original LP solutions and the transformed points"""
        if self.df is None:
            return

        self.origdist = []
        self.transdist = []
        for i in range(len(self.df)):
            for j in range(i+1, len(self.df)):
                self.origdist.append(distance(self.df['LPsol'].iloc[i], self.df['LPsol'].iloc[j]))
                self.transdist.append(distance(self.df[['x', 'y']].iloc[i], self.df[['x', 'y']].iloc[j]))


def distance(p1, p2):
    """euclidean distance between two coordinates (dict-like storage)"""
    dist = 0
    for k in set([*p1.keys(), *p2.keys()]):
        dist += (p1.get(k, 0) - p2.get(k, 0))**2
    return math.sqrt(dist)

if __name__ == "__main__":

    treed = TreeD()
    if len(sys.argv) == 1:
        print(treed.__doc__)
        print("usage: {} <MIP-instance>".format(sys.argv[0]))
    else:
        treed.probpath = sys.argv[1]
        treed.main()
        treed.draw()
