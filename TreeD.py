from pyscipopt import Model, Eventhdlr, quicksum, SCIP_EVENTTYPE
from sklearn import manifold
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot, iplot
from plotly import tools
# import pysal
import sys
import math
import networkx as nx


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
        use_iplot (bool): whether to plot inline in a notebook
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
        self.scip_settings = [('limits/totalnodes', 1000)]
        self.use_iplot = False
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

        if not separate_frames:
            if self.showcuts:
                self.nxgraph.add_nodes_from(range(len(self.df)))
                for index, curr in self.df.iterrows():
                    if curr['first']:
                        self._symbol += ['circle']
                        # skip root node
                        if curr['number'] == 1:
                            continue
                        # found first LP solution of a new child node
                        # parent is last LP of parent node
                        parent = self.df[self.df['number'] == curr['parent']].iloc[-1]
                    else:
                        # found an improving LP solution at the same node as before
                        self._symbol += ['diamond']
                        parent = self.df.iloc[index - 1]

                    Xe += [float(parent['x']), curr['x'], None]
                    Ye += [float(parent['y']), curr['y'], None]
                    Ze += [float(parent['objval']), curr['objval'], None]
                    self.nxgraph.add_edge(parent.name, curr.name)
            else:
                self.nxgraph.add_nodes_from(list(self.df['number']))
                for index, curr in self.df.iterrows():
                    self._symbol += ['circle']
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
                        self._symbol += ['circle']
                        # skip root node
                        if curr['number'] == 1:
                            continue
                        # found first LP solution of a new child node
                        # parent is last LP of parent node
                        parent = self.df[self.df['number'] == curr['parent']].iloc[-1]
                    else:
                        # found an improving LP solution at the same node as before
                        self._symbol += ['diamond']
                        parent = self.df.iloc[index - 1]

                    Xe_ += [float(parent['x']), curr['x'], None]
                    Ye_ += [float(parent['y']), curr['y'], None]
                    Ze_ += [float(parent['objval']), curr['objval'], None]
                Xe.append(Xe_)
                Ye.append(Ye_)
                Ze.append(Ze_)

        self.Xe = Xe
        self.Ye = Ye
        self.Ze = Ze

    def _create_nodes_and_projections(self, nodetype='age'):
        colorbar = go.scatter3d.marker.ColorBar(title=nodetype.capitalize(), thickness=10, x=0)
        marker = go.scatter3d.Marker(symbol = self._symbol,
                        size = self.nodesize,
                        color = self.df[nodetype],
                        colorscale = self.colorscale,
                        colorbar = colorbar)
        node_object = go.Scatter3d(x = self.df['x'],
                                y = self.df['y'],
                                z = self.df['objval'],
                                mode = 'markers+text',
                                marker = marker,
                                hovertext = self.df['number'],
                                hoverinfo = 'z+text+name',
                                opacity = 0.7,
                                name = 'LP solutions',
                                visible = True if nodetype == 'age' else False
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
                                visible = True if nodetype == 'age' else False
                                )
        return node_object, proj_object

    def draw(self):
        """Draw the tree, depending on the mode"""

        self.transform()

        node_object_age, proj_object_age = self._create_nodes_and_projections(nodetype='age')
        node_object_depth, proj_object_depth = self._create_nodes_and_projections(nodetype='depth')
        node_object_cond, proj_object_cond = self._create_nodes_and_projections(nodetype='condition')
        node_object_iter, proj_object_iter = self._create_nodes_and_projections(nodetype='iterations')

        edge_object = go.Scatter3d(x = self.Xe,
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

        optval_object = go.Scatter3d(x = [min_x, min_x, max_x, max_x, min_x],
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
        title = 'TreeD of '+self.probname+', generated with '+self.scipversion if self.title else ''

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
                        args=['visible', [True, True, False, False, False, False, False, False, True, True]],
                        label='Node Age',
                        method='restyle'
                    ),
                    dict(
                        args=['visible', [False, False, True, True, False, False, False, False, True, True]],
                        label='Tree Depth',
                        method='restyle'
                    ),
                    dict(
                        args=['visible', [False, False, False, False, True, True, False, False, True, True]],
                        label='LP Condition (log 10)',
                        method='restyle'
                    ),
                    dict(
                        args=['visible', [False, False, False, False, False, False, True, True, True, True]],
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

        # annotations = list([
        #     dict(text='Color Mode:', showarrow=False, x=1.2, y=0.6, )
        # ])
        # layout['annotations'] = annotations
        layout['updatemenus'] = updatemenus

        data = [node_object_age, proj_object_age,
                node_object_depth, proj_object_depth,
                node_object_cond, proj_object_cond,
                node_object_iter, proj_object_iter,
                edge_object, optval_object]
        self.fig = go.Figure(data = data, layout = layout)

        nicefilename = layout.title.text.replace(' ', '_')
        nicefilename = nicefilename.replace('"', '')
        nicefilename = nicefilename.replace(',', '')

        if not self.use_iplot:
            plot(self.fig, filename = nicefilename + '.html', show_link=False, include_plotlyjs=self.include_plotlyjs)

        # generate html code to include into a website as <div>
        self.div = plot(self.fig, filename = nicefilename + '.html', show_link=False, include_plotlyjs=self.include_plotlyjs, output_type='div')


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
