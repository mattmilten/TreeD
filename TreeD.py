from pyscipopt import Model, Eventhdlr, quicksum, SCIP_EVENTTYPE
from sklearn import manifold
import pandas as pd
from plotly.graph_objs import *
from plotly.offline import plot, iplot
from plotly import tools
import pysal
import sys


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
        if node.getNumber() is not 1:
            parentnode = node.getParent()
            parent = parentnode.getNumber()
        else:
            parent = 1

        age = self.model.getNNodes()
        condition = self.model.getCondition()
        self.nodelist.append({'number': node.getNumber(),
                              'LPsol': LPsol,
                              'objval': objval,
                              'parent': parent,
                              'age': age,
                              'first': firstlp,
                              'condition': condition
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

    Dependencies:
     - PySCIPOpt to solve the instance and generate the necessary tree data
     - Plot.ly to draw the 3D visualization
     - pandas to organize the collected data
    """

    def __init__(self):
        self.mode = '3D'
        self.use_iplot = False
        self.colorcondition = False
        self.weights = 'knn'
        self.kernelfunction = 'triangular'
        self.knn_k = 2
        self.fig = None
        self.df = None
        self.div = None   # used for saving a plot.ly html object to be included as div
        self._symbol = []

    def performMDS(self):
        """compute multidimensional scaling of LP solution values"""
        df = pd.DataFrame(self.nodelist, columns = ['LPsol'])
        df = df['LPsol'].apply(pd.Series).fillna(value=0)
        if self.mode == '3D':
            mds = manifold.MDS(2)
            self.xy = mds.fit_transform(df)
        if self.mode == '2D':
            mds = manifold.MDS(1)
            self.x2 = mds.fit_transform(df)

    def performSpatialAnalysis(self):
        """compute spatial correlation between LP solutions and their condition numbers"""
        df = pd.DataFrame(self.nodelist, columns = ['LPsol', 'condition'])
        lpsols = df['LPsol'].apply(pd.Series).fillna(value=0)
        if self.weights == 'kernel':
            weights = pysal.Kernel(lpsols, function=self.kernelfunction)
        else:
            weights = pysal.knnW_from_array(lpsols, k=self.knn_k)
        self.moran = pysal.Moran(df['condition'].tolist(), weights)

    def generateEdges(self, separate_frames=False):
        """Generate edge information corresponding to parent information in df

        :param separate_frames: whether to generate separate edge sets for each node age
                                (used for tree growth animation)
        """

        if not self.mode in ['3D', '2D']:
            return
        # 3D edges
        Xe = []
        Ye = []
        Ze = []

        # 2D edges
        xe = []

        if not separate_frames:
            for index, curr in self.df.iterrows():
                if curr['first']:
                    self._symbol += ['circle']
                    # skip root node
                    if curr['number'] == 1:
                        continue
                    # found first LP solution of a new child node
                    # parent is last LP of parent node
                    parent = self.df[self.df['number'] == curr['parent']].iloc[-1]
                    # mark endpoint of branch for previous node
                    Xe += [None]
                    Ye += [None]
                    Ze += [None]
                    xe += [None]
                else:
                    # found an improving LP solution at the same node as before
                    self._symbol += ['diamond']
                    parent = self.df.iloc[index - 1]

                if self.mode in ['3D', 'combined']:
                    Xe += [float(parent['x']), curr['x']]
                    Ye += [float(parent['y']), curr['y']]
                if self.mode in ['2D', 'combined']:
                    xe += [float(parent['x2']), curr['x2']]
                Ze += [float(parent['objval']), curr['objval']]

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
                        Xe_ += [None]
                        Ye_ += [None]
                        Ze_ += [None]
                    else:
                        # found an improving LP solution at the same node as before
                        self._symbol += ['diamond']
                        parent = self.df.iloc[index - 1]

                    Xe_ += [float(parent['x']), curr['x']]
                    Ye_ += [float(parent['y']), curr['y']]
                    Ze_ += [float(parent['objval']), curr['objval']]
                Xe.append(Xe_)
                Ye.append(Ye_)
                Ze.append(Ze_)

        self.Xe = Xe
        self.Ye = Ye
        self.Ze = Ze
        self.xe = xe

    def draw(self):
        """Draw the tree, depending on the mode"""

        if self.colorcondition:
            color = self.df['condition']
            marker = Marker(symbol = self._symbol,
                            size = 4,
                            color = color,
                            colorscale = 'Portland',
                            colorbar = ColorBar(title='Condition', thickness=10, x=0),
                            # cmin = 1.0,
                            # cmax = 1e10
                            )
        else:
            color = self.df['age']
            marker = Marker(symbol = self._symbol,
                            size = 4,
                            color = color,
                            colorscale = 'Portland',
                            )

        self.generateEdges(separate_frames=False)

        if self.mode in ['3D']:
            node_object = Scatter3d(x = self.df['x'],
                                    y = self.df['y'],
                                    z = self.df['objval'],
                                    mode = 'markers+text',
                                    marker = marker,
                                    hovertext = self.df['number'],
                                    hoverinfo = 'z+text+name',
                                    opacity = 0.7,
                                    name = 'LP solutions'
                                   )

            proj_object = Scatter3d(x = self.df['x'],
                                    y = self.df['y'],
                                    z = self.df['objval'],
                                    mode = 'markers+text',
                                    marker = marker,
                                    hovertext = self.df['number'],
                                    hoverinfo = 'z+text+name',
                                    opacity = 0.0,
                                    projection = dict(z = dict(show = True)),
                                    name = 'projection of LP solutions'
                                   )

            edge_object = Scatter3d(x = self.Xe,
                                    y = self.Ye,
                                    z = self.Ze,
                                    mode = 'lines',
                                    line = Line(color = 'rgb(75,75,75)',
                                                width = 2
                                               ),
                                    hoverinfo = 'none',
                                    name = 'edges'
                                  )

            min_x = min(self.df['x'])
            max_x = max(self.df['x'])
            min_y = min(self.df['y'])
            max_y = max(self.df['y'])

            optval_object = Scatter3d(x = [min_x, min_x, max_x, max_x, min_x],
                                      y = [min_y, max_y, max_y, min_y, min_y],
                                      z = [self.optval] * 5,
                                      mode = 'lines',
                                      line = Line(color = 'rgb(0,200,0)',
                                                  width = 10
                                                 ),
                                      hoverinfo = 'name+z',
                                      name = 'optimal value',
                                      opacity = 0.3
                                     )

        if self.mode in ['2D']:
            node_object_2d = Scatter(x = self.df['x2'],
                                     y = self.df['objval'],
                                     mode = 'markers+text',
                                     marker = Marker(symbol = self._symbol,
                                                     size = 10,
                                                     color = color,
                                                     colorscale = 'Portland',
                                                    ),
                                     hovertext = self.df['number'],
                                     hoverinfo = 'y+text+name',
                                     name = 'LP solutions',
                                    )

            edge_object_2d = Scatter(x = self.xe,
                                     y = self.Ze,
                                     mode = 'lines',
                                     line = Line(color = 'rgb(75,75,75)',
                                                 width = 1
                                                ),
                                     hoverinfo = 'none',
                                     name = 'edges',
                                    )

            min_x = min(self.df['x2'])
            max_x = max(self.df['x2'])

            optval_object_2d = Scatter(x = [min_x, max_x],
                                       y = [self.optval] * 2,
                                       mode = 'lines',
                                       line = Line(color = 'rgb(0,200,0)',
                                                   width = 1
                                                  ),
                                       hoverinfo = 'name+z',
                                       name = 'optimal value',
                                      )

        xaxis = XAxis(showticklabels=False, title='X')
        yaxis = YAxis(showticklabels=False, title='Y')
        zaxis = ZAxis(title='obj value')
        scene = Scene(xaxis=xaxis, yaxis=yaxis, zaxis=zaxis)

        layout = Layout(title = 'TreeD for instance '+self.probname+', generated with '+self.scipversion,
                        autosize = True,
                        showlegend = True,
                        hovermode = 'closest',
                        scene = scene
                       )

        if self.mode == '3D':
            data = Data([node_object, proj_object, edge_object, optval_object])
            fig = Figure(data = data, layout = layout)
        elif self.mode == '2D':
            data = Data([node_object_2d, edge_object_2d, optval_object_2d])
            fig = Figure(data = data, layout = layout)
        else:
            print('deprecated mode: ', self.mode)
            return
            fig = tools.make_subplots(cols = 2,
#                                       subplot_titles = ('Multidimensional Scaling in 1D', ''),
                                      specs = [[{'is_3d': True}, {'is_3d': False}]],
                                      print_grid = False
                                     )
            fig.append_trace(node_object, 1, 1)
            fig.append_trace(proj_object, 1, 1)
            fig.append_trace(edge_object, 1, 1)
            fig.append_trace(optval_object, 1, 1)
            fig.append_trace(node_object_2d, 1, 2)
            fig.append_trace(edge_object_2d, 1, 2)
            fig.append_trace(optval_object_2d, 1, 2)
            fig['layout'].update(layout)

        self.fig = fig

        if self.use_iplot:
            iplot(fig, filename = layout.title.replace(' ', '_'))
        else:
            nicefilename = layout.title.replace(' ', '_')
            nicefilename = nicefilename.replace('"', '')
            nicefilename = nicefilename.replace(',', '')
            # generate html code to include into a website as <div>
            self.div = plot(fig, filename = nicefilename + '.html', show_link=False, include_plotlyjs=False, output_type='div')
            plot(fig, filename = nicefilename + '.html', show_link=False)


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

        print("solving instance "+self.probname)
        model.optimize()

        self.scipversion = 'SCIP '+str(model.version())
        self.scipversion = self.scipversion[:-1]+'.'+self.scipversion[-1]

        if model.getStatus() == 'optimal':
            self.optval = model.getObjVal()
        else:
            self.optval = None

        print("performing MDS to transform high dimensional LP solutions to 1D/2D")
        self.performMDS()

        print("performing Spatial Analysis on similarity of LP condition numbers")
        self.performSpatialAnalysis()

        print("storing all collected data in a DataFrame")
        self.df = pd.DataFrame(self.nodelist, columns = ['number', 'parent', 'objval', 'age', 'first', 'condition'])
        if self.mode in ['3D', 'combined']:
            coords = pd.DataFrame(self.xy, columns = ['x', 'y'])
            self.df = pd.merge(self.df, coords, left_index = True, right_index = True, how = 'outer')
        if self.mode in ['2D', 'combined']:
            coords_2d = pd.DataFrame(self.x2, columns = ['x2'])
            self.df = pd.merge(self.df, coords_2d, left_index = True, right_index = True, how = 'outer')


if __name__ == "__main__":

    treed = TreeD()
    if len(sys.argv) == 1:
        print(treed.__doc__)
        print("usage: {} <MIP-instance>".format(sys.argv[0]))
    else:
        treed.probpath = sys.argv[1]
        treed.main()
        treed.draw()
