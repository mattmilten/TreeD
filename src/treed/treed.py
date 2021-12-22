from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE
from sklearn import manifold
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import networkx as nx
import os
import math
from time import time


class LPstatEventhdlr(Eventhdlr):
    """PySCIPOpt Event handler to collect data on LP events."""

    transvars = {}

    def collectNodeInfo(self, firstlp=True):
        objval = self.model.getSolObjVal(None)
        if abs(objval) >= self.model.infinity():
            return

        LPsol = {}
        if self.transvars == {}:
            self.transvars = self.model.getVars(transformed=True)
        for var in self.transvars:
            solval = self.model.getSolVal(None, var)
            # store only solution values above 1e-6
            if abs(solval) > 1e-6:
                LPsol[var.name] = self.model.getSolVal(None, var)

        # skip duplicate nodes
        # if self.nodelist and LPsol == self.nodelist[-1].get("LPsol"):
        #     return
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
        pb = self.model.getPrimalbound()
        if pb >= self.model.infinity():
            pb = None

        nodedict = {
            "number": node.getNumber(),
            "LPsol": LPsol,
            "objval": objval,
            "parent": parent,
            "age": age,
            "depth": depth,
            "first": firstlp,
            "condition": condition,
            "iterations": iters,
            # "variables": self.model.getNVars(),
            # "constraints": self.model.getNConss(),
            "rows": self.model.getNLPRows(),
            "primalbound": pb,
            "dualbound": self.model.getDualbound(),
            "time": self.model.getSolvingTime()
        }
        # skip 0-iterations LPs (duplicates?)
        if firstlp:
            self.nodelist.append(nodedict)
        elif iters > 0:
            prevevent = self.nodelist[-1]
            if nodedict["number"] == prevevent["number"] and not prevevent["first"]:
                # overwrite data from previous LP event
                self.nodelist[-1] = nodedict
            else:
                self.nodelist.append(nodedict)

    def eventexec(self, event):
        if event.getType() == SCIP_EVENTTYPE.FIRSTLPSOLVED:
            self.collectNodeInfo(firstlp=True)
        elif event.getType() == SCIP_EVENTTYPE.LPSOLVED:
            self.collectNodeInfo(firstlp=False)
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

    def __init__(self, **kwargs):
        self.probpath = kwargs.get("probpath", "")
        self.scip_settings = [("limits/totalnodes", kwargs.get("nodelimit", 500))]
        self.setfile = kwargs.get("setfile", None)
        self.transformation = kwargs.get("transformation", "mds")
        self.showcuts = kwargs.get("showcuts", True)
        self.verbose = kwargs.get("verbose", True)
        self.color = "age"
        self.colorscale = "Portland"
        self.colorbar = kwargs.get("colorbar", True)
        self.title = kwargs.get("title", True)
        self.showlegend = kwargs.get("showlegend", True)
        self.showbuttons = kwargs.get("showbuttons", True)
        self.showslider = kwargs.get("showslider", True)
        self.fontsize = None
        self.nodesize = 5
        self.weights = "knn"
        self.kernelfunction = "triangular"
        self.knn_k = 2
        self.fig = None
        self.df = None
        self.div = None
        self.include_plotlyjs = "cdn"
        self.nxgraph = nx.Graph()
        self.stress = None
        self.start_frame = kwargs.get("start_frame", 1)

    def transform(self):
        """compute transformations of LP solutions into 2-dimensional space"""
        # df = pd.DataFrame(self.nodelist, columns = ['LPsol'])
        df = self.df["LPsol"].apply(pd.Series).fillna(value=0)
        if self.transformation == "tsne":
            mf = manifold.TSNE(n_components=2)
        elif self.transformation == "lle":
            mf = manifold.LocallyLinearEmbedding(n_components=2)
        elif self.transformation == "ltsa":
            mf = manifold.LocallyLinearEmbedding(n_components=2, method="ltsa")
        elif self.transformation == "spectral":
            mf = manifold.SpectralEmbedding(n_components=2)
        else:
            mf = manifold.MDS(n_components=2)

        if self.verbose:
            print("transforming LP solutions", end="...")
            start = time()
        start = time()
        xy = mf.fit_transform(df)
        if self.verbose:
            print(f"✔, time: {time()-start:.2f} seconds")

        try:
            self.stress = mf.stress_  # not available with all transformations
        except:
            print("no stress information available for {self.transformation} transformation")

        self.df["x"] = xy[:, 0]
        self.df["y"] = xy[:, 1]

    # def performSpatialAnalysis(self):
    #     """compute spatial correlation between LP solutions and their condition numbers"""
    #     import pysal

    #     df = pd.DataFrame(self.nodelist, columns=["LPsol", "condition"])
    #     lpsols = df["LPsol"].apply(pd.Series).fillna(value=0)
    #     if self.weights == "kernel":
    #         weights = pysal.Kernel(lpsols, function=self.kernelfunction)
    #     else:
    #         weights = pysal.knnW_from_array(lpsols, k=self.knn_k)
    #     self.moran = pysal.Moran(df["condition"].tolist(), weights)

    def _generateEdges(self, separate_frames=False):
        """Generate edge information corresponding to parent information in df

        :param separate_frames: whether to generate separate edge sets for each node age
                                (used for tree growth animation)
        """

        # 3D edges
        Xe = []
        Ye = []
        Ze = []

        if not "x" in self.df or not "y" in self.df:
            self.df["x"] = 0
            self.df["y"] = 0

        symbol = []

        if self.showcuts:
            self.nxgraph.add_nodes_from(self.df["id"])
            for index, curr in self.df.iterrows():
                if curr["first"]:
                    symbol += ["circle"]
                    # skip root node
                    if curr["number"] == 1:
                        continue
                    # found first LP solution of a new child node
                    # parent is last LP of parent node
                    parent = self.df[self.df["number"] == curr["parent"]].iloc[-1]
                else:
                    # found an improving LP solution at the same node as before
                    symbol += ["diamond"]
                    parent = self.df.iloc[index - 1]

                Xe += [parent["x"], curr["x"], None]
                Ye += [parent["y"], curr["y"], None]
                Ze += [parent["objval"], curr["objval"], None]
                self.nxgraph.add_edge(parent["id"], curr["id"])
        else:
            self.nxgraph.add_nodes_from(self.df["id"])
            for index, curr in self.df.iterrows():
                symbol += ["circle"]
                if curr["number"] == 1:
                    continue
                parent = self.df[self.df["number"] == curr["parent"]].iloc[-1]
                Xe += [parent["x"], curr["x"], None]
                Ye += [parent["y"], curr["y"], None]
                Ze += [parent["objval"], curr["objval"], None]
                self.nxgraph.add_edge(parent["id"], curr["id"])

        self.df["symbol"] = symbol
        self.Xe = Xe
        self.Ye = Ye
        self.Ze = Ze

    def _create_nodes_and_projections(self):
        colorbar = go.scatter3d.marker.ColorBar(title="", thickness=10, x=-0.05)
        marker = go.scatter3d.Marker(
            symbol=self.df["symbol"],
            size=self.nodesize,
            color=self.df["age"],
            colorscale=self.colorscale,
            colorbar=colorbar if self.colorbar else None,
        )
        node_object = go.Scatter3d(
            x=self.df["x"],
            y=self.df["y"],
            z=self.df["objval"],
            mode="markers+text",
            marker=marker,
            hovertext=self.df["number"],
            hovertemplate="LP obj: %{z}<br>node number: %{hovertext}<br>%{marker.color}",
            hoverinfo="z+text+name",
            opacity=0.7,
            name="LP solutions",
        )
        proj_object = go.Scatter3d(
            x=self.df["x"],
            y=self.df["y"],
            z=self.df["objval"],
            mode="markers+text",
            marker=marker,
            hovertext=self.df["number"],
            hoverinfo="z+text+name",
            opacity=0.0,
            projection=dict(z=dict(show=True)),
            name="projection of LP solutions",
            visible="legendonly",
        )
        return node_object, proj_object

    def _create_nodes_frames(self):
        colorbar = go.scatter3d.marker.ColorBar(title="", thickness=10, x=-0.05)
        marker = go.scatter3d.Marker(
            symbol=self.df["symbol"],
            size=self.nodesize,
            color=self.df["age"],
            colorscale=self.colorscale,
            colorbar=colorbar,
        )

        frames = []
        sliders_dict = dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue={"prefix": "Age:", "visible": True, "xanchor": "right",},
            len=0.9,
            x=0.05,
            y=0.1,
            steps=[],
        )

        # get start and end points for bound line plots
        min_x = min(self.df["x"])
        max_x = max(self.df["x"])
        min_y = min(self.df["y"])
        max_y = max(self.df["y"])

        maxage = max(self.df["age"])
        for a in np.linspace(1, maxage, min(200, maxage)):
            a = int(a)
            adf = self.df[self.df["age"] <= a]
            node_object = go.Scatter3d(
                x=adf["x"],
                y=adf["y"],
                z=adf["objval"],
                mode="markers+text",
                marker=marker,
                hovertext=adf["number"],
                # hovertemplate="LP obj: %{z}<br>node number: %{hovertext}<br>%{marker.color}",
                hoverinfo="z+text+name",
                opacity=0.7,
                name="LP Solutions",
            )

            primalbound = go.Scatter3d(
                x=[min_x, min_x, max_x, max_x, min_x],
                y=[min_y, max_y, max_y, min_y, min_y],
                z=[adf["primalbound"].iloc[-1]] * 5,
                mode="lines",
                line=go.scatter3d.Line(width=5),
                hoverinfo="name+z",
                name="Primal Bound",
                opacity=0.5,
            )
            dualbound = go.Scatter3d(
                x=[min_x, min_x, max_x, max_x, min_x],
                y=[min_y, max_y, max_y, min_y, min_y],
                z=[adf["dualbound"].iloc[-1]] * 5,
                mode="lines",
                line=go.scatter3d.Line(width=5),
                hoverinfo="name+z",
                name="Dual Bound",
                opacity=0.5,
            )

            frames.append(
                go.Frame(data=[node_object, primalbound, dualbound], name=str(a))
            )

            slider_step = {
                "args": [
                    [a],
                    {
                        "frame": {"redraw": True, "restyle": False},
                        "fromcurrent": True,
                        "mode": "immediate",
                    },
                ],
                "label": a,
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

        return frames, sliders_dict

    def _create_nodes_frames_2d(self):
        colorbar = go.scatter.marker.ColorBar(title="", thickness=10, x=-0.05)
        marker = go.scatter.Marker(
            symbol=self.df["symbol"],
            size=self.nodesize * 2,
            color=self.df["age"],
            colorscale=self.colorscale,
            colorbar=colorbar,
        )

        frames = []
        sliders_dict = dict(
            active=self.start_frame-1,
            yanchor="top",
            xanchor="left",
            currentvalue={"prefix": "Age:", "visible": True, "xanchor": "right",},
            len=0.9,
            x=0.05,
            y=0.1,
            steps=[],
        )

        # get start and end points for bound line plots
        xmin = min([self.pos2d[k][0] for k in self.pos2d])
        xmax = max([self.pos2d[k][0] for k in self.pos2d])

        maxage = max(self.df["age"])
        for a in np.linspace(1, maxage, min(200, maxage)):
            a = int(a)
            adf = self.df[self.df["age"] <= a]
            node_object = go.Scatter(
                x=[self.pos2d[k][0] for k in adf["id"]],
                # y=[self.pos2d[k][1] for k in adf["id"]],
                y=[self.df["objval"][k] for k in adf["id"]],
                mode="markers",
                marker=marker,
                # hovertext=[
                #     f"LP obj: {adf['objval'].iloc[i]:.3f}\
                #     <br>node number: {adf['number'].iloc[i]}\
                #     <br>node age: {adf['age'].iloc[i]}\
                #     <br>depth: {adf['depth'].iloc[i]}\
                #     <br>LP cond: {adf['condition'].iloc[i]:.1f}\
                #     <br>iterations: {adf['iterations'].iloc[i]}"
                #     for i in range(len(adf))
                # ],
                hoverinfo="text+name",
                opacity=0.7,
                name="LP Solutions",
            )
            primalbound = go.Scatter(
                x=[xmin, xmax],
                y=2 * [adf["primalbound"].iloc[-1]],
                mode="lines",
                opacity=0.5,
                name="Primal Bound",
            )
            dualbound = go.Scatter(
                x=[xmin, xmax],
                y=2 * [adf["dualbound"].iloc[-1]],
                mode="lines",
                opacity=0.5,
                name="Dual Bound",
            )

            frames.append(
                go.Frame(data=[node_object, primalbound, dualbound], name=str(a))
            )

            slider_step = {
                "args": [
                    [a],
                    {
                        "frame": {"redraw": True, "restyle": False},
                        "fromcurrent": True,
                        "mode": "immediate",
                    },
                ],
                "label": a,
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

        return frames, sliders_dict

    def updatemenus(self):
        return list(
            [
                dict(
                    buttons=list(
                        [
                            dict(
                                label="Node Age",
                                method="restyle",
                                args=[
                                    {
                                        "marker.color": [self.df["age"]],
                                        "marker.cauto": min(self.df["age"]),
                                        "marker.cmax": max(self.df["age"]),
                                    }
                                ],
                            ),
                            dict(
                                label="Tree Depth",
                                method="restyle",
                                args=[
                                    {
                                        "marker.color": [self.df["depth"]],
                                        "marker.cauto": min(self.df["depth"]),
                                        "marker.cmax": max(self.df["depth"]),
                                    }
                                ],
                            ),
                            dict(
                                label="LP Condition (log 10)",
                                method="restyle",
                                args=[
                                    {
                                        "marker.color": [self.df["condition"]],
                                        "marker.cmin": 1,
                                        "marker.cmax": 20,
                                    }
                                ],
                            ),
                            dict(
                                label="LP Iterations",
                                method="restyle",
                                args=[
                                    {
                                        "marker.color": [self.df["iterations"]],
                                        "marker.cauto": min(self.df["iterations"]),
                                        "marker.cmax": max(self.df["iterations"]),
                                    }
                                ],
                            ),
                            # dict(
                            #     label="logarithmic obj",
                            #     method="relayout",
                            #     args=[
                            #         {
                            #             "yaxis.type": 'log'
                            #         }
                            #     ],
                            # ),
                            # dict(
                            #     label="linear obj",
                            #     method="relayout",
                            #     args=[
                            #         {
                            #             "yaxis.type": 'linear'
                            #         }
                            #     ],
                            # ),
                        ]
                    ),
                    direction="down",
                    showactive=True,
                    type="buttons",
                ),
                dict(
                    buttons=list(
                        [
                            dict(
                                label="▶",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": 50, "redraw": True,},
                                        "fromcurrent": True,
                                    },
                                ],
                            ),
                            dict(
                                label="◼",
                                method="animate",
                                args=[
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                        ]
                    ),
                    direction="left",
                    yanchor="top",
                    xanchor="right",
                    showactive=True,
                    type="buttons",
                    x=0,
                    y=0,
                ),
            ]
        )

    def draw(self):
        """Draw the tree, depending on the mode"""

        self.transform()

        if self.verbose:
            print("generating 3D objects", end="...")
            start = time()
        self._generateEdges()
        nodes, nodeprojs = self._create_nodes_and_projections()
        frames, sliders = self._create_nodes_frames()

        edges = go.Scatter3d(
            x=self.Xe,
            y=self.Ye,
            z=self.Ze,
            mode="lines",
            line=go.scatter3d.Line(color="rgb(75,75,75)", width=2),
            hoverinfo="none",
            name="Edges",
        )

        min_x = min(self.df["x"])
        max_x = max(self.df["x"])
        min_y = min(self.df["y"])
        max_y = max(self.df["y"])

        primalbound = go.Scatter3d(
            x=[min_x, min_x, max_x, max_x, min_x],
            y=[min_y, max_y, max_y, min_y, min_y],
            z=5 * [self.df["primalbound"].iloc[-1]],
            mode="lines",
            line=go.scatter3d.Line(width=5),
            hoverinfo="name+z",
            opacity=0.5,
            name="Primal Bound",
        )
        dualbound = go.Scatter3d(
            x=[min_x, min_x, max_x, max_x, min_x],
            y=[min_y, max_y, max_y, min_y, min_y],
            z=5 * [self.df["dualbound"].iloc[-1]],
            mode="lines",
            line=go.scatter3d.Line(width=5),
            hoverinfo="name+z",
            opacity=0.5,
            name="Dual Bound",
        )
        optval = go.Scatter3d(
            x=[min_x, min_x, max_x, max_x, min_x],
            y=[min_y, max_y, max_y, min_y, min_y],
            z=[self.optval] * 5,
            mode="lines",
            line=go.scatter3d.Line(width=5),
            hoverinfo="name+z",
            name="Optimum",
            opacity=0.5,
        )

        xaxis = go.layout.scene.XAxis(
            showticklabels=False,
            title="X",
            backgroundcolor="white",
            gridcolor="lightgray",
        )
        yaxis = go.layout.scene.YAxis(
            showticklabels=False,
            title="Y",
            backgroundcolor="white",
            gridcolor="lightgray",
        )
        zaxis = go.layout.scene.ZAxis(
            title="Objective value", backgroundcolor="white", gridcolor="lightgray"
        )
        scene = go.layout.Scene(xaxis=xaxis, yaxis=yaxis, zaxis=zaxis)

        if self.title:
            title = f"TreeD: {self.probname} ({self.scipversion}, {self.status})"
        else:
            title = ""

        filename = "TreeD_" + self.probname + ".html"

        camera = dict(eye=dict(x=1.5, y=1.3, z=0.5))
        layout = go.Layout(
            title=title,
            font=dict(size=self.fontsize),
            font_family="Fira Sans",
            autosize=True,
            # width=900,
            # height=600,
            showlegend=self.showlegend,
            hovermode="closest",
            scene=scene,
            scene_camera=camera,
            template="none",
        )

        if self.showbuttons:
            layout["updatemenus"] = self.updatemenus()
        if self.showslider:
            layout["sliders"] = [sliders]
        
        self.fig = go.Figure(
            data=[nodes, primalbound, dualbound, optval, nodeprojs, edges],
            layout=layout,
            frames=frames,
        )

        self.fig.write_html(file=filename, include_plotlyjs=self.include_plotlyjs)

        # generate html code to include into a website as <div>
        # self.div = self.fig.write_html(
        #     file=filename, include_plotlyjs=self.include_plotlyjs, full_html=False
        # )

        if self.verbose:
            print(f"✔, time: {time()-start:.2f} seconds")

        return self.fig

    def draw2d(self):
        """Draw the 2D tree"""
        self._generateEdges()
        self.hierarchy_pos()
        frames, sliders = self._create_nodes_frames_2d()

        start_frame = self.df[self.df["age"] <= self.start_frame]

        Xv = [self.pos2d[k][0] for k in self.df["id"]]
        # Yv = [self.pos2d[k][1] for k in self.df["id"]]
        # Yv = self.df["objval"]
        Xed = []
        Yed = []
        for edge in self.nxgraph.edges:
            Xed += [self.pos2d[edge[0]][0], self.pos2d[edge[1]][0], None]
            # Yed += [self.pos2d[edge[0]][1], self.pos2d[edge[1]][1], None]
            Yed += [self.df["objval"][edge[0]], self.df["objval"][edge[1]], None]

        colorbar = go.scatter.marker.ColorBar(title="", thickness=10, x=-0.05)
        marker = go.scatter.Marker(
            symbol=self.df["symbol"],
            size=self.nodesize * 2,
            color=self.df["age"],
            colorscale=self.colorscale,
            colorbar=colorbar if self.colorbar else None,
        )

        edges = go.Scatter(
            x=Xed,
            y=Yed,
            mode="lines",
            line=dict(color="rgb(75,75,75)", width=1),
            hoverinfo="none",
            name="Edges",
        )
        nodes = go.Scatter(
            x=[self.pos2d[k][0] for k in start_frame["id"]],
            y=start_frame["objval"],
            name="LP solutions",
            mode="markers",
            marker=marker,
            hovertext=[
                f"LP obj: {self.df['objval'].iloc[i]:.3f}\
                <br>node number: {self.df['number'].iloc[i]}\
                <br>node age: {self.df['age'].iloc[i]}\
                <br>depth: {self.df['depth'].iloc[i]}\
                <br>LP cond: {self.df['condition'].iloc[i]:.1f}\
                <br>iterations: {self.df['iterations'].iloc[i]}"
                for i in range(len(self.df))
            ],
            hoverinfo="text+name",
        )

        xmin = min(Xv)
        xmax = max(Xv)
        primalbound = go.Scatter(
            x=[xmin, xmax],
            y=2 * [start_frame["primalbound"].iloc[-1]],
            mode="lines",
            opacity=0.5,
            name="Primal Bound",
        )
        dualbound = go.Scatter(
            x=[xmin, xmax],
            y=2 * [start_frame["dualbound"].iloc[-1]],
            mode="lines",
            opacity=0.5,
            name="Dual Bound",
        )
        optval = go.Scatter(
            x=[xmin, xmax],
            y=2 * [self.optval],
            mode="lines",
            opacity=0.5,
            name="Optimum",
        )

        margin = 0.05 * xmax
        xaxis = go.layout.XAxis(
            title="",
            visible=False,
            range=[xmin - margin, xmax + margin],
            autorange=False,
        )
        yaxis = go.layout.YAxis(
            title="Objective value", visible=True, side="right", position=0.98
        )

        if self.title:
            title = f"Tree 2D: {self.probname} ({self.scipversion}, {self.status})"
        else:
            title = ""
        filename = "Tree_2D_" + self.probname + ".html"

        layout = go.Layout(
            title=title,
            font=dict(size=self.fontsize),
            font_family="Fira Sans",
            autosize=True,
            template="none",
            showlegend=self.showlegend,
            hovermode="closest",
            xaxis=xaxis,
            yaxis=yaxis,
        )

        if self.showbuttons:
            layout["updatemenus"] = self.updatemenus()
        layout["sliders"] = [sliders]

        self.fig2d = go.Figure(
            data=[nodes, primalbound, dualbound, optval, edges],
            layout=layout,
            frames=frames,
        )

        self.fig2d.write_html(file=filename, include_plotlyjs=self.include_plotlyjs)

        return self.fig2d

    def solve(self):
        """Solve the instance and collect and generate the tree data"""

        self.nodelist = []

        self.probname = os.path.splitext(os.path.basename(self.probpath))[0]

        model = Model("TreeD")

        if self.verbose:
            model.redirectOutput()
        else:
            model.hideOutput()

        eventhdlr = LPstatEventhdlr()
        eventhdlr.nodelist = self.nodelist
        model.includeEventhdlr(
            eventhdlr, "LPstat", "generate LP statistics after every LP event"
        )
        model.readProblem(self.probpath)
        if self.setfile:
            model.readParams(self.setfile)
        model.setIntParam("presolving/maxrestarts", 0)
        model.setParam("estimation/restarts/restartpolicy", "n")

        for setting in self.scip_settings:
            model.setParam(setting[0], setting[1])

        if self.verbose:
            print("optimizing problem", end="... ")
            start = time()
        try:
            model.optimize()
        except:
            print("optimization failed")

        if self.verbose:
            print(f"{model.getStatus()}, time: {time()-start:.2f} seconds")

        self.scipversion = "SCIP " + str(model.version())
        # self.scipversion = self.scipversion[:-1]+'.'+self.scipversion[-1]

        self.status = model.getStatus()
        if self.status == "optimal":
            self.optval = model.getObjVal()
        else:
            self.optval = None

        # print("performing Spatial Analysis on similarity of LP condition numbers")
        # self.performSpatialAnalysis()

        columns = self.nodelist[0].keys()
        self.df = pd.DataFrame(self.nodelist, columns=columns)

        # drop last data point of every node, since it's a duplicate
        # for n in self.df["number"]:
        #     seq = self.df[(self.df["first"] == False) & (self.df["number"] == n)]
        #     if len(seq) > 0:
        #         self.df.drop(
        #             index=seq.index[-1], inplace=True,
        #         )

        # merge solutions from cutting rounds into one node, preserving the latest
        # if self.showcuts:
        #     self.df = (
        #         self.df[self.df["first"] == False]
        #         .drop_duplicates(subset="age", keep="last")
        #         .reset_index()
        #     )
        if not self.showcuts:
            self.df = self.df[self.df["first"]].reset_index()

        self.df["id"] = self.df.index

    def hierarchy_pos(self, root=0, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """compute abstract node positions of the tree
        From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
        Licensed under Creative Commons Attribution-Share Alike 
        """
        G = self.nxgraph
        if not nx.is_tree(G):
            # raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")
            self.pos2d = nx.kamada_kawai_layout(G)
            return

        def _hierarchy_pos(
            G,
            root,
            width=1.0,
            vert_gap=0.2,
            vert_loc=0,
            xcenter=0.5,
            pos=None,
            parent=None,
        ):
            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children) != 0:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(
                        G,
                        child,
                        width=dx,
                        vert_gap=vert_gap,
                        vert_loc=vert_loc - vert_gap,
                        xcenter=nextx,
                        pos=pos,
                        parent=root,
                    )
            return pos

        self.pos2d = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    def compute_distances(self):
        """compute all pairwise distances between the original LP solutions and the transformed points"""
        if self.df is None:
            return

        origdist = []
        transdist = []
        for i in range(len(self.df)):
            for j in range(i + 1, len(self.df)):
                origdist.append(
                    self.distance(self.df["LPsol"].iloc[i], self.df["LPsol"].iloc[j])
                )
                transdist.append(
                    self.distance(
                        self.df[["x", "y"]].iloc[i], self.df[["x", "y"]].iloc[j]
                    )
                )
        self.distances = pd.DataFrame()
        self.distances["original"] = origdist
        self.distances["transformed"] = transdist

    @staticmethod
    def distance(p1, p2):
        """euclidean distance between two coordinates (dict-like storage)"""
        dist = 0
        for k in set([*p1.keys(), *p2.keys()]):
            dist += (p1.get(k, 0) - p2.get(k, 0)) ** 2
        return math.sqrt(dist)
