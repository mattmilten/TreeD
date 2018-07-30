import sys
from TreeD import TreeD

def normalise_list(values):
    min_value = min(values)
    max_value = max(values)
    return [(value - min_value) / (max_value - min_value) for value in values]

def normalise_value(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

class AmiraTreeD:
    """
    Generate Amira data (.am file) to draw a visual representation of the
    branch-and-cut tree of SCIP for a particular instance using spatial
    dissimilarities of the node LP solutions.

    Dependencies:
     - TreeD.py to solve the instance and generate the necessary tree data
    """

    def __init__(self):
        self.folderPath = ''

    def treeAmCode(self, dataFrame):
        """Generate Amira tree code"""

        tree_am_data = ''

        # Number of items
        VERTEX = len(dataFrame['x'])
        EDGE = VERTEX - 1
        POINT = EDGE * 2

        # Header
        tree_am_data += 'define VERTEX ' + str(VERTEX) + '\n'
        tree_am_data += 'define EDGE ' + str(EDGE) + '\n'
        tree_am_data += 'define POINT ' + str(POINT) + '\n'
        tree_am_data += '\n'
        tree_am_data += 'Parameters {\n'
        tree_am_data += '\tContentType "HxSpatialGraph"\n'
        tree_am_data += '}\n'
        tree_am_data += '\n'
        tree_am_data += 'VERTEX { float[3] VertexCoordinates } @1\n'
        tree_am_data += 'EDGE { int[2] EdgeConnectivity } @2\n'
        tree_am_data += 'EDGE { int NumEdgePoints } @3\n'
        tree_am_data += 'POINT { float[3] EdgePointCoordinates } @4\n'
        tree_am_data += '\n'

        x_box = normalise_list(dataFrame['x'])
        y_box = normalise_list(dataFrame['y'])
        z_box = normalise_list(dataFrame['objval'])

        ## MATPLOTLIB STUFF
        '''
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_box, y_box, z_box, color='r', marker='o')
        plt.show()
        '''
        ##

        # Nodes
        tree_am_data += '@1\n'
        for x, y, z in zip(x_box, y_box, z_box):
            tree_am_data += str(x) + ' ' + str(y) + ' ' + str(z) + '\n'
        tree_am_data += '\n'

        # Edges

        ## IdVertex to IdVertex
        edges = [] # (node_id, node_parent, number, parent, first, x, y, z, x1, y1, z1)
        for node_id, number, parent, first, x, y, z in zip([i for i in range(VERTEX)], dataFrame['number'], dataFrame['parent'], dataFrame['first'], x_box, y_box, z_box):
            edges.append((node_id, None, number, parent, first, x, y, z, None, None, None))

        for i in range(len(edges) - 1, 0, -1): # First node (node_id == 0) is the root
            node_id, _, number, parent, first, x, y, z, _, _, _ = edges[i]

            parent_node = None
            if first is False:
                parent_node = (edges[i-1][0], edges[i-1][5], edges[i-1][6], edges[i-1][7])
            else:
                parent_node = [(node_id, x, y, z) for node_id, _, number, _, _, x, y, z, _, _, _ in edges[0:i] if number == parent][-1]

            edges[i] = (node_id, parent_node[0], number, parent, first, x, y, z, parent_node[1], parent_node[2], parent_node[3])

        tree_am_data += '@2\n'
        for node_id, node_parent, _, _, _, _, _, _, _, _, _ in edges[1:]:
            tree_am_data += str(node_parent) + ' ' + str(node_id) + '\n'        
        tree_am_data += '\n'

        ## Points per Edge
        tree_am_data += '@3\n'
        for i in range(EDGE):
            tree_am_data += '2\n'
        tree_am_data += '\n'

        ## Vertex to Vertex
        tree_am_data += '@4\n'
        for _, _, _, _, _, x, y, z, x1, y1, z1 in edges[1:]:
            tree_am_data += str(x1) + ' ' + str(y1) + ' ' + str(z1) + '\n'
            tree_am_data += str(x) + ' ' + str(y) + ' ' + str(z) + '\n'
        tree_am_data += '\n'

        # Node color (use labels256 on Amira)
        tree_am_data += 'VERTEX { int Age } @5\n'
        tree_am_data += '\n'        
        tree_am_data += '@5\n'
        for color_number in dataFrame['age']:
            tree_am_data += str(color_number) + '\n'
        tree_am_data += '\n'

        print('Storing DataTree.am ...')
        with open(self.folderPath + '/DataTree.am', 'w') as file:
            file.write(tree_am_data)

    def optAmCode(self, dataFrame, optval):
        """Generate Amira tree code"""

        opt_am_data = ''

        # Number of items
        VERTEX = 4
        EDGE = 4
        POINT = EDGE * 2

        # Header
        opt_am_data += 'define VERTEX ' + str(VERTEX) + '\n'
        opt_am_data += 'define EDGE ' + str(EDGE) + '\n'
        opt_am_data += 'define POINT ' + str(POINT) + '\n'
        opt_am_data += '\n'
        opt_am_data += 'Parameters {\n'
        opt_am_data += '\tContentType "HxSpatialGraph"\n'
        opt_am_data += '}\n'
        opt_am_data += '\n'
        opt_am_data += 'VERTEX { float[3] VertexCoordinates } @1\n'
        opt_am_data += 'EDGE { int[2] EdgeConnectivity } @2\n'
        opt_am_data += 'EDGE { int NumEdgePoints } @3\n'
        opt_am_data += 'POINT { float[3] EdgePointCoordinates } @4\n'
        opt_am_data += '\n'

        min_x = min(dataFrame['x'])
        max_x = max(dataFrame['x'])
        min_y = min(dataFrame['y'])
        max_y = max(dataFrame['y'])

        min_x_box = normalise_value(min_x, min_x, max_x)
        max_x_box = normalise_value(max_x, min_x, max_x)
        min_y_box = normalise_value(min_y, min_y, max_y)
        max_y_box = normalise_value(max_y, min_y, max_y)

        optval_box = normalise_value(optval, min(dataFrame['objval']), max(dataFrame['objval']))

        points_x = [min_x_box, min_x_box, max_x_box, max_x_box]
        points_y = [min_y_box, max_y_box, max_y_box, min_y_box]
        points_z = [optval_box] * 4

        ## MATPLOTLIB STUFF
        '''
        x_box = normalise_list(dataFrame['x'])
        y_box = normalise_list(dataFrame['y'])
        z_box = normalise_list(dataFrame['objval'])

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.collections import PolyCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_box, y_box, z_box, c='r', marker='o')

        xs = points_x
        ys = points_y
        zs = points_z
        verts = []
        verts.append(list(zip(xs, ys)))
        poly = PolyCollection(verts, facecolors='g', edgecolors='g')
        poly.set_alpha(0.7)
        ax.add_collection3d(poly, zs=zs, zdir='z')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
        '''
        ##

        # Nodes
        opt_am_data += '@1\n'
        for x, y, z in zip(points_x, points_y, points_z):
            opt_am_data += str(x) + ' ' + str(y) + ' ' + str(z) + '\n'
        opt_am_data += '\n'

        # Edges

        ## IdVertex to IdVertex
        opt_am_data += '@2\n'
        for i in range(0, 3):
            opt_am_data += str(i) + ' ' + str(i+1) + '\n'        
        opt_am_data += str(3) + ' ' + str(0) + '\n'        
        opt_am_data += '\n'

        ## Points per Edge
        opt_am_data += '@3\n'
        for i in range(EDGE):
            opt_am_data += '2\n'
        opt_am_data += '\n'

        ## Vertex to Vertex
        opt_am_data += '@4\n'
        for i in range(0, 3):
            opt_am_data += str(points_x[i]) + ' ' + str(points_y[i]) + ' ' + str(points_z[i]) + '\n'
            opt_am_data += str(points_x[i+1]) + ' ' + str(points_y[i+1]) + ' ' + str(points_z[i+1]) + '\n'
        opt_am_data += str(points_x[3]) + ' ' + str(points_y[3]) + ' ' + str(points_z[3]) + '\n'
        opt_am_data += str(points_x[0]) + ' ' + str(points_y[0]) + ' ' + str(points_z[0]) + '\n'
        opt_am_data += '\n'

        print('Storing Dataopt.am ...')
        with open(self.folderPath + '/DataOpt.am', 'w') as file:
            file.write(opt_am_data)

if __name__ == "__main__":

    treed = TreeD()
    amiratreed = AmiraTreeD()

    if len(sys.argv) == 1:
        print(amiratreed.__doc__)
        print("usage: {} <MIP-instance> <output-am-files-folder>".format(sys.argv[0]))
    elif len(sys.argv) == 3:
        treed.probpath = sys.argv[1]
        treed.main()
        amiratreed.folderPath = sys.argv[2]
        amiratreed.treeAmCode(treed.df)
        amiratreed.optAmCode(treed.df, treed.optval)
        print('Color map from %s to %s' % (min(treed.df['age']), max(treed.df['age'])))
    else:
        print("usage: {} <MIP-instance> <output-am-files-folder>".format(sys.argv[0]))