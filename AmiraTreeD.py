import sys
from TreeD import TreeD

class AmiraTreeD:
    """
    Generate Amira data (.am file) to draw a visual representation of the
    branch-and-cut tree of SCIP for a particular instance using spatial
    dissimilarities of the node LP solutions.

    Dependencies:
     - TreeD.py to solve the instance and generate the necessary tree data
    """

    def __init__(self):
        self.am_data = ''

    def generateAmCode(self, dataFrame):
        """Generate Amira tree code"""

        # Number of items
        VERTEX = len(dataFrame['x'])
        EDGE = VERTEX - 1
        POINT = EDGE * 2

        # Header
        self.am_data += 'define VERTEX ' + str(VERTEX) + '\n'
        self.am_data += 'define EDGE ' + str(EDGE) + '\n'
        self.am_data += 'define POINT ' + str(POINT) + '\n'
        self.am_data += '\n'
        self.am_data += 'Parameters {\n'
        self.am_data += '\tContentType "HxSpatialGraph"\n'
        self.am_data += '}\n'
        self.am_data += '\n'
        self.am_data += 'VERTEX { float[3] VertexCoordinates } @1\n'
        self.am_data += 'EDGE { int[2] EdgeConnectivity } @2\n'
        self.am_data += 'EDGE { int NumEdgePoints } @3\n'
        self.am_data += 'POINT { float[3] EdgePointCoordinates } @4\n'
        self.am_data += '\n'

        # Nodes
        self.am_data += '@1\n'
        for x, y, z in zip(dataFrame['x'], dataFrame['y'], dataFrame['objval']):
            self.am_data += str(x) + ' ' + str(y) + ' ' + str(z) + '\n'
        self.am_data += '\n'

        # Edges

        ## IdVertex to IdVertex
        edges = [] # (node_id, node_parent, number, parent, first, x, y, z, x1, y1, z1)
        for node_id, number, parent, first, x, y, z in zip([i for i in range(VERTEX)], dataFrame['number'], dataFrame['parent'], dataFrame['first'], dataFrame['x'], dataFrame['y'], dataFrame['objval']):
            edges.append((node_id, None, number, parent, first, x, y, z, None, None, None))

        for i in range(len(edges) - 1, 0, -1): # First node (node_id == 0) is the root
            node_id, _, number, parent, first, x, y, z, _, _, _ = edges[i]

            parent_node = None
            if first is False:
                parent_node = (edges[i-1][0], edges[i-1][5], edges[i-1][6], edges[i-1][7])
            else:
                parent_node = [(node_id, x, y, z) for node_id, _, number, _, _, x, y, z, _, _, _ in edges[0:i] if number == parent][-1]

            edges[i] = (node_id, parent_node[0], number, parent, first, x, y, z, parent_node[1], parent_node[2], parent_node[3])

        self.am_data += '@2\n'
        for node_id, node_parent, _, _, _, _, _, _, _, _, _ in edges[1:]:
            self.am_data += str(node_parent) + ' ' + str(node_id) + '\n'        
        self.am_data += '\n'

        ## Points per Edge
        self.am_data += '@3\n'
        for i in range(EDGE):
            self.am_data += '2\n'
        self.am_data += '\n'

        ## Vertex to Vertex
        self.am_data += '@4\n'
        for _, _, _, _, _, x, y, z, x1, y1, z1 in edges[1:]:
            self.am_data += str(x1) + ' ' + str(y1) + ' ' + str(z1) + '\n'
            self.am_data += str(x) + ' ' + str(y) + ' ' + str(z) + '\n'
        self.am_data += '\n'

        # Node color (use labels256 on Amira)
        self.am_data += 'VERTEX { int VertexColor } @5\n'
        self.am_data += '\n'        
        self.am_data += '@5\n'
        for color_number in dataFrame['age']:
            self.am_data += str(color_number) + '\n'
        self.am_data += '\n'
        
    def saveAmFile(self, name_file):
        """Save Amira code to a .am file"""

        with open(name_file, 'w') as file:
            file.write(self.am_data)

if __name__ == "__main__":

    treed = TreeD()
    amiratreed = AmiraTreeD()

    if len(sys.argv) == 1:
        print(amiratreed.__doc__)
        print("usage: {} <MIP-instance> <output-am-file>".format(sys.argv[0]))
    elif len(sys.argv) == 3:
        treed.probpath = sys.argv[1]
        treed.main()
        amiratreed.generateAmCode(treed.df)
        amiratreed.saveAmFile(sys.argv[2])
    else:
        print("usage: {} <MIP-instance> <output-am-file>".format(sys.argv[0]))