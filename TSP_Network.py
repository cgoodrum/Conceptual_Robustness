import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#np.random.seed(1)

# creates a network using the information in the yaml template file.
def create_network(node_spec, edge_spec):

    G = nx.DiGraph()
    n = node_spec['num_nodes']
    m = edge_spec['num_edges']
    length_info = edge_spec['length']

    if length_info['type'] == 'default':
        wtMatrix = [[1 for _ in range(n)] for _ in range(n) ]
        for i in range(n):
            for j in range(n):
                if i == j:
                    wtMatrix[i][j] = 0

        #Adds egdes along with their weights to the graph
        for i in range(n):
            for j in range(n)[i:]:
                if wtMatrix[i][j] > 0:
                    G.add_edge(i, j, length = wtMatrix[i][j])

    elif length_info['type'] == 'uniform':
        lb = length_info['parameters']['lower_bound']
        ub = length_info['parameters']['upper_bound']

        wtMatrix = np.random.random_integers(lb,ub,size=(n,n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    wtMatrix[i][j] = 0

        ut_wtMatrix = np.triu(wtMatrix, k=0)
        for i in range(n):
            for j in range(n):
                ut_wtMatrix[j][i]= ut_wtMatrix[i][j]
        # Adds egdes along with their weights to the graph
        for i in range(n):
            for j in range(n)[i:]:
                if wtMatrix[i][j] > 0:
                    G.add_edge(i, j, length=wtMatrix[i][j])
                    G.add_edge(j, i, length=wtMatrix[j][i])

    elif length_info['type'] == 'normal':
        pass
    else:
        print('Error in edge weights entry')

    return G

def draw_network(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels = True)  #with_labels=true is to show the node number in the output graph
    edge_labels = nx.get_edge_attributes(G,'length')
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels,  font_size = 11) #prints weight on all the edges
    plt.show()
    return pos

