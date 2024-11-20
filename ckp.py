import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.utils import py_random_state
from random import choices
from scipy.stats import bernoulli
from ckp_experiments import *
from checking_mechanisms import *

def draw_graph(G, labels=True):
    '''
    Draw the multigraph G.
    '''
    # Source: https://stackoverflow.com/questions/15053686/networkx-overlapping-edges-when-visualizing-multigraph
    G = G.reverse()
    for layer, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            G.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer")

    ax = plt.gca()
    # Color CT nodes green, PF nodes red, CF nodes blue
    color_state_map = {"CT": 'green', "PF": 'red', "CF": 'blue'}

    node_colors = [color_state_map.get(
        G.nodes[node]['truth_value']) for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=100, alpha=1,  ax=ax)
    for e in G.edges:
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3*e[2])
                                                                           ),
                                    ),
                    )
    if labels:
        nx.draw_networkx_labels(G, pos=pos)
    plt.axis('off')
    plt.show()


def initialize_ckp(type, init=None):
    '''
    Initialize the CKP process, either with a given graph init, or with a single node.
    '''
    G = nx.MultiDiGraph()
    repeated_PT_nodes = []
    MF_nodes = []
    if init is not None:
        G = init
        for node in G.nodes():
            if G.nodes[node]['truth_value'] == "CT" or G.nodes[node]['truth_value'] == "CF":
                repeated_PT_nodes.extend([node] * (G.in_degree(node) + 1))
            if G.nodes[node]['truth_value'] == "CF":
                MF_nodes.append(node)
    else:
        if (type == "simple"):
            G.add_node(0, truth_value="CF")
            MF_nodes.append(0)
        elif (type == "general"):
            G.add_node(0, truth_value="CT")
        # repeated nodes has each PT node in the graph, appearing with frequency deg_PT(v) + 1.
        repeated_PT_nodes.append(0)
    return G, repeated_PT_nodes, MF_nodes


def update_ckp(G, iteration_number, repeated_PT_nodes,  MF_nodes, type, num_parents, p, k, epsilon=0, checking='all'):
    '''
    Run one iteration of the CKP process.
    - Add a new node with truth value depending on type (simple or general)
    - With probaility p, check ALL ancestors of the new node up to depth k. 
    - If any of these nodes are CF or PF, mark them as PF and remove them from repeated_PT_nodes.
    Parameters:
      iteration_number: label of the new node to be added
    Returns:
      G: the updated DAG
      repeated_PT_nodes: list of nodes that are PT nodes, with repetitions for preferential attachment weight
    '''
    if (type == "simple"):
        G.add_node(iteration_number, truth_value="CT")
    elif (type == "general"):
        if np.random.rand() < epsilon:
            G.add_node(iteration_number, truth_value="CF")
            MF_nodes.append(iteration_number)
        else:
            G.add_node(iteration_number, truth_value="CT")
    targets = choices(repeated_PT_nodes, k=num_parents)
    # Add edges FROM children to parents, so that we can use nx.descendants_at_distance to figure out
    # nodes to remove upon checking later. (This doesn't impact the preferential attachment since
    # the weights are stored separately in repeated_nodes.)
    edges = zip([iteration_number] * num_parents, targets)
    G.add_edges_from(edges)
    repeated_PT_nodes.extend(targets)
    repeated_PT_nodes.extend([iteration_number] * 1)

    # Handle exhaustive BFS first because it runs a separate check with probability p for each parent.
    if checking == 'exhaustive-BFS':
        G, repeated_PT_nodes, MF_nodes = check_exhaustive_BFS(G, iteration_number, p, k, repeated_PT_nodes, MF_nodes, targets)

    # Phase of removing nodes according to the checking model
    elif np.random.rand() < p:  # check with probability p
        # CHECKING ALL ANCESTORS
        if checking == 'all':
            G, repeated_PT_nodes = check_all(G, iteration_number, k, repeated_PT_nodes)

        # CHECK ONLY ONE PATH AT RANDOM
        elif checking == 'random-path':
            G, repeated_PT_nodes = check_random_path(G, iteration_number, k, repeated_PT_nodes)
        elif checking == 'BFS-wp-p':
            false_node_found_unused = False
            G, repeated_PT_nodes, MF_nodes, false_node_found_unused = check_BFS(G, iteration_number, k, repeated_PT_nodes, MF_nodes)

    return G, repeated_PT_nodes, MF_nodes


def check_parameters(num_parents, num_iterations, p, k, epsilon):
    '''
    Throw exceptions if any parameter (num_parents, num_iterations, p, k and epsilon) is not valid.
    '''
    # throw exception if num_parents is not an integer >= 1
    if not isinstance(num_parents, int) or num_parents < 1:
        raise Exception("num_parents must be an integer >= 1")
    # throw exception if num_iterations is not an integer >= 1
    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise Exception("num_iterations must be an integer >= 1")
    # throw exception if p is out of bounds
    if p < 0 or p > 1:
        raise Exception("p must be in [0,1]")
    # throw exception if k is not an integer >= 1
    if not isinstance(k, int) or k < 1:
        raise Exception("k must be an integer >= 1")
    # throw exception if epsilon is out of bounds
    if epsilon < 0 or epsilon > 1:
        raise Exception("epsilon must be in [0,1]")


async def ckp(type, num_parents, num_iterations, p, k, epsilon=0, init=None, draw_each_step=False, checking='all'):
    '''
    Parameters:
      type: "simple" (new node is CT) or "general" (new node is CF with probability epsilon)
      num_parents: number of edges to attach from a new node to existing nodes (note the same parent can appear multiple times)
      num_iterations: number of nodes to add to the DAG (note the DAG may be smaller if there are no more PT nodes)
      p: probability of checking each new node
      k: checking depth
      epsilon: probability of a new node being CF (only used if type is "general")
    Returns:
      G: the DAG after it either reaches size num_iterations or there are no more PT nodes
    '''
    check_parameters(num_parents, num_iterations, p, k, epsilon)
    G, repeated_PT_nodes, MF_nodes = initialize_ckp(type, init)
    if draw_each_step:
        draw_graph(G)
    iteration_number = len(G)-1
    while iteration_number < num_iterations and len(repeated_PT_nodes) > 0:
        iteration_number += 1
        G, repeated_PT_nodes, MF_nodes = update_ckp(
            G, iteration_number, repeated_PT_nodes, MF_nodes, type, num_parents, p, k, epsilon, checking)
        if draw_each_step: 
            draw_graph(G)
    return G, iteration_number


async def main():
    print("Creating graph")
    G_init = unit_test_instantiation2(3, 7)
    draw_graph(G_init)
    G, repeated_PT_nodes, MF_nodes = initialize_ckp("simple", init=G_init)
    G, repeated_PT_nodes, MF_nodes, false_node_found= check_BFS(G, 5, 2, repeated_PT_nodes, MF_nodes)
    draw_graph(G)

if __name__ == "__main__":
    asyncio.run(main())
