# This file contains experiments pertaining to the CKPs.
# For example, we may want to see how long it takes until all or most errors are removed (if ever) in the simple or general model
import numpy as np
import math
from ckp import *
import asyncio

# Returns True if all errors have been found: all nodes are either PF or are completely True,
# meaning the node is CT and all ancestors are CT.
def check_G_no_errors(G, repeated_PT_nodes):
    # If any CF nodes, not all errors have been found
    CFnodes = [n for n, y in G.nodes(data=True) if y['truth_value'] == "CF"]
    if (len(CFnodes) > 0):
        return False

    # See if the CT nodes are all completely True, meaning the node is CT and all ancestors are CT.
    CTnodes = [n for n, y in G.nodes(data=True) if y['truth_value'] == "CT"]
    for ctnode in CTnodes:
        ancestors = list(nx.descendants(G, ctnode))
        print("Ancestors: ", ancestors)

# Returns the first time in which all errors have been eliminated (after the first 10 iterations,
# to allow the graph to become more stable -- this decision can be changed.)

def check_error_elimination_time(type, num_parents, p, k, epsilon=0):
    G, repeated_PT_nodes = initialize_ckp(type)
    # Boolean of whether all nodes are True
    no_errors = False
    iteration_number = len(G)-1
    while (iteration_number < 10 or not no_errors):
        iteration_number += 1
        G, repeated_PT_nodes = update_ckp(
            G, iteration_number, repeated_PT_nodes, type, num_parents, p, k, epsilon)
        no_errors = check_G_no_errors(G, repeated_PT_nodes)
        CForPFnodes = [n for n, y in G.nodes(
            data=True) if y['truth_value'] == "CF" or y['truth_value'] == "PF"]
        print(len(CForPFnodes))
        # update truth value

    return "Number of iterations until all true: " + iteration_number


def low_prob_chain(num_parents, num_nodes=10):
    '''
    Initialize a simple DAG with num_nodes nodes forming a chain.
    '''
    G = nx.MultiDiGraph()
    G.add_node(0, truth_value="CF")
    for i in range(1, num_nodes):
        G.add_node(i, truth_value="CT")
    edges = [(i+1, i) for i in range(num_nodes-1)] * num_parents
    G.add_edges_from(edges)
    return G


def low_prob_chain_ball(num_parents, chain_len, ball_len):
    '''
    Initialize a simple DAG with num_nodes nodes forming a chain and a node with high degree.
    '''
    G = nx.MultiDiGraph()
    G.add_node(0, truth_value="CF")
    for i in range(1, chain_len):
        G.add_node(i, truth_value="CT")
    edges = [(i+1, i) for i in range(chain_len-1)] * num_parents
    G.add_edges_from(edges)
    for i in range(chain_len, chain_len+ball_len):
        G.add_node(i, truth_value="CT")
        edges = [(i, chain_len - 1)] * num_parents
        G.add_edges_from(edges)
    return G

# 3 initial nodes, only one False, followed by a chain
def unit_test_instantiation(num_parents, chain_len):
    G = nx.MultiDiGraph()
    G.add_node(0, truth_value="CF")
    G.add_node(1, truth_value="CT")
    G.add_node(2, truth_value="CT")
    G.add_node(3, truth_value="CT") 
    init_edges = [(3, j) for j in range(0, 3)]

    for i in range(4, chain_len+2):
        G.add_node(i, truth_value="CT")
    edges = [(i+1, i) for i in range(3, chain_len+1)] * num_parents
    G.add_edges_from(edges + init_edges)
    return G

# example to show that BFS search check can remove all nodes encountered in the BFS search that are descendants of the False node.
def unit_test_instantiation2(num_parents, chain_len):
    G = nx.MultiDiGraph()
    G.add_node(0, truth_value="CF")
    for i in range(1, 6):
        G.add_node(i, truth_value="CT")

    edges = [(1, 0), (1, 0), (2, 1), (2, 1), (3, 2), (5, 3), (5, 4), (4, 0)]
    G.add_edges_from(edges)
    return G

async def survival_experiment(N, cutoff_num, num_pars=2, p=1.0, k=2, checking='all', draw_graph=True, type="simple"):
    '''
    Run the survival experiment N times and compute how many survive.
    '''
    stopped_iters = []  # DAG size (-1) at cutoff for each experiment
    for _ in range(N):
        G_init = low_prob_chain(num_pars, 25)
        G1, stopped_iter = await ckp(type, num_pars, cutoff_num, p, k, epsilon=0.25,
                               init=G_init, draw_each_step=False, checking=checking)
        stopped_iters.append(stopped_iter)
        # survived if stopped_iter == cutoff_num
        if stopped_iter >= cutoff_num and draw_graph:
            draw_graph(G1, labels=False)
    num_survivals = sum(np.array(stopped_iters) >= cutoff_num)
    '''print('Setup: num parents =', num_pars, ', p = ', p,  ', k =', k,
          ', checking =', checking, 'path \n***********')
    print('Experiments stopped at iterations: ', stopped_iters)
    print('Out of', N, 'experiments', num_survivals,
          'survived until iteration', cutoff_num)
    print('Survival rate:', num_survivals / N)'''
    return num_survivals

def heights(H, node):

    # can define height as length of path to pf, or to pf or cf.
    if node == 0 or H.nodes[node]['truth_value'] == "PF" or H.nodes[node]['truth_value'] == "CF":
        return [0]

    node_heights = []
    
    # Remember that the parents of the node vertex in the CKP are the children in the p
    for parent in H.successors(node):
        # multiply by number of edges to parent. Add 1 to each parent height
        node_heights += [height+1 for height in heights(H, parent)] * H.number_of_edges(node, parent)

    return node_heights

def compute_heights(H):
    path_heights = {}
    for node in H.nodes():
        path_heights.update({node: heights(H, node)})

    return path_heights
    
def exponential_potential(G, c=0.5):
    exponential_pot = 0
    path_lengths = compute_heights(G)
    for node in G.nodes():
        if G.nodes[node]['truth_value'] != "PF":
            pt_outdeg = 0
            # compute pt outdegree
            for child in G.predecessors(node):
                if G.nodes[child]['truth_value'] != "PF":
                    pt_outdeg += G.number_of_edges(child, node)
            dv = 1 + pt_outdeg
            sum_of_c_powers = 0
            for height in path_lengths[node]:
                sum_of_c_powers += pow(c, height)
            exponential_pot += dv * sum_of_c_powers

    print(exponential_pot)
    return exponential_pot

def small_outdeg_potential(G, k, num_parents):
    leaves_pot = 0
    for node in G.nodes():
        if G.nodes[node]['truth_value'] != "PF":
            pt_outdeg = 0
            # compute pt outdegree
            for child in G.predecessors(node):
                if G.nodes[child]['truth_value'] != "PF":
                    pt_outdeg += G.number_of_edges(child, node)
            const = num_parents
            if pt_outdeg <= const:
               leaves_pot += 1 + const - pt_outdeg

    print(leaves_pot)
    return leaves_pot

def leaves_potential(G, num_parents):
    pot = 0
    for node in G.nodes():
        if G.nodes[node]['truth_value'] != "PF":
            pt_outdeg = 0
            for child in G.predecessors(node):
                if G.nodes[child]['truth_value'] != "PF":
                    pt_outdeg += G.number_of_edges(child, node)
            if pt_outdeg == 0:
                pot += 1
    print(pot)
    return pot

def leaves_and_parents_potential(G, num_parents):
    pot = 0
    for node in G.nodes():
        if G.nodes[node]['truth_value'] != "PF":
            pt_indeg = 0
            pt_outdeg = 0
            const = num_parents
            
            for child in G.predecessors(node):
                if G.nodes[child]['truth_value'] != "PF":
                    pt_outdeg += G.number_of_edges(child, node)
            if pt_outdeg == 0:
                for parent in G.successors(node):
                    if G.nodes[parent]['truth_value'] != "PF":
                        pt_indeg += G.number_of_edges(parent, node)
                
                pot += 1 + const - pt_indeg

    print(pot)
    return pot

def root_to_leaf_paths_potential(G):
    pot = 0
    for node in G.nodes():
        if G.nodes[node]['truth_value'] != "PF":
            pt_outdeg = 0
            for child in G.predecessors(node):
                if G.nodes[child]['truth_value'] != "PF":
                    pt_outdeg += G.number_of_edges(child, node)
            if pt_outdeg == 0:
                # the length of heights is the number of root to node (= leaf) paths
                path_heights = heights(G, node)
                pot += len(path_heights)
    print(pot)
    return pot

# compute average height weighted by the preferential attachment weights
def average_height(H):

    pot = 0
    sum_of_weights = 0
    path_lengths = compute_heights(H)
    for node in H.nodes():
        if H.nodes[node]['truth_value'] != "PF":
            minheight = min(path_lengths[node])
            pt_outdeg = 0
            for child in H.predecessors(node):
                if H.nodes[child]['truth_value'] != "PF":
                    pt_outdeg += H.number_of_edges(child, node)
            pot += minheight * (pt_outdeg + 1)
            sum_of_weights += pt_outdeg + 1

    if sum_of_weights == 0:
        return 0
    # Divide by number of alive nodes
    pot = pot / sum_of_weights
    return pot

def average_num_PT_children(H):

    outdegree_sum = 0
    outdegree_list = []
    num_alive_nodes = 0
    for node in H.nodes():
        if H.nodes[node]['truth_value'] != "PF":
            pt_outdeg = 0
            for child in H.predecessors(node):
                if H.nodes[child]['truth_value'] != "PF":
                    pt_outdeg += H.number_of_edges(child, node)
            outdegree_sum += pt_outdeg
            outdegree_list += [pt_outdeg]
            num_alive_nodes += 1
    
    if num_alive_nodes == 0:
        return 0, []
    
    pot= outdegree_sum / num_alive_nodes
    return pot, outdegree_list

def ckp_with_potential(type, num_parents, num_iterations, p, k, epsilon=0, init=None, draw_each_step=False, checking='all', potential='exponential'):
    print('Potential: ', potential)
    check_parameters(num_parents, num_iterations, p, k, epsilon)
    graphG = nx.MultiDiGraph()
    graphG.clear()
    potential_values = []
    graphG, repeated_PT_nodes = initialize_ckp(type, init)
    if potential == 'exponential':
        potential_values += [exponential_potential(graphG)]
    elif potential == 'small-outdeg':
        potential_values += [small_outdeg_potential(graphG, k, num_parents)]
    elif potential == 'leaves-and-indeg':
        potential_values += [leaves_and_parents_potential(graphG, num_parents)]
    elif potential == 'leaves':
        potential_values += [leaves_potential(graphG, num_parents)]
    elif potential == 'root-to-leaf-paths':
        potential_values += [root_to_leaf_paths_potential(graphG)]
    elif potential == 'avg-height':
        potential_values += [average_height(graphG)]
    elif potential == 'avg-num-children':
            potential_values += [average_num_PT_children(graphG)[0]]
    if draw_each_step:
        draw_graph(graphG)
    iteration_number = len(graphG)-1

    while iteration_number < num_iterations and len(repeated_PT_nodes) > 0:
        iteration_number += 1
        graphG, repeated_PT_nodes = update_ckp(
            graphG, iteration_number, repeated_PT_nodes, type, num_parents, p, k, epsilon, checking)
        if potential == 'exponential':
            potential_values += [exponential_potential(graphG)]
        elif potential == 'small-outdeg':
            potential_values += [small_outdeg_potential(graphG, k, num_parents)]
        elif potential == 'leaves-and-indeg':
            potential_values += [leaves_and_parents_potential(graphG, num_parents)]
        elif potential == 'leaves':
            potential_values += [leaves_potential(graphG, num_parents)]
        elif potential == 'root-to-leaf-paths':
            potential_values += [root_to_leaf_paths_potential(graphG)]
        elif potential == 'avg-height':
            potential_values += [average_height(graphG)]
        elif potential == 'avg-num-children':
            potential_values += [average_num_PT_children(graphG)[0]]
    
    if len(repeated_PT_nodes) > 0:
        print('Graph survived')
    return graphG, iteration_number, potential_values

def main():
    '''survival_experiment(N=25, cutoff_num=2000, num_pars=2, p=0.2, k=10, checking='all', draw_graph=False)
    survival_experiment(N=25, cutoff_num=2000, num_pars=2, p=0.3, k=10, checking='all', draw_graph=False)'''

    p = 0.2
    k = 2
    m = 2
    G_init = low_prob_chain(m, 4)
    G, iternum = ckp("general", m, 200, p, k, draw_each_step=True, init=G_init, checking='random-path')

    return

    G_init = low_prob_chain(m, 1)
    last_height_val = []
    p_vals = [0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    k_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for pval in p_vals:
        print(pval)
        G1, iternum, pot_values = ckp_with_potential("general", m, 1000, pval, k, epsilon=0.2, draw_each_step=False, init=G_init, checking='random-path', potential='avg-height')
        
        plt.plot(pot_values)
        plt.title("Potential values")
        plt.show()

        print(pot_values[len(pot_values)-1])
        last_height_val += [pot_values[len(pot_values)-1]]
        G_init.clear()
        G_init = low_prob_chain(m, 1)

    plt.plot(last_height_val)
    plt.show()

    #G_init = low_prob_chain(2, 10)
    #G1, stopped_iter = ckp("simple", 2, 250, 1, 2, init = G_init, draw_each_step=True, checking='all')
    #draw_graph(G1)
    '''G1, stopped_iter = ckp("simple", 2, 250, 0.5, 2, init = G_init, draw_each_step=False, checking='all')
    draw_graph(G1)
    G1, stopped_iter = ckp("simple", 2, 250, 0.5, 2, init = G_init, draw_each_step=False, checking='all')
    draw_graph(G1)
    G1, stopped_iter = ckp("simple", 2, 250, 0.5, 2, init = G_init, draw_each_step=False, checking='all')
    draw_graph(G1)'''


if __name__ == "__main__":
    main()
