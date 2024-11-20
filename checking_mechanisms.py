import networkx as nx
import numpy as np

# check all may have bugs
def check_all(G, iteration_number, k, repeated_PT_nodes):
    nodes_to_set_PF = []
    # print("Check at step ", iteration_number)
    # Check all ancestors of distance <= k away from the new node (can change later)
    descendents_at_dist_leq_k = []
    for dist in range(0, k+1):
        descendents_at_dist_leq_k += list(
            nx.descendants_at_distance(G, iteration_number, dist))

    for node in descendents_at_dist_leq_k:
        if G.nodes[node]['truth_value'] == "CF" or G.nodes[node]['truth_value'] == "PF":
            nodes_to_set_PF += [node]

            G_reverse = G.reverse()
            ancestors_of_node_at_dist_leq_k = []
            # find descendants of the CF/PF node found, at dist <= k - 1 away
            for dist in range(0, k):
                ancestors_of_node_at_dist_leq_k += list(
                    nx.descendants_at_distance(G_reverse, node, dist))

            # Update the truth values for all nodes with 'node' as an descendant (remember edges are reversed, so means ancestor)
            # and 'node_counter' (the new node) as an ancestor (remember this really means descendant). If we check not all paths
            # this will need to be changed.
            ancestor_and_descendant = list(
                set(ancestors_of_node_at_dist_leq_k).intersection(descendents_at_dist_leq_k))
            nodes_to_set_PF += ancestor_and_descendant
            nodes_to_set_PF += [iteration_number]

    #print("nodes_to_set_PF: ", nodes_to_set_PF)
    for node in descendents_at_dist_leq_k:
        if node in nodes_to_set_PF:
            G.nodes[node]['truth_value'] = "PF"
            if node in repeated_PT_nodes:
                # Remove all occurences of the node from repeated_PT_nodes (the PT nodes with repetitions for preferential attachment weight)
                repeated_PT_nodes = [
                    x for x in repeated_PT_nodes if x != node]
                
    return G, repeated_PT_nodes

def check_random_path(G, iteration_number, k, repeated_PT_nodes):
    path = [iteration_number]
    i = 0  # counter for number of nodes in path
    while i < k:
        # get the children of the last node in the path
        children = list(G.successors(path[-1]))
        if len(children) == 0:  # reached root (shouldn't ever hit this)
            break
        # choose one of the children at random and add it to the path
        child = np.random.choice(children)
        path.append(child)
        if G.nodes[child]['truth_value'] == "CF" or G.nodes[child]['truth_value'] == "PF":
            # set the whole path to PF and remove each node from repeated_PT_nodes
            for node in path:
                G.nodes[node]['truth_value'] = "PF"
                if node in repeated_PT_nodes:
                    repeated_PT_nodes = [
                        x for x in repeated_PT_nodes if x != node]
            break
        else:
            # remove a node if at least one of its parents is marked pf
            for parent in G.successors(child):
                if G.nodes[parent]['truth_value'] != "PF":
                    # set the whole path to PF and remove each node from repeated_PT_nodes
                    for node in path:
                        G.nodes[node]['truth_value'] = "PF"
                        if node in repeated_PT_nodes:
                            repeated_PT_nodes = [
                                x for x in repeated_PT_nodes if x != node]
                    break
        # if didn't break, the new added node is CT; increment the counter
        i += 1
    return G, repeated_PT_nodes

def check_BFS(G, startnode, k, repeated_PT_nodes, MF_nodes):
    false_node_found = False
    nodes_set_PF = []
    is_visited = [False] * len(G.nodes())
    BFS_queue = []
    #BFS_parent = [-1] * len(G.nodes())
    nodes_visited = []

    # Find nodes within k of start node
    BFS_possible_nodes = []
    for dist in range(0, k+1):
        BFS_possible_nodes += list(
            nx.descendants_at_distance(G, startnode, dist))
    #print("BFS_possible_nodes", BFS_possible_nodes)

    # Add new node to queue (but don't visit/check it until the checks from parents) or should it be visited?
    BFS_queue.append(startnode)
    is_visited[startnode] = True # visit before or after popping? (visit is more for BFS queue, not if it's been checked...)

    #check_depth = 0
    #while check_depth <= k and BFS_queue and not false_node_found:
    while BFS_queue and not false_node_found:
        # need to fix so that don't increase check depth each time pop a node!!
        
        # check order of checking parents and adding parents is fine...
        node = BFS_queue.pop()
        nodes_visited.append(node)
        #if G.nodes[node]['truth_value'] == "PF" or G.nodes[node]['truth_value'] == "CF":
        if node in MF_nodes:
            #print("found a False node")
            false_node_found = True
            # Trace back BFS parents to mark nodes as PF
            nodes_set_PF.append(node)
            '''temp_node = node
            if temp_node != startnode:
                while BFS_parent[temp_node] != startnode:
                    nodes_set_PF.append(BFS_parent[temp_node])
                    temp_node = BFS_parent[temp_node]
                nodes_set_PF.append(startnode)
            '''
            G_reverse = G.reverse()
            ancestors_of_node_at_dist_leq_k = []
            # find descendants of the False node found, at dist <= k - 1 away
            for dist in range(0, k+1):
                ancestors_of_node_at_dist_leq_k += list(
                    nx.descendants_at_distance(G_reverse, node, dist))
            nodes_visited = [i for i in range(len(G.nodes())) if is_visited[i]]
            ancestor_and_descendant = list(
            set(ancestors_of_node_at_dist_leq_k).intersection(nodes_visited))
            nodes_set_PF += ancestor_and_descendant
            break
        
        else:
            # Add all parent nodes to the queue (are child nodes in the Python implementation)
            for parent in G.successors(node):
                if is_visited[parent] == False and parent in BFS_possible_nodes:
                    BFS_queue.append(parent)
                    is_visited[parent] = True
                    #BFS_parent[parent] = node
        
        #check_depth += 1

    # remove duplicates for efficiency
    nodes_set_PF = list(set(nodes_set_PF))
    #print("nodes_set_PF: ", nodes_set_PF)
    #print("nodes visited/popped: ", nodes_visited)
    while len(nodes_set_PF) > 0:
        node = nodes_set_PF.pop()
        G.nodes[node]['truth_value'] = "PF"
        if node in repeated_PT_nodes:
            # Remove all occurences of the node from repeated_PT_nodes (the PT nodes with repetitions for preferential attachment weight)
            repeated_PT_nodes = [
                x for x in repeated_PT_nodes if x != node]
        # Make all children of the new PF nodes minimal-false nodes
        for child in G.predecessors(node):
            MF_nodes.append(child)
    
    return G, repeated_PT_nodes, MF_nodes, false_node_found


def check_exhaustive_BFS(G, iteration_number, p, k, repeated_PT_nodes, MF_nodes, parent_nodes):
    # checking parent nodes in the order they connected to them. Is this ok or sort parent nodes ordering?
    #print("New node ", iteration_number, " ------ parents ", parent_nodes[0], " and ", parent_nodes[1])
    false_node_found = False

    for parent in parent_nodes:
        if false_node_found:
            break
        if np.random.rand() < p:
            #print("Check parent: ", parent)
            # Check the new node. If is CF, then is in MF nodes (note that new node can't be a root when it's first created)
            if iteration_number in MF_nodes:
                false_node_found = True
                G.nodes[iteration_number]['truth_value'] = "PF"
                if iteration_number in repeated_PT_nodes:
                    # Remove all occurences of the node from repeated_PT_nodes (the PT nodes with repetitions for preferential attachment weight)
                    repeated_PT_nodes = [
                        x for x in repeated_PT_nodes if x != iter]
                # Make all children of the new PF nodes minimal-false nodes
                for child in G.predecessors(iteration_number):
                    MF_nodes.append(child)
            # BFS search depth k - 1 for parent node
            elif not false_node_found:
                G, repeated_PT_nodes, MF_nodes, false_node_found = check_BFS(G, parent, k-1, repeated_PT_nodes, MF_nodes)
                if false_node_found:
                    G.nodes[iteration_number]['truth_value'] = "PF"
                    if iteration_number in repeated_PT_nodes:
                        # Remove all occurences of the node from repeated_PT_nodes (the PT nodes with repetitions for preferential attachment weight)
                        repeated_PT_nodes = [
                            x for x in repeated_PT_nodes if x != iteration_number]
                        # Make all children of the new PF nodes minimal-false nodes
                        for child in G.predecessors(iteration_number):
                            MF_nodes.append(child)

    return G, repeated_PT_nodes, MF_nodes