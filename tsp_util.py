import numpy as np
from itertools import combinations, product
from scipy.spatial.distance import cdist

def generate_tsp_instance(instance_size, area_ratio=1, scale=10):
    """
    Generate uniformly distributed random TSP instance

    Parameters:

    instance_size : Number of nodes in the TSP instance
    area_ratio : Ratio of width/length of the smalles rectangle
        that covers all nodes
    scale : Scale of the positions of all nodes
    """
    ratio = area_ratio
    if np.random.randint(2) == 0:
        size_x = scale*ratio
        size_y = scale/ratio
    else:
        size_x = scale/ratio
        size_y = scale*ratio
 
    positions = tuple((x,y) for x,y in zip(np.random.rand(instance_size)*2*size_x-size_x, np.random.rand(instance_size)*2*size_y-size_y))
    positions = np.array(positions)

    adj_matrix = cdist(positions, positions)
    
    return positions, adj_matrix

def tsp_dynamic(adj_matrix, first_node=0, last_node=0):
    """
    Solve a TSP instance using dynamic programming

    Parameters:

    adj_matrix : Adjacency matrix for the TSP, must be quadratic
    first_node : The node the tour should start on
    last_node : The node the tour should end on. If last_node is not
        equal to first_node, the function returns a minimum hamiltonian path
    """
    n = adj_matrix.shape[0]
    init_minimum = np.sum(adj_matrix)
    table = {}
    nodes = frozenset(range(n)).difference((first_node, last_node))
    for i in nodes:
        table[(frozenset((i,)),i)] = ((i,), adj_matrix[first_node, i])
    for s in range(2, len(nodes)+1):
        for S in combinations(nodes, s):
            set_S = frozenset(S)
            for k in S:
                minimum_distance = init_minimum
                minimum_node = None
                minimum_path = None
                set_Smk = set_S.difference((k,))
                for m in set_Smk:
                    table_entry = table[(set_Smk, m)]
                    distance = table_entry[1] + adj_matrix[m,k]
                    if distance < minimum_distance:
                        minimum_distance = distance
                        minimum_node = m
                        minimum_path = table_entry[0]
                table[(set_S, k)] = (minimum_path + (k,), minimum_distance)
    
    minimum_distance = init_minimum
    minimum_path = None
    # Find Minimum
    for i in nodes:
        table_entry = table[(nodes, i)]
        distance = table_entry[1] + adj_matrix[i,last_node]
        if distance < minimum_distance:
            minimum_distance = distance
            minimum_path = table_entry[0] + (last_node,)
    return (minimum_path, minimum_distance)

def tsp_dynamic_support(adj_matrix, fixpoints, first_node=0, last_node=0):
    """
    Solve a TSP instance where equidistant fixpoints along the sought path
    are given. The number of nodes in the TSP instance should be divisible
    by the number of fixpoints.
    """
    n = adj_matrix.shape[0]
    init_minimum = np.sum(adj_matrix)
    table = {}
    for i in range(1, n):
        table[(frozenset((i,)),i)] = ((i,), adj_matrix[first_node, i])
    nodes = frozenset(range(n)).difference((first_node, last_node))
    steps = (n-1)//len(fixpoints)
    
    for s in range(2, len(nodes)+1):
        for S in combinations(nodes, s):
            set_S = frozenset(S)
            included_fixpoints = set_S.intersection(fixpoints)
            if len(included_fixpoints) == s//steps:
                next_set = included_fixpoints if s % steps == 0 else set_S.difference(fixpoints)
                for k in next_set:
                    minimum_distance = init_minimum
                    minimum_node = None
                    minimum_path = None
                    set_Smk = set_S.difference((k,))
                    set_Smk_fixpoints = set_Smk.intersection(fixpoints)
                    prev_set = set_Smk_fixpoints if len(set_Smk) % steps == 0 else set_Smk.difference(fixpoints)
                    for m in prev_set:
                        table_entry = table[(set_Smk, m)]
                        distance = table_entry[1] + adj_matrix[m,k]
                        if distance < minimum_distance:
                            minimum_distance = distance
                            minimum_node = m
                            minimum_path = table_entry[0]

                    table[(set_S, k)] = (minimum_path + (k,), minimum_distance)
    
    minimum_distance = init_minimum
    minimum_path = None
    # Find Minimum
    # I assume here that the last point is a fixpoint
    # This is obviously only true if a correct number of fixpoints has been given (i.e. len(nodes) is divisible by len(fixpoints))
    for i in fixpoints:
        table_entry = table[(nodes, i)]
        distance = table_entry[1] + adj_matrix[i,last_node]
        if distance < minimum_distance:
            minimum_distance = distance
            minimum_path = table_entry[0] + (last_node,)
    return (minimum_path, minimum_distance)

def compute_path_length(path, adj_matrix):
    length = 0
    for i,j in zip(path[:-1], path[1:]):
        length += adj_matrix[i,j]
    return length
