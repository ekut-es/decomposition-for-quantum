import numpy as np
from torch import load, Tensor
from torch.nn import ReLU

weights1 = load("weights1.dat")
weights2 = load("weights2.dat")
relu = ReLU()

means = np.load("means.npy")
stds = np.load("stds.npy")

def compute_tsp_features_supported(positions, adj_matrix, points_support, first_node=0):
    
    def get_area(pos):
        min_x, min_y = np.min(pos, axis=0)
        max_x, max_y = np.max(pos, axis=0)
        l_x = abs(max_x-min_x)
        l_y = abs(max_y-min_y)
        
        if l_x == 0 or l_y == 0:
            ratio = 0
        else:
            ratio = l_y/l_x
            if ratio < 1:
                ratio = l_x/l_y
        
        return l_x*l_y, ratio
    
    n_support = len(points_support)
    n_work = len(positions) - n_support
    
    positions_support = positions[points_support,:]
    
    points_work = tuple(frozenset(range(len(positions))).difference(points_support))
    positions_work = positions[points_work,:]

    A_support, ratio_support = get_area(positions_support)
    A_work, ratio_work = get_area(positions_work)
    
    c_supp = np.mean(adj_matrix[points_support,:][:,points_support])    
    c_source = np.mean(adj_matrix[first_node,:])
    c_drain = np.mean(adj_matrix[:,0])
        
    return (np.sqrt(n_support),
            np.sqrt(n_work),
            np.sqrt(A_support),
            np.sqrt(A_work),
            np.sqrt(ratio_support),
            np.sqrt(ratio_work),
            c_supp,
            c_source,
            c_drain)

def predict_tsp_supported(positions, adj_matrix, support_points, first_node=0):
    features = np.array(compute_tsp_features_supported(positions, adj_matrix, support_points, first_node=first_node))
    features = (features-means[:-1])/stds[:-1]
    prediction = float(weights2(relu(weights1(Tensor(features)))))
    return (prediction*stds[-1]+means[-1])
