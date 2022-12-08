import numpy as np

def normalize_features(features, means=None, stds=None):
    if means is None:
        means = np.mean(features, axis=0)
    if stds is None:
        stds = np.std(features, axis=0)
        
    normalized_features = (features.copy()-means[np.newaxis, :])/stds[np.newaxis, :]
    
    return normalized_features, means, stds

def denormalize_features(features, means, stds):
    return features.copy()*stds[np.newaxis, :] + means[np.newaxis, :]
