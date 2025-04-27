from scipy.stats import kurtosis, entropy
import numpy as np

def kce_feature_selection(features):
    selected_indices = []
    for i in range(features.shape[1]):
        feat_column = features[:, i]
        k = kurtosis(feat_column)
        e = entropy(np.histogram(feat_column, bins=10, density=True)[0] + 1e-8)
        if k > 1 and e < 2:
            selected_indices.append(i)
    selected_features = features[:, selected_indices]
    return selected_features, selected_indices
