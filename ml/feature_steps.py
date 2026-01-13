"""
Feature engineering pipeline steps: PCA, KMeansFeatures.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union


def create_pca_step(
    enabled: bool,
    n_components: Optional[Union[int, float]] = None,
    whiten: bool = False,
    random_state: int = 42
):
    """
    Create a PCA transformer step.
    
    Args:
        enabled: Whether PCA is enabled
        n_components: Number of components (int) or variance threshold (float 0-1)
        whiten: Whether to whiten components
        random_state: Random seed
        
    Returns:
        PCA transformer or None if disabled
    """
    if not enabled:
        return None
    
    return PCA(
        n_components=n_components,
        whiten=whiten,
        random_state=random_state
    )


class KMeansFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer that adds KMeans cluster-based features.
    
    Adds:
    - Distances to cluster centroids (always)
    - Optional one-hot cluster label
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        add_distances: bool = True,
        add_onehot_label: bool = False,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.add_distances = add_distances
        self.add_onehot_label = add_onehot_label
        self.random_state = random_state
        self.kmeans_ = None
        self.cluster_centers_ = None
    
    def fit(self, X, y=None):
        """Fit KMeans on X."""
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans_.fit(X)
        self.cluster_centers_ = self.kmeans_.cluster_centers_
        return self
    
    def transform(self, X):
        """Transform X by adding cluster features."""
        if self.kmeans_ is None:
            raise ValueError("KMeansFeatures must be fitted before transform")
        
        features = []
        
        # Distances to centroids
        if self.add_distances:
            distances = self.kmeans_.transform(X)  # Shape: (n_samples, n_clusters)
            features.append(distances)
        
        # One-hot cluster labels
        if self.add_onehot_label:
            labels = self.kmeans_.predict(X)  # Shape: (n_samples,)
            onehot = np.zeros((len(labels), self.n_clusters))
            onehot[np.arange(len(labels)), labels] = 1
            features.append(onehot)
        
        if not features:
            # If nothing enabled, return original X
            return X
        
        # Concatenate all features
        result = np.hstack(features)
        
        # If only distances, return distances
        # If only onehot, return onehot
        # If both, concatenate: [X, distances, onehot]
        # For now, we're replacing X with cluster features (not appending)
        # To append, would need: np.hstack([X, result])
        return result
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        names = []
        
        if self.add_distances:
            for i in range(self.n_clusters):
                names.append(f'kmeans_dist_cluster_{i}')
        
        if self.add_onehot_label:
            for i in range(self.n_clusters):
                names.append(f'kmeans_cluster_{i}')
        
        return np.array(names) if names else input_features
