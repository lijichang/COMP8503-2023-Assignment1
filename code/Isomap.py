
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import pairwise_distances


def isomap(X, n_neighbors, n_components=2):
    ## Step 1. Build adjacency graph 
    # compute pairwise distances between data points using Euclidean distances 
    graph = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            distance = np.linalg.norm(X[i] - X[j])
            graph[i, j] = distance
            graph[j, i] = distance

    # compute top k nearest neighbors
    neighbors = np.argsort(graph)[:, 1:n_neighbors+1]

    # construct adjacency matrix
    adjacency_matrix = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        adjacency_matrix[i, neighbors[i]] = graph[i, neighbors[i]]
        adjacency_matrix[neighbors[i], i] = graph[neighbors[i], i]
    
    ## Step 2. Estimate geodesics 
    # compute shortest path distances
    graph = csr_matrix(adjacency_matrix)
    graph, _ = shortest_path(csgraph=graph, directed=False, return_predecessors=True)

    # compute H = I - (1/n) * e * e^T, n is the sample size, e is the vector of ones
    H = np.eye(X.shape[0]) - np.ones((X.shape[0], X.shape[0])) / X.shape[0]

    # compute Gram matrix K = -(1/2) * H*(D)^2*H, D is the distance metrix graph
    K = -1/2 * np.dot(np.dot(H, graph ** 2), H)

    ## Step 3. Metric MDS
    # compute eigenvectors and eigenvalues of Gram matrix K using Singular Value Decomposistion (SVD)
    eigen_value, eigen_vector = np.linalg.eigh(K)

    # sort the eigenvectors ordered by the corresponding eigenvalues
    indices = np.argsort(eigen_value)[::-1]

    # obtain the top d (i.e., n_components) eigenvalues 
    eigen_vector = eigen_vector[:, indices]

    # project the given data onto top d eigenvectors
    embedding = eigen_vector[:, :n_components]

    return embedding