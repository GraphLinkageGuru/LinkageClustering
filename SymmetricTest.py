from sklearn.cluster import AgglomerativeClustering
from graph_clustering_maker import GeneratePartitionedPointsList as generate_list
from graph_clustering_maker import MakeAdjacencyMatrix as make_matrix # hee hee ha
from graph_clustering_maker import AdjustedMutualInformation as adjust
import numpy as np

partition_uneven = [1, 3, 9, 7, 5, 11]

syntheticMatrix = make_matrix(partition_uneven, .9, .4) # exterior is between clusters, interior is in a cluster

exampleMatrix = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
              [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
              [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
              [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

agglom_cluster = AgglomerativeClustering(n_clusters=len(partition_uneven), metric='euclidean', linkage="ward")
labelsAsymmetric = agglom_cluster.fit_predict(syntheticMatrix)

def upper(matrix): 
    return  np.array([[(0 if (i > j) else matrix[i][j]) for j in range(0, len(matrix[0]))] for i in range(0, len(matrix))])   
print("before")
print(syntheticMatrix)

print("after")
print(upper(syntheticMatrix))


labelsSymmetric = agglom_cluster.fit_predict(upper(syntheticMatrix))

print("Before Label:")
print(labelsAsymmetric)
print("After Label:")
print(labelsSymmetric)

