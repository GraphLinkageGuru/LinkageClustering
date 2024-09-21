from Tools.GraphCluster import *

PartitionList = [3,4,3] # repreasenting a 10-node graph with three clusters, cluster 0 has 3 nodes, cluster 1 has 4 nodes, and cluster 2 has 3 nodes
PointsList = generate_list(PartitionList) # this will "decompress" the partition list into a list with one element per node, where the number in the entry is the partition it belongs to 
# PointsList = [0,0,0,1,1,1,1,2,2,2]
AdjMat = make_matrix(PointsList, 0.9, 0.1) # creates a graph with 10 nodes, where the odds of a link between any two nodes are 90% if those nodes were assigned the same cluster, and 10% if those nodes were assigned different clusters
# AdjMat is an adjacency matrix representing a directed, unweighted graph