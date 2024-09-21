
from Tools import GraphVisualizer
from datasets import *
from Tools import GraphCluster as cluster

# Example Linkage clustering
sharplinks = cluster.linkage_cluster_agglom(leaguedatasharpstone)
tilinks = cluster.linkage_cluster_agglom(leaguedataTI)

# Hints to use the function with the example dataset:
# second number stays zero to access the labels from the label list
# first number is which set of labels you want to access, starting from 0 (0 being the labels with 2 clusters, ward linkage (etc))
# ctrl p to save a png of current graph picture window
# ctrl r to resize the window 
# ctrl s to save the positioning
# space pauses the simulation

# The following two lines are how the graph linkage visualizations for the paper were created.
GraphVisualizer.make_graph(leaguedatasharpstone,sharplinks[9][0])
GraphVisualizer.make_graph(leaguedataTI,tilinks[6][0])


