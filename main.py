import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

from Tools.GraphMaker.graph_clustering_maker import GeneratePartitionedPointsList as generate_list
from Tools.GraphMaker.graph_clustering_maker import MakeAdjacencyMatrix as make_matrix #create a synthetic adjacency matrix

# Bring in sharpstone and TI League datasets
from datasets import *

# Linkage types
linkagenames = ['ward', 'single', 'complete', 'average'] # four linkages used for Agglomerative Clustering

# Functions to simplify clustering the datasets multiple times
# used for the Little League Datasets
def linkageClusterAgglom(dataset): # takes in a dataset and clusters it for n=2 through 5 clusters, and for each of the four given linkages
    labels_list = [] # cluster labelings for each iteration, each node is given a number representing its cluster number
    for n in range(2,5):
        for x in linkagenames:
            agglom_cluster = AgglomerativeClustering(n_clusters=n, metric='euclidean', linkage=x) # set up agglomerative clustering
            labels = agglom_cluster.fit_predict(dataset) # apply labels to the existing dataset
            labels_list.append([labels, n, x]) # add current clustering scheme labels to label list
    return labels_list

# Find the distibution of wins for each algorithm with a given partition list of form:
# [1, 5, 2, 6], where each element is the amount of items in that partition
def FindBestPerformer(partition_list, num_samples, caseName):
    truth_list = generate_list(partition_list)
    linkagePerformance = [[] for x in linkagenames]

    for x in range(num_samples):
        syntheticMatrix = make_matrix(truth_list, .9, .2) # exterior is between clusters, interior is in a cluster

        for i,name in enumerate(linkagenames):
            agglom_cluster = AgglomerativeClustering(n_clusters=len(partition_list), metric='euclidean', linkage=name)
            labels = agglom_cluster.fit_predict(syntheticMatrix)

            linkagePerformance[i].append(metrics.adjusted_mutual_info_score(truth_list,labels,average_method='max'))
        
    plt.figure(caseName)
    performance = [sum(linkageType)/num_samples for linkageType in linkagePerformance]
    standardDeviation = [np.std(linkageType)/np.sqrt(num_samples) for linkageType in linkagePerformance]
    print("\n",caseName)
    for i in range(len(linkagenames)):            
        print("Linkage Method:",linkagenames[i],"    \t Mean:",performance[i],"\t Confidence Interval:",standardDeviation[i])
    plt.errorbar(linkagenames, performance, yerr=standardDeviation, fmt='o', color='k',ecolor = "black", capsize=3)
    plt.xlabel('Linkage Method')
    plt.ylabel('Average Adjusted Mutual Information Score')
    plt.title('Linkage Method vs Performance ['+str(caseName)+']')
    plt.ylim([0,1])
    for i, v in enumerate(performance):
        plt.text(i+(-0.25 if i == 3 else 0.05), v + 0.01, f'{v:.2f}')

# Evaluating different linkage methods on the sharpstone dataset and TI dataset from datasets.py
sharplinks = linkageClusterAgglom(leaguedatasharpstone)
tilinks = linkageClusterAgglom(leaguedataTI)

# Graph visualization for sharpstone
print('Sharpstone Dataset')
print('Agglomerative clustering')
for element in sharplinks:
    print(element)
    
print('TI Dataset')
print('Agglomerative clustering')
for element2 in tilinks:
    print(element2)

# Test various partition configurations
# partitions!! :P

partition_even = [3, 3, 3, 3, 3, 3]
FindBestPerformer(partition_even,10000,"Partition Even")

partition_uneven = [1, 3, 9, 7, 5, 11]
FindBestPerformer(partition_uneven,10000,"Partition Uneven")

partition_small_uneven = [2, 3, 1, 4]
FindBestPerformer(partition_small_uneven,10000,"Partition Small Uneven")

# Show graphs
plt.show()