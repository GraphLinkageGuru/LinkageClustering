import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
import networkx as nx
import scipy.stats as st
import CustomGraphVis.CustomGraph
from graph_clustering_maker import GeneratePartitionedPointsList as generate_list
from graph_clustering_maker import MakeAdjacencyMatrix as make_matrix # hee hee ha
from graph_clustering_maker import AdjustedMutualInformation as adjust
import time
import random
from tqdm import tqdm

linkagenames = ['ward', 'single', 'complete', 'average']
#function to find number of edges given number of vertices for a complete graph
def findEdges(v):
    e = .5*v*(v-1)
    return e

# functions to simplify clustering the datasets multiple times
# used for the kiddo datasets
def linkageClusterAgglom(dataset):
    labels_list = []
    for n in range(2,5):
        for x in linkagenames:
            agglom_cluster = AgglomerativeClustering(n_clusters=n, metric='euclidean', linkage=x)
            labels = agglom_cluster.fit_predict(dataset)
            labels_list.append([labels, n, x])
    return labels_list

def linkageClusterAff(dataset):
    labels_list = []
    affin_cluster = AffinityPropagation()
    labels = affin_cluster.fit_predict(dataset)
    labels_list.append([labels])
    return labels_list



#pg 141 dataset
# right now I am dealing with two versions: the non-symmetric directionalized data and the symmetric non-directionalized data
# Euclidian distance, Ward linkage?
# plan A is the Louvain clustering algorithm

# non-symmetric dataset 1st little league data
leaguedatasharpstone = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
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
#non-symmetric second little league data
leaguedataTI = [[1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

# using networkx to calculate transitivity on sharpstone symmetricized data
sGraph = nx.Graph()
sGraph.add_node(13)
edgesSH = [(1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), (1,12), (1,13), (2,3), (2,4), (2,5), (2,7), (2,8), (2,10), (2,12), (2,13),
         (3,5), (3,6), (3,7), (3,8), (3,9), (3,10), (3,11), (3,12), (4,5), (4,7), (5,6), (6,8), (6,11), (8,9), (8,13), (9,11)]
sGraph.add_edges_from(edgesSH) 
sharpTriangles = nx.transitivity(sGraph)
print('Sharpstone Transitivity:')
print(sharpTriangles)


# calculate transitivity on sharpstone directed dataset
directedS = nx.MultiDiGraph()
directedS.add_node(13)
directedges = [(2,1), (3,1), (4,1), (5,1), (6,1), (7,1), (8,1), (9,1), (10,1), (12,1), (13,1), (1,2), (3,2), (4,2), (5,2), (7,2), (8,2), (10,2), (12,2), (13,2),
               (1,3), (2,3), (5,3), (6,3), (7,3), (9,3), (10,3), (11,3), (12,3), (1,4), (2,4), (4,5), (6,5), (8,6), (11,6), (3,8), (9,8), (13,8), (11,9)]
directedS.add_edges_from(directedges)


#TI symmetricized data finding transitivity
tiGraph = nx.Graph()
tiGraph.add_node(13)
edgesTI = [(2,1), (3,1), (5,1), (6,1), (10,1), (11,1), (12,1), (3,2), (6,2), (7,2), (8,2), (9,2), (11,2), (5,3), (6,3), (11,3), (5,4), (10,4), (12,4), (13,4),
           (6,5), (13,5), (8,7), (9,7), (9,8), (11,10), (12,10)]
tiGraph.add_edges_from(edgesTI) 
tiTriangles = nx.transitivity(tiGraph)
print('TI Transitivity:')
print(tiTriangles)


# doing calculations for what the transitivity means
# I was right! The complete graph is denoted by K
# according to Theorem 2 in "Introduction to Graph Theory" by Trudeau, 
# the formula for finding the number of edges in a complete graph is e = .5*v*(v-1) 
# e=edges
# v=vertices

edgesK13 = findEdges(13)
# finding pairing numbers for SH and TI symmetric networks
SHpairs_symm = edgesK13*sharpTriangles
TIpairs_symm = edgesK13*tiTriangles
print('Sharpstone Symmetric Pairings:')
print(SHpairs_symm)
print('TI Symmetric Pairings:')
print(TIpairs_symm)

# directed TI data finding transitivity

directedTIedges = [(2,1), (3,1), (5,1), (6,1), (10,1), (11,1), (12,1), (6,2), (7,2), (8,2), (9,2), (11,2), (1,3), (2,3), (5,3), (6,3), (10,4), (12,4), (13,4),
                   (1,5), (4,5), (13,5), (1,6), (3,6), (5,6), (8,7), (9,7), (7,8), (9,8), (7,9), (8,9), (11,10), (12,10), (2,11), (3,11), (4,12), (4,13)]

# Create an adjacency matrix sharpstone
matrix = leaguedatasharpstone - np.identity(13)
#doing the same for TI
matrixTI = leaguedataTI - np.identity(13)



# testing different linkage methods on the sharpstone dataset
sharplinks = linkageClusterAgglom(leaguedatasharpstone)

#TI dataset
tilinks = linkageClusterAgglom(leaguedataTI)

# graph visualization for sharpstone
print('Sharpstone Dataset')
print('Agglomerative clustering')
for element in sharplinks:
    print(element)
print('TI Dataset')
print('Agglomerative clustering')
for element2 in tilinks:
    print(element2)
#CustomGraphVis.CustomGraph.MakeGraph(leaguedatasharpstone,sharplinks[9][0])
#CustomGraphVis.CustomGraph.MakeGraph(leaguedataTI,tilinks[6][0])
# second number stays zero to access the labels
# first number is which set of labels you want to access, starting from 0
# ctrl p to save a png of current graph picture window
# ctrl r to resize the window 
# ctrl s to save the positioning
# do ward vs single vs average vs complete for 3 clusters, I think that is the appropriate size for now 
# for both the sharpstone and TI dataset

# testing affinity propagation
print('Affinity propagation test')
affinityprop = linkageClusterAff(leaguedatasharpstone)
for element3 in affinityprop:
    print(element3)



def FindMutualInformation(truelist, clusterlist): # set up for 3 clusters, fix later
    ward = adjust(truelist, clusterlist[1][0])
    single = adjust(truelist, clusterlist[4][0])
    comp = adjust(truelist, clusterlist[7][0])
    avg = adjust(truelist, clusterlist[10][0])
    list = [ward, single, comp, avg]
    max_AMI = max(list)
    return list.index(max_AMI)


# Find the distibution of wins for each algorithm with a given partition list of form:
# [1, 5, 2, 6], where each element is the amount of items in that partition
def FindBestPerformer(partition_list, num_samples, caseName):
    truth_list = generate_list(partition_list)
    linkagePerformance = [[] for x in linkagenames]
    linkageTimes = {name:0 for name in linkagenames}

    sampleRatio = 1/num_samples

    for x in range(num_samples):
        syntheticMatrix = make_matrix(truth_list, .9, .2) # exterior is between clusters, interior is in a cluster

        for i,name in enumerate(linkagenames):
            startTime = time.time()
            agglom_cluster = AgglomerativeClustering(n_clusters=len(partition_list), metric='euclidean', linkage=name)
            labels = agglom_cluster.fit_predict(syntheticMatrix)

            timeTaken = time.time()-startTime
            linkageTimes[name] += timeTaken*sampleRatio
            linkagePerformance[i].append(adjust(truth_list, labels))
        
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

    return linkageTimes


# sharpstone
#truelistS = sharplinks[4][0] # I am still confused on how to make this...I kinda just guessed 
#wards = adjust(truelist, testlist)

# Test various partition configurations
# partitions!! :P

partition_even = [3, 3, 3, 3, 3, 3]
timesTaken = FindBestPerformer(partition_even,10000,"Partition Even")
print(timesTaken)
partition_uneven = [1, 3, 9, 7, 5, 11]
FindBestPerformer(partition_uneven,10000,"Partition Uneven")

partition_small_uneven = [2, 3, 1, 4]
FindBestPerformer(partition_small_uneven,10000,"Partition Small Uneven")

def getComputationTime(num_samples, figureName, minSize, maxSize, stepSize):
    countValues = []
    timesTaken = []
    
    for count in tqdm(range(minSize,maxSize,stepSize)):
        countValues.append(count)

        linkageTimes = [0 for name in linkagenames]
        partition_list = []
        while(len(partition_list)<count):
            partitionSize = random.randint(1,count-len(partition_list))
            partition_list.append(partitionSize)

        truth_list = generate_list(partition_list)

        for x in range(num_samples):
            syntheticMatrix = make_matrix(truth_list, .9, .2) # exterior is between clusters, interior is in a cluster

            for i,name in enumerate(linkagenames):
                startTime = time.time()
                agglom_cluster = AgglomerativeClustering(n_clusters=len(partition_list), metric='euclidean', linkage=name)
                labels = agglom_cluster.fit_predict(syntheticMatrix)

                timeTaken = time.time()-startTime
                linkageTimes[i] += timeTaken
                
        timesTaken.append([item/num_samples for item in linkageTimes])

    timesTaken = np.transpose(timesTaken)

    plt.figure(figureName)

    for i,name in enumerate(linkagenames):
        # Plotting the lines
        plt.plot(countValues, timesTaken[i], label=name)
    
    plt.title('Node Count vs Computation Time')
    # Adding labels
    plt.xlabel('Node Count')
    plt.ylabel('Time (seconds)')

    # Adding a legend
    plt.legend()

#getComputationTime(10000,"Time Comparison",5,50,5)

plt.show()