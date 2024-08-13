import numpy as np
import math as math
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
import networkx as nx
from itertools import combinations

# this code uses the methodology outlined in "On the detection of transitive clusters in undirected networks" by M. Perry to cluster undirected networks
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


#function to find number of edges given number of vertices for a complete graph
def findEdges(v):
    e = .5*v*(v-1)
    return e

#function to find the number of triangles in a complete graph with v vertices
def findTriangles(v):
    tria = v*(v-1)*(v-2)/6
    return tria

#function to cluster given number nodes, sigma, E(T), and T of a graph network
def PerryCluster(Sig, E, T, n):
    cluster = 1
    for k in range(2, n):
        print('lol')
    return cluster


def findZ(graph, edgelist, n):
    # find the z value given the graph, edges list of graph, and number of nodes
    k_edges = findEdges(n) #find number of edges in the complete graph
    k_tria = findTriangles(n) # find number of triangles in the complete graph
    num_edges = len(edgelist) # find number of edges in our graph
    Transitivity = nx.transitivity(graph) # percentage of triangles that could exist that do exist in our graph
    numTriangles = round(Transitivity*k_tria) # number of triangles in our graph = T
    p = num_edges / k_edges # probability that an edge exists for SH graph
    eT = p**3*len(list(combinations(vertices_list,3))) # find E(t)
    var = len(list(combinations(vertices_list,3)))*((3*n-9)*p**5-(3*n-8)*p**6+p**3) # variance
    sigma = math.sqrt(var) #stdev
    Z = (numTriangles - eT)/sigma # find z value
    return Z

# analysis on K13
n = 13; # number of vertices
k13tria = findTriangles(n)
k13edges = findEdges(n)
vertices_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # list of vertices for use in the combinations function


# sharpstone undirected network
sGraph = nx.Graph()
sGraph.add_node(13)
edgesSH = [(1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), (1,12), (1,13), (2,3), (2,4), (2,5), (2,7), (2,8), (2,10), (2,12), (2,13),
         (3,5), (3,6), (3,7), (3,8), (3,9), (3,10), (3,11), (3,12), (4,5), (4,7), (5,6), (6,8), (6,11), (8,9), (8,13), (9,11)]
num_edgesSH = len(edgesSH)
sGraph.add_edges_from(edgesSH)
sharpTransit = nx.transitivity(sGraph) # percentage of triangles that could exist that do exist in our graph
sharpTriangles = round(sharpTransit*k13tria)
print('Sharpstone Transitivity:')
print(sharpTransit)
print('Number of Triangles:')
print(sharpTriangles)
Z1 = findZ(sGraph, edgesSH, 13)
print('A priori Z value')
print(Z1)


# TI undirected network
tiGraph = nx.Graph()
tiGraph.add_node(13)
edgesTI = [(2,1), (3,1), (5,1), (6,1), (10,1), (11,1), (12,1), (3,2), (6,2), (7,2), (8,2), (9,2), (11,2), (5,3), (6,3), (11,3), (5,4), (10,4), (12,4), (13,4),
           (6,5), (13,5), (8,7), (9,7), (9,8), (11,10), (12,10)]
num_edgesTI = len(edgesTI)
tiGraph.add_edges_from(edgesTI)
tiTransit = nx.transitivity(tiGraph)
tiTriangles = round(tiTransit*k13tria)
print('TI Transitivity:')
print(tiTransit)
print('Number of Triangles:')
print(tiTriangles)
Z2 = findZ(tiGraph, edgesTI, 13)
print('A priori Z value')
print(Z2)


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
# directed TI data finding transitivity

directedTIedges = [(2,1), (3,1), (5,1), (6,1), (10,1), (11,1), (12,1), (6,2), (7,2), (8,2), (9,2), (11,2), (1,3), (2,3), (5,3), (6,3), (10,4), (12,4), (13,4),
                   (1,5), (4,5), (13,5), (1,6), (3,6), (5,6), (8,7), (9,7), (7,8), (9,8), (7,9), (8,9), (11,10), (12,10), (2,11), (3,11), (4,12), (4,13)]

# Create an adjacency matrix sharpstone
matrix = leaguedatasharpstone - np.identity(13)
#doing the same for TI
matrixTI = leaguedataTI - np.identity(13)


#TI symmetricized data finding transitivity
tiGraph = nx.Graph()
tiGraph.add_node(13)
edgesTI = [(2,1), (3,1), (5,1), (6,1), (10,1), (11,1), (12,1), (3,2), (6,2), (7,2), (8,2), (9,2), (11,2), (5,3), (6,3), (11,3), (5,4), (10,4), (12,4), (13,4),
           (6,5), (13,5), (8,7), (9,7), (9,8), (11,10), (12,10)]
tiGraph.add_edges_from(edgesTI) 
tiTriangles = nx.transitivity(tiGraph)
print('TI Transitivity:')
print(tiTriangles)

edgesK13 = findEdges(13)
# finding pairing numbers for SH and TI symmetric networks
SHpairs_symm = edgesK13*sharpTriangles
TIpairs_symm = edgesK13*tiTriangles
print('Sharpstone Symmetric Pairings:')
print(SHpairs_symm)
print('TI Symmetric Pairings:')
print(TIpairs_symm)


#pseudocode
# 1. find T, number of triangles observed in the test matrix
# - use C(n, 3) to find number of triangles on the complete graph (a)
# - use transitivity() on our graph (b)
# - calculate number of triangles (a*b)
# 2. find E(t) = C(n)p^3, where p is probability of an edge existing and C(n) is [n 3] (number of expected triangles in a
# random matrix with same number nodes and edges)
# - determine p: what percentage of edges exist on the test matrix that would exist on complete graph with x nodes
# 3. find variance for E(T)
# - take sqrt variance


# 4. find Z for each cluster using E, T, and sigma
# 5. loop through possible values of k: start with smallest logical value, calculate Z, add 1, repeat, until we pass the hypothesis test 
# that the existing cluster can be random