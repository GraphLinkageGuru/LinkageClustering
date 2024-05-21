# unfinished according to Sam

import random
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import time
from math import log2

PartitionList = [3, 4, 5, 6]


def DummyGradingScheme(truelist, testlist): #These are lists of vertices labeled with cluster number e.g. [1,1,0,0,0,2,2,2,2] for clusters {2,3,4}, {0,1}, {5,6,7,8}
    if len(truelist) != len(testlist):
        raise Exception("you goober that is not the same number of vertices")
    mistakes = 0
    for i in range(len(truelist)):
        for j in range(i):
            if (truelist[i]==truelist[j]) ^ (testlist[i]==testlist[j]):
                mistakes+=1
    return 1 - mistakes / (len(truelist) * (len(truelist)-1) / 2.0)

def AdjustedMutualInformation(truelist, testlist): #These are lists of vertices labeled with cluster number e.g. [1,1,0,0,0,2,2,2,2] for clusters {2,3,4}, {0,1}, {5,6,7,8}
    #                                               This is the pre-adjustment one; it needs to have the second half added
    if len(truelist) != len(testlist):
        raise Exception("you goober that is not the same number of vertices")

    U=[set() for i in range(max(truelist)+1)]
    for i in range(len(truelist)):
        U[truelist[i]].add(i)

    V=[set() for i in range(max(testlist)+1)]
    for i in range(len(testlist)):
        V[testlist[i]].add(i)

    AMI=0
    N=len(truelist)
    for i in range(len(U)):
        for j in range(len(V)):
            uv = len(U[i].intersection(V[j]))
            if uv>0:
                AMI+=uv/N * log2((uv*N) / (len(U[i])*len(V[j])))
    return AMI

def GeneratePartitionedPointsList(PartitionList):
    PointsList = []
    for ClassIndex, ClassSize in enumerate(PartitionList):
        for i in range(ClassSize):
            PointsList.append(ClassIndex)
    return PointsList

def MakeAdjacencyMatrix(PointsList, InteriorChance, ExteriorChance):
    AdjMat = np.zeros((len(PointsList), len(PointsList)), dtype=int)
    for Xi, Xcls in enumerate(PointsList):
        for Yi, Ycls in enumerate(PointsList):
            r = random.random()
            if (Xcls == Ycls and r < InteriorChance) or (Xcls != Ycls and r < ExteriorChance):
                AdjMat[Xi, Yi] = 1
    return AdjMat

steps = 10
trials = 10
start = time.time()
ptList = GeneratePartitionedPointsList(PartitionList)
random.shuffle(ptList)
print(ptList)
propagation_cluster2 = AgglomerativeClustering(n_clusters=len(PartitionList), metric='euclidean', linkage='complete')
#propagation_cluster2 = SpectralClustering(n_clusters=len(PartitionList),affinity="precomputed")
GradeMat = np.zeros((steps+1, steps+1), dtype=float)
for inner in range(0, steps+1):
    for outer in range(0, steps+1):
        for trial in range(trials):
            AdjMat = MakeAdjacencyMatrix(ptList, inner / steps, outer / steps)
            labels = propagation_cluster2.fit_predict(AdjMat)
            GradeMat[inner, outer] += AdjustedMutualInformation(ptList, labels)
        GradeMat[inner, outer] /= trials
print(f"Finished in {time.time()-start}")
X, Y = np.meshgrid(np.linspace(0, 1, steps+1), np.linspace(0, 1, steps+1))
plt.pcolor(X, Y, GradeMat)
plt.colorbar()
plt.show()