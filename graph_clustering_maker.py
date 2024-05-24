import random
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import time
from math import log2
from math import factorial as fact
from functools import reduce

PartitionList = [3, 4, 5, 6] # make partitions for a graph with 1 cluster of size 3, 1 of size 4, 1 of size 5, 1 of size 6


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
    N=len(truelist)

    U=[set() for i in range(max(truelist)+1)] #create a list of clusters by index; if the partition is [0,1,0,2,0,2] then U is [{0, 2, 4}, {1}, {3, 5}]
    for i in range(len(truelist)):
        U[truelist[i]].add(i)
    HU = -reduce(lambda x,y: x+(y*log2(y)), [len(Ui)/N for Ui in U], 0)

    V=[set() for i in range(max(testlist)+1)]
    for i in range(len(testlist)):
        V[testlist[i]].add(i)
    HV = -reduce(lambda x,y: x+(y*log2(y)), [len(Vi)/N for Vi in V], 0)

    n=np.zeros((N, N), dtype=int) #contingency table
    a=np.zeros((N), dtype=int)    #marginal sums over 1st...
    b=np.zeros((N), dtype=int)    #...and 2nd index
    for i in range(len(U)):
        for j in range(len(V)):
            uv = len(U[i].intersection(V[j]))
            n[i,j]=uv
            a[i]+=uv
            b[j]+=uv
    MI=0 #Mutual Information
    EMI=0 #Expected Mutual Information
    for i in range(len(U)):
        for j in range(len(V)):
            if n[i,j]>0:
                MI+=n[i,j]/N * log2((n[i,j]*N) / (len(U[i])*len(V[j])))
            for nij in range(int(max(0, a[i]+b[j]-N)), int(min(a[i], b[j])+1)):
                if N*nij/(a[i]*b[j]) > 0:
                    EMI += (nij/N)*log2(N*nij/(a[i]*b[j]))*(fact(a[i])*fact(b[j])*fact(N-a[i])*fact(N-b[j]))/(fact(N)*fact(nij)*fact(a[i]-nij)*fact(b[j]-nij)*fact(N-a[i]-b[j]+nij))
    AMI = (MI-EMI)/(max(HU, HV)-EMI)
    return AMI

def GeneratePartitionedPointsList(PartitionList): # takes in partition list and writes a list of nodes in clusters from it 
    PointsList = []
    for ClassIndex, ClassSize in enumerate(PartitionList):
        for i in range(ClassSize):
            PointsList.append(ClassIndex)
    return PointsList

def MakeAdjacencyMatrix(PointsList, InteriorChance, ExteriorChance): # makes an ajancency matrix using the generate partitioned points list fnctn
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
