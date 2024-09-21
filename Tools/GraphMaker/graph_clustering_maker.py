import random
import numpy as np

def GeneratePartitionedPointsList(PartitionList): # takes in partition list and writes a list of nodes in clusters from it 
    PointsList = []
    for ClassIndex, ClassSize in enumerate(PartitionList):
        for i in range(ClassSize):
            PointsList.append(ClassIndex)
    return PointsList

def MakeAdjacencyMatrix(PointsList, InteriorChance, ExteriorChance): # makes an ajancency matrix using the generatepartitionedpointslist fnctn
    AdjMat = np.zeros((len(PointsList), len(PointsList)), dtype=int)
    for Xi, Xcls in enumerate(PointsList):
        for Yi, Ycls in enumerate(PointsList):
            r = random.random()
            if (Xcls == Ycls and r < InteriorChance) or (Xcls != Ycls and r < ExteriorChance):
                AdjMat[Xi, Yi] = 1
    return AdjMat