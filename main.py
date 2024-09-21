# Bring in sharpstone and TI League datasets
from datasets import *
import Tools.GraphCluster as cluster

# Evaluating different linkage methods on the sharpstone dataset and TI dataset from datasets.py
sharplinks = cluster.linkage_cluster_agglom(leaguedatasharpstone)
tilinks = cluster.linkage_cluster_agglom(leaguedataTI)

# Graph visualization for sharpstone and transatlantic
print('Sharpstone Dataset')
print('Agglomerative clustering')
for element in sharplinks:
    print(element)
    
print('TI Dataset')
print('Agglomerative clustering')
for element2 in tilinks:
    print(element2)

# Test various partition configurations to find the most accurate linkage
# partitions!! :P

partition_even = [3, 3, 3, 3, 3, 3]
cluster.find_best_performer(partition_even,10000,"Partition Even")

partition_uneven = [1, 3, 9, 7, 5, 11]
cluster.find_best_performer(partition_uneven,10000,"Partition Uneven")

partition_small_uneven = [2, 3, 1, 4]
cluster.find_best_performer(partition_small_uneven,10000,"Partition Small Uneven")