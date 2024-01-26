import numpy as np
import os


num_components = 16

class ComponentGraph:
    def __init__(self) -> None:
        self.graphDict = dict()
    
    def addConnection(self, t : tuple):
        if ( not t[0] in self.graphDict):
            self.graphDict[t[0]] = list()
        self.graphDict[t[0]].append(t[1])
    
    def checkTuple(self, t: tuple):
        if ( t[0] in self.graphDict):
            return t[1] in self.graphDict[t[0]]
        return False
    
    def areNeighbors(self, t: tuple):
        return self.checkTuple(t) or self.checkTuple((t[1],t[0]))

def buildGraph():
    graph = ComponentGraph()
    graph.addConnection((0,1))
    graph.addConnection((0,3))
    graph.addConnection((1,3))
    graph.addConnection((0,2))
    graph.addConnection((2,3))
    graph.addConnection((0,4))
    graph.addConnection((2,5))
    graph.addConnection((3,4))
    graph.addConnection((4,5))
    graph.addConnection((4,6))
    graph.addConnection((4,7))
    graph.addConnection((5,10))
    graph.addConnection((5,6))
    graph.addConnection((6,7))
    graph.addConnection((6,9))
    graph.addConnection((6,10))
    graph.addConnection((7,8))
    graph.addConnection((7,9))
    graph.addConnection((9,10))
    graph.addConnection((10,11))
    graph.addConnection((12,13))
    graph.addConnection((12,14))
    graph.addConnection((13,14))
    graph.addConnection((13,15))
    graph.addConnection((14,15))

    return graph

graph = buildGraph()

def AreElectrodesNeighbors(num_a, num_b):
    return graph.areNeighbors((num_a, num_b))

def AreElectrodesSecondNeighbors(num_a, num_b):
    if(AreElectrodesNeighbors(num_a,num_a)):
        return True
    # is there a node that is neighbor of us both:
    possibleNeighbors = list(range(0,num_components))
    possibleNeighbors.remove(num_a)
    possibleNeighbors.remove(num_b)
    for n in possibleNeighbors:
        if ( AreElectrodesNeighbors(num_a, n) and AreElectrodesNeighbors(num_b, n) ):
            return True
    return False

# KEEP_CHECKING = 0
# PROBLEM = 1
# COMPONENT_OK = 2

# def AreWeightsValid(electrodesVec, sortedVec, j):
#     if ( electrodesVec[sortedVec[j]] - electrodesVec[sortedVec[j+1]] < 0.1 ):
#         if ( AreElectrodesNeighbors(sortedVec[j],sortedVec[j+1] )):
#             return KEEP_CHECKING
#         return PROBLEM
#     return COMPONENT_OK


# if going by pic - maybe have a standard pic of heatmap, and then determine by a 
# hist of the "red areas" size. a big size means that it's noisy.

def get_noise_components_by_sort_w(w):
    new_w = np.absolute(np.linalg.inv(w)) 
    log_w = np.log(np.transpose(new_w))
    components_to_remove = []
    
    for i in range(0,num_components):
        allow_once = True
        log_w_sorted_indexes = np.argsort(log_w[i])[::-1]
        threshold = 0.1 #np.std(log_w[i][log_w_sorted_indexes[0:4]]) # was 0.1
        for j in range(0,num_components//3):
            if ( log_w[i][log_w_sorted_indexes[j]] - log_w[i][log_w_sorted_indexes[j+1]] < threshold ):
                #used to be just AreElectrodesSecondNeighbors
                if ( AreElectrodesNeighbors(log_w_sorted_indexes[j],log_w_sorted_indexes[j+1] )):
                    continue
                else:
                    if ( allow_once and AreElectrodesSecondNeighbors(log_w_sorted_indexes[j],log_w_sorted_indexes[j+1])):
                        allow_once = False
                        continue
                    components_to_remove += [i]
                    break
            else:
                break
        if ( j == ((num_components//3) - 1) and (not i in components_to_remove)): # 5 really close components:
            components_to_remove += [i]

    return components_to_remove

def GetNoiseComponentsBySort(data_path, current_session):
    w_path = os.path.join(data_path, current_session, "weightsAfterSharedICA.csv")
    w = np.loadtxt(w_path, delimiter=",")
    return get_noise_components_by_sort_w(w)


if __name__ == "__main__":
    print(np.array(GetNoiseComponentsBySort(r"C:\Liron\DataEmg\Try","04122022_1545")) + 1)
    print(np.array(GetNoiseComponentsBySort(r"C:\Liron\DataEmg\Try","06122022_1415")) + 1)
    print(np.array(GetNoiseComponentsBySort(r"C:\Liron\DataEmg\Try","13122022_1215")) + 1)

    # print(AreElectrodesNeighbors(15,4))
    # print(AreElectrodesNeighbors(14,13))
    # print(AreElectrodesNeighbors(10,11))
