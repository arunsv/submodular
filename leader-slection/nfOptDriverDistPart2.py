import numpy as np
import random
import nfGreedyModule as nf
from time import time
import networkx as nx

def twoStageGreeDi(G,k):
    
    #Two stage distributed greedy 
    partitions = G.graph['partition']
    ckNodes = []
    timeStage1 = 0.0
    for part in partitions:
        
        print "Processing partition"
        Gsub = G.subgraph(part)
        
        #Mapping so that nodes range from 0 to nodesC
        mapping = dict(zip(list(Gsub.nodes),range(nx.number_of_nodes(Gsub))))
        inv_mapping = {v: k1 for k1, v in mapping.iteritems()}
        H = nx.relabel_nodes(Gsub,mapping)
                
        start = time()
        #kSubmap, nFnodesSub = nf.runLazyGreedyNFScalable(H,k)
        kSubmap, nFnodesSub = nf.runStochGreedyNFScalable(H,k,0.5)
        kSub = []
        for nodemapped in kSubmap:
            kSub.append(inv_mapping[nodemapped])
        
        timeStage1 += (time()-start)
        ckNodes = ckNodes + kSub
    
    print "Second stage selection : "
    timeStage2 = 0.0
    start = time()
    
    ipNodesDi, nFDi = nf.runStochGreedyNFScalableSubset(G,ckNodes,k,0.5)
    
    timeStage2 = time() - start
    total_time = timeStage1+timeStage2
    print "Time taken - two stage lazy greedy with WSM inverse = ", total_time
    
    return (ipNodesDi, nFDi, total_time)
    
    

if __name__ == "__main__":
    

    outfile = "npzGreeDiOG.npz"
    npzfile = np.load(outfile)
    
    nodesC = npzfile['nodesC']
    c = npzfile['c']
    k = npzfile['k']
    G = npzfile['G']
    pi = npzfile['pi']
    po = npzfile['po']
    nFOG = npzfile['nFOG']
    tOG = npzfile['tOG']    
       
    ipNodesDi, nFDi, tHSOG = twoStageGreeDi(G,k)