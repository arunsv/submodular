import numpy as np
import random
import nfGreedyModule as nf
from time import time
import networkx as nx

def twoStageGreeDi():
    
    #Two stage distributed greedy 
    partitions = G.graph['partition']
    ckNodes = []
    timeStage1 = 0.0
    for part in partitions:
        
        print "Processing partition"
        Gsub = G.subgraph(part)
        
        #Mapping so that nodes range from 0 to nodesC
        mapping = dict(zip(list(Gsub.nodes),range(nx.number_of_nodes(Gsub))))
        inv_mapping = {v: k for k, v in mapping.iteritems()}
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
    #ipNodesDi, nFDi = nf.runLazyGreedyNFScalableSubset(G,ckNodes,k)
    ipNodesDi, nFDi = nf.runStochGreedyNFScalableSubset(G,ckNodes,k,0.5)
    
    timeStage2 = time() - start
    total_time = timeStage1+timeStage2
    print "Time taken - two stage lazy greedy with WSM inverse = ", total_time
    
    return (ipNodesDi, nFDi, total_time)
    
    

if __name__ == "__main__":
    
    np.random.seed(0)
    random.seed(0)
    np.set_printoptions(precision=3,suppress=True)
    
    nodesC = 160
    c = 10
    k = 20
    
    pi = 3.00*np.log(nodesC)/float(nodesC)
    po = pi/10.0
    print "pin = ", pi, " pout = ", po
    
    G = nf.createAGraphWithERCommunities(c,nodesC,pi,po)
    print "Number of nodes : ", nx.number_of_nodes(G)
    print "Number of edges : ", nx.number_of_edges(G)


    start = time()
    ipNodes1, nF1 = nf.runGreedyNF(G,k)
    tOG = time()-start
    print "Time taken - Ordinary greedy and direct inverse : ", tOG


    #start = time()
    #ipNodes1, nF1 = nf.runGreedyNF(G,k)
    #print "Time taken - Ordinary greedy and direct inverse : ", time()-start
    
    #start = time()
    #ipNodes2, nF2 = nf.runGreedyNFScalable(G,k)
    #print "Time taken - Ordinary greedy and WSM inverse : ", time()-start
    
    #start = time()
    #ipNodes3, nF3 = nf.runLazyGreedyNF(G,k)
    #print "Time taken - Lazy greedy and direct inverse : ", time()-start
    
    #start = time()
    #ipNodes4, nF4 = nf.runLazyGreedyNFScalable(G,k)
    #print "Time taken - Lazy greedy and WSM inverse : ", time()-start    
    
    #start = time()
    #ipNodes5, nF5 = nf.runStochGreedyNF(G, k, 0.1)
    #print "Time taken - Stochastic greedy and direct inverse : ", time()-start
    
    
    #start = time()
    #ipNodes6, nF6 = nf.runStochGreedyNFScalable(G,k,0.5)
    #print "Time taken - Stochastic greedy and WSM inverse : ", time()-start
    
    
    #print ipNodes1
    #print ipNodesDi
        
    print
    print
    
    #print nF1[1:]
    #print nFDi[1:]
   