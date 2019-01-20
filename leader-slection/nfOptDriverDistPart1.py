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
        
        #print "Processing partition"
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

    timeStage2 = 0.0
    start = time()
    ipNodesDi, nFDi = nf.runStochGreedyNFScalableSubset(G,ckNodes,k,0.5)
    
    timeStage2 = time() - start
    total_time = timeStage1+timeStage2
    
    return (ipNodesDi, nFDi, total_time)
    
    

if __name__ == "__main__":
    
    np.random.seed(0)
    random.seed(0)
    np.set_printoptions(precision=3,suppress=True)
    
    nodesC = 100
    c = 10
    k = 10
    pi = 0.1
    factors = [0.2,0.4,0.6,0.8,1.0]
    percent_nf = np.zeros((len(factors),k))
    
    for idx in range(len(factors)):
        
        po = pi*factors[idx]
        print "pin = ", pi, " pout = ", po
        
        G = nf.createAGraphWithERCommunities(c,nodesC,pi,po)
       
        start = time()
        ipNodesLG, nFLG = nf.runLazyGreedyNFScalable(G, k)
        tLG = time()-start
        print nFLG
        
        start = time()
        ipNodesDi, nFDi, tHSDG = twoStageGreeDi(G,k)
        print nFDi
        tDG = time()-start
        
        percent_nf[idx,:] = 100.0*(nFDi[1:]-nFLG[1:])/nFLG[1:]
        print
    
    outfile = "npzGreeDiNF.npz"
    np.savez(outfile, nodesC = nodesC, c = c, k = k, factors = factors, pi = pi, po = po, percent_nf = percent_nf )
    print "Saved all data"   
    
    print percent_nf
   