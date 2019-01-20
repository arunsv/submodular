import numpy as np
import random
import nfGreedyModule as nf
from time import time

if __name__ == "__main__":
    
    np.random.seed(0)
    random.seed(0)
    np.set_printoptions(precision=3,suppress=True)
    
    numNodes = 10000
    k = 10
    pEdge = 2.00*np.log(numNodes)/float(numNodes)
    print "pEdge = ", pEdge
    
    
    ##G = nf.createAnERGraph(numNodes, pEdge)
    #G = nf.createABAGraph(numNodes, 10)
    
    ##start = time()
    ##ipNodes1, nF1 = nf.runGreedyNF(G,k)
    ##print "Time taken - Ordinary greedy and direct inverse : ", time()-start
    
    ##start = time()
    ##ipNodes2, nF2 = nf.runGreedyNFScalable(G,k)
    ##print "Time taken - Ordinary greedy and WSM inverse : ", time()-start
    
    ##start = time()
    ##ipNodes3, nF3 = nf.runLazyGreedyNF(G,k)
    ##print "Time taken - Lazy greedy and direct inverse : ", time()-start
    
    #start = time()
    #ipNodes4, nF4 = nf.runLazyGreedyNFScalable(G,k)
    #print "Time taken - Lazy greedy and WSM inverse : ", time()-start    
    
    start = time()
    ipNodes5, nF5 = nf.runStochGreedyNF(G, k, 0.1)
    print "Time taken - Stochastic greedy and direct inverse : ", time()-start
    
    
    start = time()
    ipNodes6, nF6 = nf.runStochGreedyNFScalable(G,k,0.5)
    print "Time taken - Lazy stochastic greedy and WSM inverse : ", time()-start
    
    
    print ipNodes4
    print ipNodes6
        
    print
    print
    
    print nF4[1:]
    print nF6[1:]
   