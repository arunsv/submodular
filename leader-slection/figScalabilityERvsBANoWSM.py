import numpy as np
import random
import nfGreedyModule as nf
from time import time
import matplotlib.pyplot as plt
import networkx as nx

if __name__ == "__main__":
    
    np.random.seed(0)
    random.seed(0)
    np.set_printoptions(precision=3,suppress=True)
    runMode = True
    
    
    if (runMode == True):
        #numNodes = np.array([100,200,400,800,1600],dtype=int)
        numNodes = np.array([100,200,400,800],dtype=int)
        k = 10

        tLGER = np.zeros(len(numNodes))
        tLGBA = np.zeros(len(numNodes))
        
        
        for idx in range(len(numNodes)):
            
            n = numNodes[idx]
            pEdge = 3.00*np.log(n)/float(n)
            
            G1 = nf.createAnERGraph(n, pEdge)
            G2 = nf.createABAGraph(n, 10)
        
            print "n = ", n, "kAvg (ER) = ",  2.00*nx.number_of_edges(G1)/float(n)
            print "n = ", n, "kAvg (BA) = ",  2.00*nx.number_of_edges(G2)/float(n)
            
            start = clock()
            ipNodes3, nF3 = nf.runLazyGreedyNF(G1,k)
            tLGER[idx] = time()-start
            print "n = ", n, " Lazy greedy and direct inverse (ER) : ", tLGER[idx], " seconds"
            
            start = time()
            ipNodes3, nF3 = nf.runLazyGreedyNF(G2,k)
            tLGBA[idx] = time()-start
            print "n = ", n, " Lazy greedy and direct inverse (BA) : ", tLGBA[idx], " seconds"            

            
        #Save the data
        
        outfile = "npzScalabilityERvsBANoWSM.npz"
        np.savez(outfile, numNodes=numNodes, tLGER = tLGER, tLGBA = tLGBA, k=k)
        print "Saved all data"
    
    else:

        outfile = "npzScalabilityERvsBANoWSM.npz"
        npzfile = np.load(outfile)
        numNodes= npzfile['numNodes']
        tLGER = npzfile['tLGER']
        tLGBA = npzfile['tLGBA']
        
        
    
    plt.figure()
    plt.loglog(numNodes,tLGER,'g+',markersize = 22,linewidth=5,linestyle = 'solid',label="Lazy greedy (ER)")
    plt.loglog(numNodes,tLGBA,'r+',markersize = 22,linewidth=5,linestyle = 'solid',label="Lazy greedy (SF-BA)")
    plt.xlabel("Number of nodes in the network",fontsize=44)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    plt.ylabel("Time taken by the algorithm (seconds) ",fontsize=44)
    plt.grid()
    plt.legend(loc="lower right",fontsize=40)
    plt.show()