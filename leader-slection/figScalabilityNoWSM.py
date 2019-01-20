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
    runMode = False
    
    
    if (runMode == True):
        #numNodes = np.array([100,200,400,800,1600],dtype=int)
        numNodes = np.array([100,200,400,800],dtype=int)
        k = 10
        tOG = np.zeros(len(numNodes))
        tLG = np.zeros(len(numNodes))
        tSG = np.zeros(len(numNodes))
        
        
        for idx in range(len(numNodes)):
            
            n = numNodes[idx]
            pEdge = 3.00*np.log(n)/float(n)
            
            G = nf.createAnERGraph(n, pEdge)
            #G = nf.createABAGraph(n, 10)
        
            print "n = ", n, "kAvg = ",  2.00*nx.number_of_edges(G)/float(n)
            
            start = time()
            ipNodes1, nF1 = nf.runGreedyNF(G,k)
            tOG[idx] = time()-start
            print "n = ", n, " Ordinary greedy and direct inverse : ", tOG[idx], " seconds"
            
            start = time()
            ipNodes3, nF3 = nf.runLazyGreedyNF(G,k)
            tLG[idx] = time()-start
            print "n = ", n, " Lazy greedy and direct inverse : ", tLG[idx], " seconds"
            
            start = time()
            ipNodes5, nF5 = nf.runStochGreedyNF(G, k, 0.01)
            tSG[idx] = time()-start
            print "n = ", n, " Stochastic greedy and direct inverse : ", tSG[idx], " seconds"
            
        #Save the data
        
        outfile = "npzScalabilityNoWSM.npz"
        np.savez(outfile, numNodes=numNodes, tOG = tOG, tLG = tLG, tSG = tSG, k=k)
        print "Saved all data"
    
    else:

        outfile = "npzScalabilityNoWSM.npz"
        npzfile = np.load(outfile)
        numNodes= npzfile['numNodes']
        tOG = npzfile['tOG']
        tLG = npzfile['tLG']
        tSG = npzfile['tSG']
        
        
    
    plt.figure()
    plt.loglog(numNodes,tOG,'ro',markersize = 12,linewidth=5,linestyle = 'solid', label="Ordinary greedy")
    plt.loglog(numNodes,tLG,'g+',markersize = 12,linewidth=5,linestyle = 'solid',label="Lazy greedy")
    plt.loglog(numNodes,tSG,'bv',markersize = 12, linewidth=5,linestyle = 'solid',label="Stochastic greedy")
    plt.xlabel("Number of nodes in the network",fontsize=44)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    plt.ylabel("Time taken by the algorithm (seconds) ",fontsize=44)
    plt.grid()
    plt.legend(loc="lower right",fontsize=40)
    plt.show()