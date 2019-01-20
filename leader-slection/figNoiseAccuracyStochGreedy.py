import numpy as np
import random
import nfGreedyModule as nf
from time import time
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

if __name__ == "__main__":
    
    np.random.seed(0)
    random.seed(0)
    np.set_printoptions(precision=3,suppress=True)
    runMode = False
    
    
    if (runMode == True):
        
        n = 1000
        k = 12
        epsVals = [0.01,0.1,0.2,0.5]
        
        tOG = np.zeros(len(epsVals))
        tSG = np.zeros(len(epsVals))
        
        nFSG = np.zeros((len(epsVals),k+1))
        
            
        pEdge = 3.00*np.log(n)/float(n)
        
        #G = nf.createAnERGraph(n, pEdge)
        G = nf.createABAGraph(n, 10)
    
        print "n = ", n, "kAvg = ",  2.00*nx.number_of_edges(G)/float(n)
        
        start = time()
        ipNodesG, nFOG = nf.runGreedyNF(G,k)
        tOG = time()-start
        print "Ordinary greedy and direct inverse : ", tOG, " seconds"
            
        for idx in range(len(epsVals)):
            
            eps = epsVals[idx]
            
            start = time()
            ipNodesSG, nFSG[idx,:] = nf.runStochGreedyNF(G, k, eps)
            tSG[idx] = time()-start
            print "eps = ", eps , " Stochastic greedy and direct inverse : ", tSG[idx], " seconds"
            
        #Save the data
        
        outfile = "./npz_files_backup/npzNoiseAccuracyStochGreedyBA.npz"
        np.savez(outfile, n=n, tOG = tOG, tSG = tSG, k=k, nFOG = nFOG, nFSG=nFSG)
        print "Saved all data"
    
    else:

        outfile = "./npz_files/npzNoiseAccuracyStochGreedy.npz"
        npzfile = np.load(outfile)
        n = npzfile['n']
        k = npzfile['k']
        tOG = npzfile['tOG']
        tSG = npzfile['tSG']
        nFOG = npzfile['nFOG']
        nFSG = npzfile['nFSG']
        
        
   
    plt.figure()
    plt.plot(np.linspace(1,k,k,dtype=int),nFOG[1:],'ro',markersize = 14,linewidth=4,linestyle = 'solid', label="Ordinary Greedy")
    plt.plot(np.linspace(1,k,k,dtype=int),nFSG[0,1:],'bx',markersize = 14,linewidth=4,linestyle = 'dashed', label="Stochastic Greedy  "+r"$\epsilon$ = 0.01")
    plt.plot(np.linspace(1,k,k,dtype=int),nFSG[1,1:],'cD',markersize = 14,linewidth=4,linestyle = 'dashed', label="Stochastic Greedy  "+r"$\epsilon$ = 0.1")
    plt.plot(np.linspace(1,k,k,dtype=int),nFSG[2,1:],'ms',markersize = 14,linewidth=4,linestyle = 'dashed', label="Stochastic Greedy  "+r"$\epsilon$ = 0.3")
    plt.plot(np.linspace(1,k,k,dtype=int),nFSG[3,1:],'kv',markersize = 14,linewidth=4,linestyle = 'dashed', label="Stochastic Greedy  "+r"$\epsilon$ = 0.5")
    plt.xlim([1,12])
    
    plt.xlabel("Leader selection iteration",fontsize=44)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    plt.ylabel("Noise variance magnitude",fontsize=44)
    plt.grid()
    plt.legend(loc="upper right",fontsize=48)
    plt.show()
    
    spdUps = float(tOG)/tSG
    print n
    print spdUps