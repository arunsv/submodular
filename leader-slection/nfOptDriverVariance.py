import numpy as np
import random
import nfGreedyModule as nf
from time import time
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

if __name__ == "__main__":
    
    #np.random.seed(0)
    #random.seed(0)
    #np.set_printoptions(precision=3,suppress=True)
    
    runMode = False
    
    if (runMode == True):
    
        nMonte = 1000
        
        numNodes = 1000
        k = 12
        pEdge = 2.00*np.log(numNodes)/float(numNodes)
    
        nFVarER = np.zeros(nMonte)
        nFVarBA = np.zeros(nMonte)
    
        G1 = nf.createAnERGraph(numNodes, pEdge, sd = 10)
        G2 = nf.createABAGraph(numNodes, 10, sd = 10)    
    
        start = time()
        ipNodes, nFGER = nf.runLazyGreedyNFScalable(G1,k)
        ipNodes, nFGBA = nf.runLazyGreedyNFScalable(G2,k)
    
        for idx in range(nMonte):
            
            print  "Iteration # ", idx
          
            ipNodes, nFER = nf.runStochGreedyNFScalable(G1,k,0.5)
            ipNodes, nFBA = nf.runStochGreedyNFScalable(G2,k,0.5)
            
            nFVarER[idx] = 100.0*((nFER[-1]-nFGER[-1])/nFGER[-1])
            nFVarBA[idx] = 100.0*((nFBA[-1]-nFGBA[-1])/nFGBA[-1])
        
        print "Time : ", time()-start, " seconds."
          
        outfile = "npzNFVariance.npz"
        np.savez(outfile, numNodes = numNodes, nMonte = nMonte, k = k, nFGER = nFGER , nFGBA = nFGBA, nFVarER = nFVarER, nFVarBA = nFVarBA)
        print "Saved all data"    
   
    else:
        
        outfile = "npz_files/npzNFVariance1k.npz"
        npzFile = np.load(outfile)
        nFVarER = npzFile['nFVarER']
        nFVarBA = npzFile['nFVarBA']
    
    kdeER = gaussian_kde(nFVarER)    
    kdeBA = gaussian_kde(nFVarBA)
    x_grid = np.linspace(0.0,1.4,num= 500, dtype = np.float)
    
    plt.figure()
    plt.subplot(1,1,1)
    plt.plot(x_grid, kdeER.evaluate(x_grid), color = 'r', linewidth = 6, label = 'ER topology')
    plt.plot(x_grid, kdeBA.evaluate(x_grid), color = 'k', linewidth = 6, label = 'BA topology')
    plt.hist(nFVarER,bins = 30, color = 'r', linewidth = 4, histtype= 'stepfilled', alpha=0.2, normed = True)
    plt.hist(nFVarBA,bins = 30, color = 'k', linewidth = 4, histtype= 'stepfilled', alpha=0.2, normed = True)
    plt.xlabel('Percent deviation from the Ordinary Greedy solution',fontsize=40)
    plt.ylabel('PDF Value', fontsize = 40)
    plt.legend(loc="upper right",fontsize=40)
    plt.grid()

    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)    

    plt.show()