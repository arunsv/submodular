import numpy as np
import random
import nfGreedyModule as nf
from time import time
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import linregress


def find_ls_fit(lin,n):
    
    a = lin.slope
    b = lin.intercept
    
    y = np.exp(a*np.log(n)+b)
    
    return y
    
if __name__ == "__main__":
    
    np.random.seed(0)
    random.seed(0)
    np.set_printoptions(precision=3,suppress=True)
    runMode = False
    
    
    if (runMode == True):
        numNodes = np.array([200,400,800,1600],dtype=int)
        k = 10
        tOG = np.zeros(len(numNodes))
        tLG = np.zeros(len(numNodes))
        tSG = np.zeros(len(numNodes))
        frac = 0.05
        
        for idx in range(len(numNodes)):
            
            n = numNodes[idx]
            k = int(frac*n)
            pEdge = 3.00*np.log(n)/float(n)
            
            G = nf.createAnERGraph(n, pEdge)
            #G = nf.createABAGraph(n, 10)
        
            print "n = ", n, "kAvg = ",  2.00*nx.number_of_edges(G)/float(n)
            
            start = time()
            ipNodes1, nF1 = nf.runGreedyNF(G,k)
            tOG[idx] = time()-start
            print "n = ", n, " Ordinary greedy and direct inverse : ", tOG[idx], " seconds"
            
            start = time()
            ipNodes3, nF3 = nf.runLazyGreedyNFScalable(G,k)
            tLG[idx] = time()-start
            print "n = ", n, " Lazy greedy and WSM inverse : ", tLG[idx], " seconds"
            
            start = time()
            ipNodes5, nF5 = nf.runStochGreedyNFScalable(G, k, 0.5)
            tSG[idx] = time()-start
            print "n = ", n, " Stochastic greedy and WSM inverse : ", tSG[idx], " seconds"
            
        #Save the data
        
        outfile = "npzScalabilityWSM-prop-WS.npz"
        np.savez(outfile, numNodes=numNodes, tOG = tOG, tLG = tLG, tSG = tSG, k=k, nfOG=nF1, nfLG = nF3, nfSG = nF5)
        print "Saved all data"
    
    else:

        outfile = "npzScalabilityWSM-prop-WS.npz"
        npzfile = np.load(outfile)
        numNodes= npzfile['numNodes']
        tOG = npzfile['tOG']
        tLG = npzfile['tLG']
        tSG = npzfile['tSG']
        #nfOG = npzfile['nfOG']
        #nfLG = npzfile['nfLG']
        #nfSG = npzfile['nfSG']
    
    linOG = linregress(np.log(numNodes), np.log(tOG))
    linLG = linregress(np.log(numNodes), np.log(tLG))
    linSG = linregress(np.log(numNodes), np.log(tSG))
    
    tOGls = find_ls_fit(linOG,numNodes)
    tLGls = find_ls_fit(linLG,numNodes)
    tSGls = find_ls_fit(linSG,numNodes)
    
    plt.figure()
    plt.plot(numNodes,tOGls,'r',linewidth=5,linestyle = 'solid', label="Ordinary greedy")
    plt.plot(numNodes,tLGls,'g',linewidth=5,linestyle = 'dashed',label="Lazy greedy")
    plt.plot(numNodes,tSGls,'b',linewidth=5,linestyle = 'dashed',label="Stochastic greedy")
    
    plt.plot(numNodes,tOG,'ro',markersize = 12,linewidth=5)
    plt.plot(numNodes,tLG,'gs',markersize = 12,linewidth=5)
    plt.plot(numNodes,tSG,'bv',markersize = 12, linewidth=5)    

    plt.xlabel("Number of nodes in the network",fontsize=40)
    plt.ylabel("Algorithm execution time" + "\n" + "(seconds) ",fontsize=40)
    plt.grid()    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc="upper left",fontsize=32)
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.75, top=0.75)    
    plt.xlim([200,2000])
        
    ax = plt.gca()
    ax.set_xticks(numNodes)
    ax.set_xticklabels(numNodes)    
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)


    #plt.show()
    

    print "Ordinary greedy : ", linOG.slope, np.exp(linOG.intercept)
    print "Lazy greedy : ", linLG.slope, np.exp(linLG.intercept)
    print "Stoch greedy : ", linSG.slope, np.exp(linSG.intercept)   
    
    #print tOG/tLG
    #print tOG/tSG
    
    #print tOG
    #print tLG
    #print tSG
    
    #print (nfSG-nfOG)*100.0/nfOG
    
    print find_ls_fit(linSG,np.array([20000,100000]))/86400.0
    