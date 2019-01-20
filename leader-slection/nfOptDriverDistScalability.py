import numpy as np
import random
import nfGreedyModule as nf
from time import time
import networkx as nx
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def func(x, a, b):
    return a * np.power(x,b)

def find_ls_fit(lin,n):
    
    a = lin.slope
    b = lin.intercept
    
    y = np.exp(a*np.log(n)+b)
    
    return y


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
    
    #np.random.seed(0)
    #random.seed(0)
    #np.set_printoptions(precision=3,suppress=True)
    
    #nodesC = [160, 200, 320, 800]
    #c = [10, 8, 5, 2]
    #tGreeDi = np.zeros((len(nodesC),len(c)),dtype=float)
    
    #k = 10
    #pi = 0.1
    #po = 0.02
    
    #for idx1 in range(len(nodesC)):
        #for idx2 in range(len(c)):
        
            #print "Nodes per cluster : ", nodesC[idx1], " # Clusters = ", c[idx2]
            #G = nf.createAGraphWithERCommunities(c[idx2],nodesC[idx1],pi,po)
            #print "Number of nodes / edges = ", nx.number_of_nodes(G), " " , nx.number_of_edges(G)
              
            #start = time()
            #ipNodesDi, nFDi, tHSDG = twoStageGreeDi(G,k)
            #tGreeDi[idx1][idx2] = float(time()-start)
            #print tGreeDi[idx1][idx2], " seconds."
            #print
    
    #outfile = "npzGreeDiTimings.npz"
    #np.savez(outfile, nodesC = nodesC, c = c, k = k, tGreeDi = tGreeDi, pi = pi, po = po)
    #print "Saved all data"  
    
    npz = np.load("./npz_files/npzGreeDiTimings.npz")
    tGreeDi = npz['tGreeDi']
    c = npz['c']
    nodesC = npz['nodesC']
    
    npz = np.load("./npz_files/npzScalabilityNoWSM-WS.npz")
    numNodes = npz['numNodes']
    tOG = npz['tOG']

    #linOG = linregress(np.log(numNodes), np.log(tOG))
    popt, pcov = curve_fit(func, numNodes, tOG)
    tOGls = func(numNodes, *popt)
    spdups = np.zeros((len(nodesC),len(c)),dtype=float)
    totalNodes = np.zeros((len(nodesC),len(c)))
    
    for idx1 in range(len(nodesC)):
        for idx2 in range(len(c)):
            nNodes = nodesC[idx1]*c[idx2]
            spdups[idx1,idx2] = func(nNodes, *popt)/tGreeDi[idx1][idx2]
            totalNodes[idx1,idx2] = nNodes
    
    #print nodesC
    #print c
    #print totalNodes
    #print tGreeDi
    print spdups
    print
    print
    

    for idx in range(len(c)):
        
        popt1, pcov1 = curve_fit(func, nodesC, spdups[:,idx])
        print popt1
    
    print
    print
    
    for idx in range(len(nodesC)):
        
        popt1, pcov1 = curve_fit(func, c, spdups[idx,:])
        print popt1
        
    
    fig = plt.figure()
    plt.subplot(111)
    clr = ['r','g','b']
    mrk =["o","s","D"]
    for idx in range(len(c)):
        labelStr = "# Clusters = " + str(c[idx])
        plt.plot(nodesC,spdups[:,idx],color=clr[idx],marker = mrk[idx], markersize = 12, linewidth=5,linestyle = 'dashed',label = labelStr)
        
    plt.xlabel("Number of nodes per cluster",fontsize=40)
    plt.ylabel("Speedup Ordinary Greedy",fontsize=40)
    plt.grid()    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc="lower right",fontsize=32)
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.75, top=0.75)    
    plt.xlim([200,800])
        
    ax = plt.gca()
    ax.set_xticks(nodesC)
    ax.set_xticklabels(nodesC)    
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)    
    
    plt.show()