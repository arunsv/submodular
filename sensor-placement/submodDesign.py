import numpy as np
import meshpy.triangle as triangle
import matplotlib.pyplot as plt
import numpy.linalg as la
import sys
import time
from sklearn.cluster import KMeans
import scipy.stats as ss
from matplotlib.ticker import ScalarFormatter
import networkx as nx


###############################################################################################################
###############################################################################################################
####################################### SENSOR PLACEMENT  #####################################################
###############################################################################################################
###############################################################################################################


def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]
    
def defineGeometry(gScale=1.0):
    
    points = [(gScale*0, gScale*0), (gScale*2, gScale*0), (gScale*2, gScale*4), (gScale*1,gScale*4), (gScale*1, gScale*2), (gScale*0, gScale*2)]
    #points = [(gScale*0, gScale*0), (gScale*4, gScale*0), (gScale*4, gScale*4), (gScale*0,gScale*4)]
    facets = round_trip_connect(0,len(points)-1)    

    return (points,facets)

def buildMesh(pt,fc,mS):
    
    mesh_info = triangle.MeshInfo()
    
    mesh_info.set_points(pt)
    mesh_info.set_facets(fc)   

    mesh = triangle.build(mesh_info, volume_constraints=True, max_volume=mS)
    
    return mesh

def visualizeMesh(mesh,centroids,locations):
    
    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    x = np.reshape(centroids[:,0],len(mesh.elements))
    y = np.reshape(centroids[:,1],len(mesh.elements))
    
    plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris, lw=1.0,color = [0.3,0.8,0.1])
    
    plt.scatter(x[locations],y[locations],c='k',s=40)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)  

    plt.show()    
 
def getCentroids(mesh):
    
    centroids = np.zeros((len(mesh.elements),2))
    pts = mesh.points
    
    for i, t in enumerate(mesh.elements) :
        centroids[i,0] = (pts[t[0],0] + pts[t[1],0] + pts[t[2],0])/3.0
        centroids[i,1] = (pts[t[0],1] + pts[t[1],1] + pts[t[2],1])/3.0
        
    return centroids
 
def  getCovarianceMatrix(cen,n,err,l):
    
    start = time.time()
    
    cov = np.zeros((n,n))
    
    for idx in range(0,n):
        for jdx in range(idx,n):
            
            cov[idx,jdx] = getCovariancePoints(cen[idx,:],cen[jdx,:],l)
            if (idx != jdx):
                cov[jdx,idx] = cov[idx,jdx]
            else:
                cov[idx,idx] = err + cov[idx,idx]
    
    timeTaken = time.time() - start
    
    return (cov,timeTaken)

def dumpCovarianceMatrix(cov,cvFile,nLoc,tT):
    
    start = time.time()
    np.savez(cvFile,cov=cov,nL=nLoc,timeTaken = tT)
    timeTaken = time.time() - start
    
    return timeTaken

def readCovarianceMatrix(cvFile):
    
    npzFile = np.load(cvFile)
    cov = npzFile['cov']
    nL = npzFile['nL']
    tT = npzFile['timeTaken']
    
    return (nL,cov,timeTaken)

def getCovariancePoints(pt1,pt2,l):
    
    r = np.linalg.norm((pt1-pt2),2)
    #print pt1," ",pt2," ",r
    
    #Squared exponential
    covPts = np.exp(-((r*r)/(2*l*l)))
    
    return covPts

def ordinaryGreedy(n,k,cv):
    
    V = list(np.linspace(0,n-1,num=n, endpoint=True, dtype=int))
    A = []
    objF = 0.0
    objFVec = np.zeros((k+1)) #First sensor -> 0.0
    AComp = []
    
    print "Running the ordinary greedy algorithm."
    start = time.time()

    for idx in range(k):
        
        candidate = -1
        #print "Selecting sensor number : ", idx+1
        AComp = list((set(V)).difference(set(A)))
        bestObjFGain = 0.0

        for y in AComp:
            
            Abar = list((set(AComp)).difference([y]))
            
            marginalObjFGain = computeMarginalMIgain([y],A,Abar,cv)
                        
            if (marginalObjFGain > bestObjFGain):
                candidate = y
                bestObjFGain = marginalObjFGain
                
        
        objF = objF + bestObjFGain
        objFVec[idx+1] = objF
    
        A.append(candidate)
    
    timeTaken = time.time()-start
    
    return (A,objFVec,timeTaken)  

def ordinaryGreedyWithLazyEvals(n,k,cv):
    
    V = list(np.linspace(0,n-1,num=n, endpoint=True, dtype=int))
    A = []
    objF = 0.0
    objFVec = np.zeros((k+1)) #First sensor -> 0.0
    AComp = []
    
    print "Running the ordinary greedy algorithm with lazy evaluations."
    print "Initializing the priority queues."
    
    Ele2Delta = {}
    IsEleCurr = {}
    
    for ele in V:
        Ele2Delta[ele] = 1e16
        IsEleCurr[ele] = False
    
    start = time.time()
    
    for idx in range(k):
        
        
        AComp = list((set(V)).difference(set(A)))
        for ele in AComp:
            
            IsEleCurr[ele] = False
        
        while (True):
        
            y = max(Ele2Delta,key = lambda key:Ele2Delta[key])

            if (IsEleCurr[y] == True):
                
                A.append(y)
                objF = objF + Ele2Delta[y]
                objFVec[idx+1] = objF 
                
                Ele2Delta.pop(y)
                IsEleCurr.pop(y)
                
                #print "Found sensor number : ", idx
    
                break
            
            else:
            
                Abar = list((set(AComp)).difference([y]))
                Ele2Delta[y] = computeMarginalMIgain([y],A,Abar,cv)
                IsEleCurr[y] = True

    timeTaken = time.time()-start
    
    return (A,objFVec,timeTaken)


def stochasticGreedy(n,k,cv,eps,IsTrueObjNeeded):
    
    V = list(np.linspace(0,n-1,num=n, endpoint=True, dtype=int))
    A = []
   
    objFVec = np.zeros((k+1)) #First sensor -> 0.0
    objFVecTrue = np.zeros((k+1))
    
    AComp = []
    
    print "Running the stochastic greedy algorithm."
   
    
    timeTaken = 0.0

    for idx in range(k):
        
        candidate = -1
        #print "Selecting sensor number : ", idx+1
        
        start = time.time()
        
        ACompFull = list((set(V)).difference(set(A)))
        AComp = generateSampledSubset(ACompFull,n,k,eps)
        
        bestObjFGain = 0.0
        marginalObjFGain = 0.0
        #diffMarginalGain = []

        for y in AComp:
            
            Abar = list((set(AComp)).difference([y]))
            marginalObjFGain = computeMarginalMIgain([y],A,Abar,cv)
            
            #marginalObjFGainTrue = computeMarginalMIgain([y],A,list((set(ACompFull)).difference([y])),cv)
            #diffMarginalGain.append((marginalObjFGainTrue-marginalObjFGain))
                        
            if (marginalObjFGain > bestObjFGain):
                candidate = y
                bestObjFGain = marginalObjFGain

        #print "Mean marginal gain difference for this iteration : ", np.mean(diffMarginalGain)
        objFVec[idx+1] = objFVec[idx]+bestObjFGain
        
        timeTaken = timeTaken + (time.time()-start)
        
        if (IsTrueObjNeeded):
            
            AbarFull = list((set(ACompFull)).difference([candidate]))
            trueMarginalGain = computeMarginalMIgain([candidate],A,AbarFull,cv)
            objFVecTrue[idx+1] = objFVecTrue[idx] + trueMarginalGain
            
        A.append(candidate)
    
    return (A,objFVec,objFVecTrue,timeTaken)


def generateSampledSubset(A,n,k,eps):
    
    nRand = int(np.ceil((n*np.log(1/eps))/float(k)))
    Asamp = list(np.random.choice(A,nRand,False))
    
    return Asamp

    
def computeMarginalMIgain(y,A,Abar,cv):
    
    #Get the two sub-matrices of the covariance matrix
    ixgrid = np.ix_(A,A)
    cA = cv[ixgrid]
    
    ixgrid = np.ix_(Abar,Abar)
    cAbar = cv[ixgrid]
    
    #Get the four vectors (portions of row[y] and column[y])
    ixgrid = np.ix_(y,A)
    cyA = cv[ixgrid]
    
    ixgrid = np.ix_(A,y)
    cAy = cv[ixgrid]
    
    ixgrid = np.ix_(y,Abar)
    cyAbar = cv[ixgrid]
    
    ixgrid = np.ix_(Abar,y)
    cAbary = cv[ixgrid] 
    
    #Get the Cov(y,y) element
    ixgrid = np.ix_(y,y)
    cyy = cv[ixgrid]
    
    num = (cyy*cyy) - np.matmul(cyA,np.matmul((np.linalg.inv(cA)),cAy))
    den = (cyy*cyy) - np.matmul(cyAbar,np.linalg.solve(cAbar,cAbary))
    MIGain = np.log2(num/den)
    MIGain = np.reshape(MIGain,(1,1))
    
    if (den == 0):
        print "Denominator became zero while calculating the MI gain"
        sys.exit(0)
    else:
        return MIGain[0]

def partitionTheSpace(points,nP):
    
    print "Partitioning the space described by the given points"
    kmeans = KMeans(n_clusters=nP, random_state=0).fit(points)
    labels = kmeans.labels_
    
    allParts = []
    for idx in range(nP):
        allParts.append([])
        
    count = 0
    
    for point in points:
        (allParts[labels[count]]).append(point)
        count = count+1
        
    return allParts      
    
def plotMI(fG,fSG,gFlag):
    
    npzfileG = np.load(fG)
    npzfileSG = np.load(fSG)
    sGSel = [0,1,2,3,4]
    c2 = ['r','b','g','brown','gold']
    
    if (gFlag == 1):
        k = npzfileG['k']
        k = k[-1:]
        mG = npzfileG['mG']
    else :
        k = npzfileSG['kS']
        print k
    
    mSG = npzfileSG['MISG']
    epsilon = npzfileSG['epsilon']
    sampSize = npzfileSG['sampSize']
    
    print npzfileSG['TSG']
    
    nPts = mSG.shape[0]
    
    
    #Plotting
    plt.figure()
    kVec= np.linspace(0,k,k+1,endpoint=True,dtype=int)

    if (gFlag == 1):
        plt.plot(kVec,mG, color=[0,0,0],linewidth = 6,label = "Ordinary Greedy Algorithm")
    
    for idx in range(len(sGSel)):
        
        #c2 = [np.random.random(),np.random.random(),np.random.random()]
        #Plot all the stochastic greedy stuff
        plt.plot(kVec,mSG[sGSel[idx],:], color=c2[sGSel[idx]],linestyle='--',linewidth = 6,label = ("SG; eps = "+str(epsilon[sGSel[idx]])))
    
    ax=plt.gca()
    ax.xaxis.set_tick_params(labelsize=42)
    ax.yaxis.set_tick_params(labelsize=42)  
    #ax.xaxis.set_major_formatter(ScalarFormatter())
    #ax.set_xticks([50,100,200,500,1000,2000])
    #ax.yaxis.set_major_formatter(ScalarFormatter()) 
    #plt.xlim([50,2000])
    plt.xlabel("Sensor Index",fontsize = 44)
    plt.ylabel("Mutual Information Metric in bits",fontsize = 44)
    plt.legend(fontsize = 40, loc="lower right")
    plt.grid()
    plt.show()
    
    
###############################################################################################################
###############################################################################################################
####################################### CONTROLLABILITY AND PERFORMANCE #######################################
###############################################################################################################
###############################################################################################################
    
def createAnERGraph(nNodes, kavg):
    
    pEdge = kavg/float(nNodes)
    G = nx.erdos_renyi_graph(nNodes, pEdge)
    
    return G

def createABAGraph(nNodes, kavg):
    
    G = nx.barabasi_albert_graph(nNodes,kavg)
    
    return G

def findUnMatchedNodesMaximalMatching(G):
    
    m_x = nx.maximal_matching(G)
    G1 = nx.from_edgelist(list(m_x))
    m_nodes = set(nx.nodes(G1))
    all_nodes = set(nx.nodes(G))
    um_nodes = all_nodes.difference(m_nodes)
    
    print "Number of unmatched nodes = ",len(um_nodes)
    
    return um_nodes

def findGCI(G,S):

    nNodes = nx.number_of_nodes(G)
    Gs = nx.Graph(G)

    for node in S:
        Gs.remove_node(node)
    
    m_x = nx.max_weight_matching(Gs)
    print m_x
    GTemp = nx.from_edgelist(list(m_x))
    gci = nNodes - nx.number_of_nodes(GTemp)
    
    return gci


def runGreedyGCI(G,k):
    
    gci = np.zeros((k+1))
    nNodes = nx.number_of_nodes(G)
    S= []
    
    for idx in range(k):
        
        print "Node number : ", (idx+1)
        
        V = list(set(nx.nodes(G)).difference(set(S)))
        #print "Size of V :",len(V)
        maxGain = 0
        selNode = -1
        
        for u in V:
            Su = list(S)
            Su.append(u)
            #print Su
            
            gain = findGCI(G,Su)-gci[idx]
            if (gain >= maxGain):
                selNode = u
                maxGain = gain
                
        S.append(selNode)
        gci[idx+1] = gci[idx]+maxGain
        
    return (S,gci)
        
        