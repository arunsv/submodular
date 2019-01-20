import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import scipy.sparse as sp
from time import time
from operator import itemgetter


def createAnERGraph(nNodes, pE):
    
    G = nx.erdos_renyi_graph(nNodes, pE)
    
    return G

def createABAGraph(nNodes, kavg):
    
    G = nx.barabasi_albert_graph(nNodes,kavg)
    
    return G

def createAGraphWithERCommunities(nNodes,nC, pin, pout):

    G = nx.planted_partition_graph(nNodes, nC, pin, pout)
    
    return G
    

def computeNF(lap,sel):
    
    grLap = lap[sel,:][:,sel]
    
    start = time()
    nf = np.trace(np.linalg.inv(grLap))
    
    #print "Time taken direct inverse Lg : ", time()-start
    
    return (-1.00*nf)


def computeNFScalable(lapginv,idx,sel,lapg):
    
    nf = 0.0
    n = lapginv.shape[0]
    LgInvMat = None
    
    if (idx == 0):
        
        #Compute Lginv for the (n-1)x(n-1) matrix via RD
        #start = time()
        LgInvMat = computeLgInvRD(lapginv,sel)
        #print "Time taken n-1 inverse RD : ", time()-start
        
    else:
        
        #Update the inverse via the WSM formula
        start = time()
        LgInvMat = computeWSMinverse(lapginv,sel,lapg)
        #print "Time taken WSM inverse : ", time()-start
        
    nf = np.matrix.trace(LgInvMat)
    return (-1.00*nf, LgInvMat)

def computeWSMinverse(D,u,C):
    

    n = D.shape[0]
    rIndex = range(u)+range(u+1,n)
    
    #Rearrange if u is not the last row/col
    if (u != n):
        
        DInvApp = reArrangeRowCol(D,n,u)
    
    else:
        
        DInvApp = D.copy()
        
    #Find U and V
    U = np.zeros((n,2))
    V = np.zeros((n,2))
    
    U[:n-1,0] = -C[u,rIndex]
    U[n-1,0] = -(C[u,u]-1.0)/2.0
    U[n-1,1] = -1.0
    
    V[n-1,0] = 1.0
    V[:n-1,1] = np.reshape(C[rIndex,u],((n-1),))
    V[n-1,1] = (C[u,u]-1.0)/2.0
    V = np.transpose(V)
    
    #WSM update
    I = np.eye(2)
    T1 = np.matmul(DInvApp,U)
    T2 = np.matmul(V,T1)
    T3 = I + T2
    T4 = np.linalg.inv(T3)
    T5 = np.matmul(V,DInvApp)
    T6 = np.matmul(T4,T5)
    T7 = np.matmul(T1,T6)
    
    D2AppInv = DInvApp - T7
    D2InvWSM = D2AppInv[:n-1,:n-1]    
    
    return D2InvWSM

def reArrangeRowCol(A,N,k):
    
    C = np.zeros((N,N))
    sInd = range(k)+range(k+1,N)

    mask=np.full((N,N),True, dtype = bool)
    mask[k,:] = False
    mask[:,k] = False
    A2 = np.reshape(A[mask],(N-1,N-1)) 

    C[:(N-1),:(N-1)] = A2
    C[(N-1),:(N-1)] = A[k,sInd]
    C[:(N-1),(N-1)] = np.reshape(A[sInd,k],(N-1))
    
    C[(N-1),(N-1)] = A[k,k]
    
    
    return C

def computeLPinvRD(lap):

    J = (1./lap.shape[0])*np.ones(lap.shape)
    B = lap + J
    C = np.linalg.inv(B)
    D = C - J   
    
    return D
    
def computeLgInvRD(D,u):
    
    #Computing the inverse of (n-1) x (n-1)
    #Random column / row removal
       
    n = D.shape[0]
    LgInvRD = np.zeros((n-1,n-1))
    
    if (u == (n-1)):
        rIndex = range(u)
    else:
        rIndex = range(u)+range(u+1,n)

    mask=np.full((n,n),True,dtype=bool)
    mask[u,:] = False
    mask[:,u] = False
    A = np.reshape(D[mask],(n-1,n-1))  

    B = np.tile(np.reshape(D[rIndex,u],(n-1,1)),(1,n-1))
    C = np.tile(D[u,rIndex],(n-1,1))
    d = D[u,u]
    
    LgInvRD = A - B - C + d
      
    return LgInvRD

def generateSampledSubset(A,n,k,eps):
    
    nRand = int(np.ceil((n*np.log(1/eps))/float(k)))
    Asamp = list(np.random.choice(A,nRand,False))
    
    
    return Asamp


def runGreedyNF(G,k):
    
    nf = np.zeros((k+1))
    nf[0] = -10000
    
    nNodes = nx.number_of_nodes(G)
    Lsp = nx.laplacian_matrix(G)
    L = sp.csr_matrix.todense(Lsp)    
    selNode = -1
    S = []
    
    for idx in range(k):
        
        print "Node number : ", (idx+1)
        
        V = list(set(nx.nodes(G)).difference(set(S)))
        maxGain = -1e16
        
        for u in V:
        
            Vu = list(set(V).difference(set([u])))
            
            gain = computeNF(L,Vu)-nf[idx]
            
            if (gain >= maxGain):
                selNode = u
                maxGain = gain
                
        S.append(selNode)
        nf[idx+1] = nf[idx]+maxGain
        
    
    nf = -1.00*nf
    
    return (S,nf) 



def runGreedyNFScalable(G,k):
    
    nf = np.zeros((k+1))
    nNodes = nx.number_of_nodes(G)
    
    #All the pre-processing
    
    #Laplacian Pseudo-inverse
    Lsp = nx.laplacian_matrix(G)
    L = sp.csr_matrix.todense(Lsp)    
    Lpinv = computeLPinvRD(L)
    Lginv = Lpinv
    
    selNode = -1
    S = []
    
    for idx in range(k):
        
        print "Node number : ", (idx+1)
        
        V = list(set(nx.nodes(G)).difference(set(S)))
       
        #Grounded Laplacian computation
        Lg = L[V,:][:,V]
    
        maxGain = -1e16
        
        for v in V:
           
            nFigure,LginvInter = computeNFScalable(Lginv,idx,V.index(v),Lg)
            gain = nFigure-nf[idx]
            
            if (gain >= maxGain):
                selNode = v
                maxGain = gain
                LginvMaxGain = LginvInter
        
        S.append(selNode)
        nf[idx+1] = nf[idx]+maxGain
        Lginv = LginvMaxGain
        
    nf = -1.00*nf
    
    return (S,nf) 


def runLazyGreedyNF(G,k):
    
    nf = np.zeros((k+1))
    nf[0] = -10000
    nNodes = nx.number_of_nodes(G)
    
    #All the pre-processing
    
    #Laplacian Pseudo-inverse
    Lsp = nx.laplacian_matrix(G)
    L = sp.csr_matrix.todense(Lsp)    

    selNode = -1
    S = []
    
    print "Running the ordinary greedy algorithm with lazy evaluations."
    print "Initializing the priority queues."
    
    Ele2Delta = {}
    IsEleCurr = {}
    
    for ele in list(nx.nodes(G)):
        Ele2Delta[ele] = 1e16
        IsEleCurr[ele] = False
    
   
    for idx in range(k):
        
        
        #print
        print "Node number : ", (idx+1)
        #print dict(sorted(Ele2Delta.iteritems(), key=itemgetter(1), reverse=True)[:3])
        
        V = list(set(nx.nodes(G)).difference(set(S)))
        
        maxGain = -1e16
        recompCount = 0
        
        for ele in V:
            
            IsEleCurr[ele] = False
        
        while (True):
        
            y = max(Ele2Delta,key = lambda key:Ele2Delta[key])
            #print "Max element : ", y, " and IsEleCurr[y] is ", IsEleCurr[y] 

            if (IsEleCurr[y] == True):
                
                S.append(y)
                nf[idx+1] = nf[idx] + Ele2Delta[y]
                
                Ele2Delta.pop(y)
                IsEleCurr.pop(y)
                
                print "Number of recomputations = ", recompCount
                print
                
                break
            
            else:
            
                Vbar = list((set(V)).difference([y]))
                Ele2Delta[y] = computeNF(L,Vbar)-nf[idx]
                IsEleCurr[y] = True
                recompCount += 1
                #print "Recomputing for : ", y, "new gain = ",Ele2Delta[y]

      
    return (S,-1.00*nf)

def runLazyGreedyNFScalable(G,k):
    
    nf = np.zeros((k+1))
    nf[0] = -10000
    nNodes = nx.number_of_nodes(G)
    
    #All the pre-processing
    
    #Laplacian Pseudo-inverse
    Lsp = nx.laplacian_matrix(G)
    L = sp.csr_matrix.todense(Lsp) 
    Lpinv = computeLPinvRD(L)
    Lginv = Lpinv    

    selNode = -1
    S = []
    
    print "Running the ordinary greedy algorithm with lazy evaluations."
    print "Initializing the priority queues."
    
    Ele2Delta = {}
    IsEleCurr = {}
    #Ele2LgInv = {}
    
    for ele in list(nx.nodes(G)):
        Ele2Delta[ele] = 1e16
        IsEleCurr[ele] = False
        #Ele2LgInv[ele] = None
    
   
    for idx in range(k):
        
        
        #print
        print "Node number : ", (idx+1)
        #print dict(sorted(Ele2Delta.iteritems(), key=itemgetter(1), reverse=True)[:3])
        
        V = list(set(nx.nodes(G)).difference(set(S)))
        #Grounded Laplacian computation
        Lg = L[V,:][:,V]         
        maxGain = -1e16
        recompCount = 0
        
        for ele in V:
            
            IsEleCurr[ele] = False
            #Ele2LgInv[ele] = None
        
        while (True):
        
            y = max(Ele2Delta,key = lambda key:Ele2Delta[key])
            #print "Max element : ", y, " and IsEleCurr[y] is ", IsEleCurr[y] 

            if (IsEleCurr[y] == True):
                
                S.append(y)
                nf[idx+1] = nf[idx] + Ele2Delta[y]
                nFigure,Lginv = computeNFScalable(Lginv,idx,V.index(y),Lg)
                
                Ele2Delta.pop(y)
                IsEleCurr.pop(y)
                #Ele2LgInv.pop(y)
                
                print "Number of recomputations = ", recompCount
                print
                
                break
            
            else:
            
                nFigure,LgInter = computeNFScalable(Lginv,idx,V.index(y),Lg)
                Ele2Delta[y] = nFigure-nf[idx]
                IsEleCurr[y] = True
                recompCount += 1
                #print "Recomputing for : ", y, "new gain = ",Ele2Delta[y]

      
    return (S,-1.00*nf)

def runStochGreedyNF(G,k,eps):
    
    nf = np.zeros((k+1))
    nNodes = nx.number_of_nodes(G)
    Lsp = nx.laplacian_matrix(G)
    L = sp.csr_matrix.todense(Lsp)    
    selNode = -1
    S = []
    
    for idx in range(k):
        
        print "Node number : ", (idx+1)
        
        V = list(set(nx.nodes(G)).difference(set(S)))
        maxGain = -1e16
        Vsamp = generateSampledSubset(V, nNodes, k, eps)
        
        for u in Vsamp:
        
            Vu = list(set(V).difference(set([u])))
            
            gain = computeNF(L,Vu)-nf[idx]
            
            if (gain >= maxGain):
                selNode = u
                maxGain = gain
                
        S.append(selNode)
        nf[idx+1] = nf[idx]+maxGain
        
    
    nf = -1.00*nf
    
    return (S,nf)


def runStochGreedyNFScalable(G,k,eps):
    
    nf = np.zeros((k+1))
    nNodes = nx.number_of_nodes(G)
    
    #All the pre-processing
    
    #Laplacian Pseudo-inverse
    Lsp = nx.laplacian_matrix(G)
    L = sp.csr_matrix.todense(Lsp)    
    Lpinv = computeLPinvRD(L)
    Lginv = Lpinv
    
    selNode = -1
    S = []
    
    for idx in range(k):
        
        print "Node number : ", (idx+1)
        
        V = list(set(nx.nodes(G)).difference(set(S)))
        Vsamp = generateSampledSubset(V, nNodes, k, eps)
       
        #Grounded Laplacian computation
        Lg = L[V,:][:,V]
    
        maxGain = -1e16
        
        for v in Vsamp:
           
            nFigure,LginvInter = computeNFScalable(Lginv,idx,V.index(v),Lg)
            gain = nFigure-nf[idx]
            
            if (gain >= maxGain):
                selNode = v
                maxGain = gain
                LginvMaxGain = LginvInter
        
        S.append(selNode)
        nf[idx+1] = nf[idx]+maxGain
        Lginv = LginvMaxGain
        
    nf = -1.00*nf
    
    return (S,nf)


def runLazyGreedyNFScalableSubset(G,Vs,k):
    
    nf = np.zeros((k+1))
    nf[0] = -10000
    nNodes = nx.number_of_nodes(G)
    
    #All the pre-processing
    
    #Laplacian Pseudo-inverse
    Lsp = nx.laplacian_matrix(G)
    L = sp.csr_matrix.todense(Lsp) 
    Lpinv = computeLPinvRD(L)
    Lginv = Lpinv    

    selNode = -1
    S = []
    
    print "Running the ordinary greedy algorithm with lazy evaluations."
    print "Initializing the priority queues."
    
    Ele2Delta = {}
    IsEleCurr = {}
    
    for ele in list(Vs):
        Ele2Delta[ele] = 1e16
        IsEleCurr[ele] = False
   
   
    for idx in range(k):

        print "Node number : ", (idx+1)
        
        V = list(set(nx.nodes(G)).difference(set(S)))
        #Grounded Laplacian computation
        Lg = L[V,:][:,V]         
        maxGain = -1e16
        recompCount = 0
        
        for ele in Vs:
            
            IsEleCurr[ele] = False
        
        while (True):
        
            y = max(Ele2Delta,key = lambda key:Ele2Delta[key])

            if (IsEleCurr[y] == True):
                
                S.append(y)
                nf[idx+1] = nf[idx] + Ele2Delta[y]
                nFigure,Lginv = computeNFScalable(Lginv,idx,V.index(y),Lg)
                
                Ele2Delta.pop(y)
                IsEleCurr.pop(y)
                
                print "Number of recomputations = ", recompCount
                print
                
                break
            
            else:
            
                nFigure,LgInter = computeNFScalable(Lginv,idx,V.index(y),Lg)
                Ele2Delta[y] = nFigure-nf[idx]
                IsEleCurr[y] = True
                recompCount += 1

      
    return (S,-1.00*nf)

def runStochGreedyNFScalableSubset(G,Vs,k,eps):
    
    nf = np.zeros((k+1))
    nNodes = nx.number_of_nodes(G)
    
    #All the pre-processing
    
    #Laplacian Pseudo-inverse
    Lsp = nx.laplacian_matrix(G)
    L = sp.csr_matrix.todense(Lsp)    
    Lpinv = computeLPinvRD(L)
    Lginv = Lpinv
    
    selNode = -1
    S = []
    
    for idx in range(k):
        
        print "Node number : ", (idx+1)
        
        V = list(set(nx.nodes(G)).difference(set(S)))
        Vs_up = list(set(Vs).difference(set(S)))
        Vsamp = generateSampledSubset(Vs_up, len(Vs_up), k, eps)
       
        #Grounded Laplacian computation
        Lg = L[V,:][:,V]
    
        maxGain = -1e16
        
        for v in Vsamp:
           
            nFigure,LginvInter = computeNFScalable(Lginv,idx,V.index(v),Lg)
            gain = nFigure-nf[idx]
            
            if (gain >= maxGain):
                selNode = v
                maxGain = gain
                LginvMaxGain = LginvInter
        
        S.append(selNode)
        nf[idx+1] = nf[idx]+maxGain
        Lginv = LginvMaxGain
        
    nf = -1.00*nf
    
    return (S,nf)