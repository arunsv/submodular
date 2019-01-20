import submodDesign as sd
import numpy as np
import matplotlib.pyplot as plt

def driverGenerateALargeCovarianceMartrix(meshSize,cvFileBase,gS):
    
    sigmaSqSens = 0.1
    lScale = 0.8
                
    pts, fcts = sd.defineGeometry(gScale=gS)
    msh = sd.buildMesh(pts, fcts, meshSize)
    cntrds = sd.getCentroids(msh)
    nL = cntrds.shape[0] 
    print "Candidate number of locations: ", nL 

    cov, timeTaken = sd.getCovarianceMatrix(cntrds,nL,sigmaSqSens,lScale) 
    
    if(nL < 1000):
        suffix = "_"+str(nL)
    else:
        numK = int(round(nL/1000.0))
        suffix = "_"+str(numK)+"k"
    
    cvFile = cvFileBase + suffix
    
    print "It took ", timeTaken, " seconds to generate the covariance matrix"
    print "Writing the covariance matrix to : ", cvFile

    
    timeTakenW = sd.dumpCovarianceMatrix(cov, cvFile, nL, timeTaken)
    print "It took ", timeTakenW, " seconds to write the covariance matrix"
    print 

def runOrdGreedy(meshSize,kSens):
    
    sigmaSqSens = 0.1
    lScale = 0.8
                
    pts, fcts = sd.defineGeometry()
    msh = sd.buildMesh(pts, fcts, meshSize)
    cntrds = sd.getCentroids(msh)
    nL = cntrds.shape[0] 
    print "Candidate number of locations: ", nL 
    cov, tT = sd.getCovarianceMatrix(cntrds,nL,sigmaSqSens,lScale)
    Aset,yVal,timeTaken = sd.ordinaryGreedy(nL,kSens,cov)
    
    print Aset
    print yVal
    
    print
    print "Time taken - ordinary greedy : ", timeTaken, " seconds."
    print    
    
    return yVal

def runOrdGreedyWithLazy(meshSize,kSens):
    
    sigmaSqSens = 0.1
    lScale = 0.8
                
    pts, fcts = sd.defineGeometry()
    msh = sd.buildMesh(pts, fcts, meshSize)
    cntrds = sd.getCentroids(msh)
    nL = cntrds.shape[0] 
    print "Candidate number of locations: ", nL 
    cov, tT = sd.getCovarianceMatrix(cntrds,nL,sigmaSqSens,lScale)
    Aset,yVal,timeTaken = sd.ordinaryGreedyWithLazyEvals(nL,kSens,cov)
    
    print Aset
    print yVal
    
    print
    print "Time taken - lazy greedy : ", timeTaken, " seconds."
    print
    
    return yVal
    
def runStochasticGreedy(meshSize,kSens):
    
    sigmaSqSens = 0.1
    lScale = 0.8
                
    pts, fcts = sd.defineGeometry()
    msh = sd.buildMesh(pts, fcts, meshSize)
    cntrds = sd.getCentroids(msh)
    nL = cntrds.shape[0] 
    
    print "Candidate number of locations: ", nL 
    
    cov, tT = sd.getCovarianceMatrix(cntrds,nL,sigmaSqSens,lScale)
    Aset,yVal,yValTrue,timeTaken = sd.stochasticGreedy(nL, kSens, cov, 0.001,True)    
    
    print Aset
    print yVal
    print yValTrue
    
    print
    print "Time taken - stochastic greedy : ", timeTaken, " seconds."
    print    
    
def driverScalabilityGreedyVsStochGreedy():
    
    sigmaSqSens = 0.1
    lScale = 0.8
    epsilon = [0.001,0.003,0.01,0.03,0.1]
    nPts = len(epsilon)
    meshSize = 0.00243
    runMode = 1
    MISG = np.zeros([])
    
    if (runMode == 0):
        
        TSG = np.zeros((nPts))
        TCov = np.zeros((nPts))
        sampSize = np.zeros((nPts))
        
        pts, fcts = sd.defineGeometry()
        msh = sd.buildMesh(pts, fcts, meshSize)
        cntrds = sd.getCentroids(msh)
        nL = cntrds.shape[0] 
        print "Candidate number of locations: ", nL
        kS = 80
        cov = sd.getCovarianceMatrix(cntrds,nL,sigmaSens,lScale)
        MISG = np.zeros((nPts,kS+1))
        
        print "Starting computation with stochastic greedy for various epsilon values."
        
        for idx in range(nPts):
            
            print "Epsilon value : ", epsilon[idx]
            
            timeTaken = 0.0
            
            start = time.time()
            optSet,mSG,sampSize[idx] = sd.runStochasticGreedyForMaxMI(nL, kS, epsilon[idx], cov, False)
            end = time.time()
            MISG[idx,:] =  mSG
            TSG[idx] = (end-start)
         
            print "Sample size : ", sampSize[idx], "Time taken : ",TSG[idx]
    
        np.savez("test-1-g",nL=nL,kS=kS,TSG=TSG,MISG=MISG,epsilon=epsilon,sampSize=sampSize)
    
    else :
    
        ipFileG = "test-1-g.npz"
        ipFileSG = "test-1-sg.npz"
        sd.plotMI(ipFileG,ipFileSG,1)
        
        
def driverPart(meshSize,kSens,nPart,multFac=1):

    sigmaSqSens = 0.1
    lScale = 0.8
                
    pts, fcts = sd.defineGeometry()
    msh = sd.buildMesh(pts, fcts, meshSize)
    cntrds = sd.getCentroids(msh)
    nL = cntrds.shape[0] 
    
    print "Candidate number of locations: ", nL
    
    allPartitions = sd.partitionTheSpace(cntrds,nPart)
    
    
    #plt.figure()
    count = 0
    col = ['r','b','g','c']
    
    for part in allPartitions:

        xy = np.array(part)
        print "Length of partition ",count, " : ",len(allPartitions[count])
        #plt.scatter(xy[:,0],xy[:,1],color=col[count],marker='o')
        count = count+1
        
    #plt.show()
    
    candSens = np.zeros((nPart,multFac*kSens),dtype=int)
    tParts = 0.0
    
    for idx in range(nPart):
        
        print "Working on partition number : ",idx
        
        cntrdsPart = np.array(allPartitions[idx])     
        nLPart = cntrdsPart.shape[0]
        covP, tTP = sd.getCovarianceMatrix(cntrdsPart,nLPart,sigmaSqSens,lScale)
        AsetP,yValP,tTP = sd.ordinaryGreedyWithLazyEvals(nLPart,multFac*kSens,covP)
        tParts = tParts + tTP         
        candSens[idx,:] = AsetP
    
    print
    print

    print "Second stage ...."
 
    print
    print    

    #Get all candidate points into one array
    allCand = []
    for idx in range(nPart):
        for jdx in range(kSens*multFac):
            elementCoords = (allPartitions[idx])[candSens[idx,jdx]]
            allCand.append(elementCoords)
            
    cntrdsAllParts = np.array(allCand)
    nLAllParts = cntrdsAllParts.shape[0]
    
    covAP, tTAP = sd.getCovarianceMatrix(cntrdsAllParts,nLAllParts,sigmaSqSens,lScale)
    
    AsetAP,yValAP,tTAP = sd.ordinaryGreedyWithLazyEvals(nLAllParts,kSens,covAP)
        
    print AsetAP
    print yValAP
        
    print
    print "Time taken - ordinary greedy for this partition : ", tParts+tTAP, " seconds."
    print                   
    
    return yValAP