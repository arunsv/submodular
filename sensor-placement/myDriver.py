import sensDrivers as sensd
import controlDrivers as cntrld
import time
import math
import numpy as np
import matplotlib.pyplot as plt

print
sensd.runOrdGreedy(0.01,10)
print

#print
#sensd.runOrdGreedyWithLazy(0.01,10)
#print

#print
#sensd.runStochasticGreedy(0.1,10)
#print

#for idx in range(6):
    
    #sensd.driverGenerateALargeCovarianceMartrix(0.093,"covSqExp",gS=math.pow((math.sqrt(math.sqrt(7.07))),idx))
    #sensd.driverGenerateALargeCovarianceMartrix(0.093,"covSqExp",gS=math.pow((math.sqrt(math.sqrt(10.0))),idx))
    
#y1 = sensd.driverPart(0.01,12,4)    
#print

#y2 = sensd.driverPart(0.01,12,4,2)    
#print

#y3 = sensd.runOrdGreedyWithLazy(0.01,12)
#print

#x = np.linspace(0,12,13)
#plt.figure()
#plt.plot(x,y1,'r',linewidth=5,label="Partitioned; $k_p=k$")
#plt.plot(x,y2,'b',linewidth=5,label="Partitioned; $k_p=2k$")
#plt.plot(x,y3,'g',linewidth=5,label="Un-Partitioned")
#plt.xlabel("Sensor number",fontsize=44)
#ax = plt.gca()
#ax.xaxis.set_tick_params(labelsize=40)
#ax.yaxis.set_tick_params(labelsize=40)
#plt.ylabel("Mutual Information in bits",fontsize=44)
#plt.grid()
#plt.legend(loc="lower right",fontsize=40)
#plt.show()

#cntrld.driverControllabilityNoiseMatroid()
#cntrld.driverControllabilityGCI()