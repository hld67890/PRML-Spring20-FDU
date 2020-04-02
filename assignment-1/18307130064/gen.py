import numpy as np 
import matplotlib.pyplot as plt
import random

mean1 = (1,2)
cov1 = [[1,0],[0,1]]
mean2 = (5,5)
cov2 = [[1,0],[0,1]]
mean3 = (-2,3)
cov3 = [[1,0],[0,1]]

num1 = 150
num2 = 150
num3 = 150

dist1 = np.random.multivariate_normal(mean1,cov1,(num1))
dist2 = np.random.multivariate_normal(mean2,cov2,(num2))
dist3 = np.random.multivariate_normal(mean3,cov3,(num3))
#dist1=np.random.normal(1,2,30)

#print ( dist1 )
out1 = dist1.transpose()
out2 = dist2.transpose()
out3 = dist3.transpose()
#print ( out1 )

plt.plot(out1[0],out1[1],'o')
plt.plot(out2[0],out2[1],'o')
plt.plot(out3[0],out3[1],'o')


fle = open ("dataset.data" , "w")

output = []
for i in dist1:
    output.append ( (i[0],i[1],1) )
for i in dist2:
    output.append ( (i[0],i[1],2) )
for i in dist3:
    output.append ( (i[0],i[1],3) )

random.shuffle ( output )

#print ( len(output) )
fle.write ( str(num1+num2+num3) + '\n' )
for i in output:
    fle.write ( str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + "\n" )

fle.close ()