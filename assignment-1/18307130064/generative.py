import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def getdata ():
    fle = open ( "dataset.data" , "r" )
    lines = int(fle.readline())
    #print ( lines )
    retx = np.empty ( (lines,2) )
    rety = np.empty ( (lines,1) )
    for i in range ( lines ):
        (retx[i][0],retx[i][1],rety[i]) = fle.readline().split()
    rety=rety.astype ( int )
    return (lines,retx,rety)

def cal ( n , x , y ):
    pi = np.zeros ( (3) )
    mu = np.zeros ( (3,2) )
    sigma = np.zeros ( (2,2) )
    for i in range ( n ):
        pi[y[i]-1] += 1
        mu[y[i]-1] += x[i]
    for i in range ( 3 ):
        mu[i] /= pi[i]
        pi[i] /= n
    for i in range ( n ):
        sigma += np.dot ( (x[i]-mu[y[i]-1]).transpose() , (x[i]-mu[y[i]-1]) )
    sigma /= n
    return (pi,mu,sigma)

def predict ( nowx , pi , mu , sigma ):
    prob = np.zeros ( 3 )
    sum = 0.0
    for i in range ( 3 ):
        prob[i] = stats.multivariate_normal ( mu[i] , sigma ).pdf ( nowx ) * pi[i]
        sum += prob[i]
    prob /= sum
    maxx = -1.0
    maxi = 0
    for i in range ( 3 ):
        if maxx < prob[i]:
            maxx=prob[i]
            maxi = i
    return maxi + 1


def test ( n , x , y , pi , mu , sigma ):
    prd = np.zeros ( n )
    acc = 0
    for i in range ( n ):
        prd[i] = predict ( x[i] , pi , mu , sigma )
        if prd[i] == y[i]: acc += 1

    print ( "Accuracy:" , acc / n , "\n" )

    out = x.transpose ()
    col = ['ro','bo','go']
    wrong = ['rx','bx','gx']
    plt.figure(figsize=(10,10))
    for i in range ( n ):
        if y[i][0] == int(prd[i]):
            plt.plot ( out[0][i] , out[1][i] , col[y[i][0]-1] )
        else:
            plt.plot ( out[0][i] , out[1][i] , wrong[y[i][0]-1] )
    plt.show ()

    print ( "Wrong cases:" )
    for i in range ( n ):
        if y[i][0] != int(prd[i]):
            print ( x[i] , "real:" , y[i][0] , "predict:" , int(prd[i]) )

    return


(n,x,y) = getdata ()

(pi,mu,sigma) = cal ( n , x , y )

#for i in range ( 3 ):
#    print ( pi[i] , mu[i] )
#print ( sigma )

test ( n , x , y , pi , mu , sigma )

