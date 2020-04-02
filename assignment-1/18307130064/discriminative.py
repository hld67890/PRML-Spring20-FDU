import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random

def sigmoid ( x ):
    return 1.0/(1+np.exp(-x))

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

def predict_sgd ( j , nowx , w , w0 ):
    #print ( nowx , w[j] , np.dot ( nowx , w[j] ) )
    return sigmoid (np.dot ( nowx , w[j] ) + w0[j])

def sgd ( n , x , y , batch , epoch , alpha ):
    w = np.zeros ( (3,2) )
    w0 = np.zeros ( (3,1) )
    prd = np.zeros ( (3,n) ) 
    dec = alpha / epoch

    zp = list(zip(x,y))
    x,y = zip (*zp)

    now = 0
    for ep in range ( epoch ):
        random.shuffle ( zp )
        for i in range ( batch ):
            now += 1
            if now == n:
                now = 0
                break
            dw = np.zeros ( (3,2) )
            dw0 = np.zeros ( (3,1) )
            for j  in range ( 3 ):
                prd[j][now] = predict_sgd ( j , x[now] , w , w0 )
                dw[j] += ((1 if y[now]==j+1 else 0)-prd[j][now]) * x[now]
                dw0[j] += ((1 if y[now]==j+1 else 0)-prd[j][now])
            #print ( dw )
            #print ( dw0 )
            w += dw/batch * alpha
            w0 += dw0/batch * alpha
        alpha -= dec
    return (w,w0)


def predict ( nowx , w , w0 ):
    prob = np.zeros ( 3 )
    sum = 0.0
    for i in range ( 3 ):
        prob[i] = sigmoid(np.dot ( w[i] , nowx ) + w0[i])
        sum += prob[i]
    prob /= sum
    maxx = -1.0
    maxi = 0
    for i in range ( 3 ):
        if maxx < prob[i]:
            maxx=prob[i]
            maxi = i
    return maxi + 1


def test ( n , x , y , w , w0 ):
    prd = np.zeros ( n )
    acc = 0
    for i in range ( n ):
        prd[i] = predict ( x[i] , w , w0 )
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

epoch = 2000
batch = 10
alpha = 1e0
(w,w0) = sgd ( n , x , y , batch , epoch , alpha )

#print ( w )
#print ( w0 )

test ( n , x , y , w , w0 )

