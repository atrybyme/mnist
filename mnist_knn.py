
# coding: utf-8
# Author: Shubhansh Awasthi


#import required libraries
from random import *
from matplotlib import pyplot as plt
import numpy as np
import math
from collections import Counter





#define the distatnce function
def distance(x1,x):
    a = x-x1
    a = np.square(a)
    dis = np.sqrt(np.sum(a,axis=1))
    return dis




#add a filter which gives us binary image
def preprocessed(x_train,amount):
    image_number=1
    mtr = x_train>amount
    mtr = mtr*1
    return mtr



#Define the knn prediction function 
def result(dis,y,k):
    a = np.argpartition(dis,k)
    b = []
    for i in range(k):
        b = np.append(b,y[a[i]]) 
    b = b.astype(int)
    d = list(Counter(b).items())
    e = len(d)
    for j in range(e):
        d[j] = list(d[j])
    d = np.array(d)
    e = np.argmax(d[:,1])
    r = d[e,0]
    return r



#import the data from csv file containing 42000 labelled examples
_data_raw = (np.genfromtxt('train.csv',delimiter=','))[1:]
#shufle the data
import sklearn
x_tr,y = sklearn.utils.shuffle((_data_raw[:,1:])/255,_data_raw[:,0])




#applying the filter over the data and splitting the data into training and testing
x_train = preprocessed(x_tr[0:41000],136.0/255.0)
y_train = y[0:41000]
x_test = preprocessed(x_tr[41000:42000],136.0/255.0)
y_test = y[41000:42000]



#choose the value of k in k nearest neighbours
k= 100
prediction = []




print("Runing... ")
for i in range(1000):
    dis_of_xi = distance(x_test[i],x_train)
    p = result(dis_of_xi,y_train,k)
    prediction = np.append(prediction,p)
test_prediction = np.transpose(prediction)
print("Completed")




#test your accuracy
e = (np.equal(test_prediction,y_test))*1
print("ACCURACY ON Testing sET",((np.sum(e))/len(y_test))*100)




#visualize the data 
#a = x_test[1]
#a = np.reshape(a,(28,28))
#get_ipython().run_line_magic('matplotlib', 'inline')
#plt.imshow(a)

# Author : Shubhansh Awasthi
