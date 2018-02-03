
from random import *
import numpy as np
import pandas as pd
import matplotlib as mt
import preprocessing as pr


def y_usable(y):
    y_1 = np.ones((len(y),1))
    th = np.array([0,1,2,3,4,5,6,7,8,9])
    y_2 = y_1*th
    y_3 = np.reshape(y,(len(y),1))
    y_4= (np.equal(y_2,y_3))*1
    return y_4


def hypothesis(x_train,theta,bias):
    a = np.dot(x_train,theta)
    a = np.exp(a+bias)
    b = np.sum(a,axis=1)
    a = a/np.reshape(b,(len(b),1))
    return a



def cross_entropy(x,y,theta,bias):
    h = hypothesis(x,theta,bias)
    e = np.log(h)
    y_1 = y_usable(y)
    return np.sum(np.sum(np.multiply(y_1,e)))



def del_cross_entropy(x,y,theta,bias):
    m = len(y)
    h = hypothesis(x,theta,bias)
    y_1 = y_usable(y)
    a = h-y_1
    x_tr = np.transpose(x)
    d = (np.dot(x_tr,a))/m
    return d



def del_cross_entr_bias(x,y,theta,bias):
    m = len(y)
    h = hypothesis(x,theta,bias)
    y_1 = y_usable(y)
    a = h-y_1
    d = (np.sum(a,axis=0))/m
    return d



training_data_raw = (np.genfromtxt('train.csv',delimiter=','))[1:]



x_tr,y_train = (training_data_raw[0:40000,1:]),training_data_raw[0:40000,0]
classes = 10
x_tr = pr.pixelthresholdfilter(x_tr,147)
x_train = pr.PCA(x_tr,600)

learning_rate = 0.8
theta,bias = np.zeros((np.shape(x_train)[1],classes)),np.zeros((1,classes))


iteration = 1000
print("Initial Cross Entropy")
print(cross_entropy(x_train,y_train,theta,bias))


for i in range(iteration):
    theta = theta -learning_rate*del_cross_entropy(x_train,y_train,theta,bias)
    bias = bias-learning_rate*del_cross_entr_bias(x_train,y_train,theta,bias)
print("Final cross entropy")
print(cross_entropy(x_train,y_train,theta,bias))



l = hypothesis(x_train,theta,bias)



train_prediction = np.argmax(l,axis=1)



e= (np.equal(y_train,train_prediction))*1



print("ACCURACY ON TRAINING sET")
print((np.sum(e))/len(y_train))



x_t,y_t = (training_data_raw[40000:42000,1:]),training_data_raw[40000:42000,0]
x_t = pr.pixelthresholdfilter(x_test,147)
x_test = pr.PCA(x_t,600)
l = hypothesis(x_test,theta,bias)
d = np.argmax(l,axis=1)

e= (np.equal(y_t,d))*1



print("ACCURACY ON Testing sET")
print((np.sum(e))/len(y_t))



