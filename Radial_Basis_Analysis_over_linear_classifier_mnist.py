# Author : Shubhansh Awasthi
# coding: utf-8

# In[30]:


#import required libraries
from random import *
import numpy as np
import matplotlib as mt
import preprocessing as pr
#%matplotlib inline


# In[3]:


#one hot encoding
def y_usable(y):
    y_1 = np.ones((len(y),1))
    th = np.array([0,1,2,3,4,5,6,7,8,9])
    y_2 = y_1*th
    y_3 = np.reshape(y,(len(y),1))
    y_4= (np.equal(y_2,y_3))*1
    return y_4

#hypothesis function
def hypothesis(x_train,theta,bias):
    a = np.dot(x_train,theta)
    a = np.exp(a+bias)
    b = np.sum(a,axis=1)
    a = a/np.reshape(b,(len(b),1))
    return a


#Cross Entropy Loss function
def cross_entropy(x,y,theta,bias):
    h = hypothesis(x,theta,bias)
    e = np.log(h)
    y_1 = y_usable(y)
    return np.sum(np.sum(np.multiply(y_1,e)))


#Differential matrix of Loss Function with respect to theta
def del_cross_entropy(x,y,theta,bias):
    m = len(y)
    h = hypothesis(x,theta,bias)
    y_1 = y_usable(y)
    a = h-y_1
    x_tr = np.transpose(x)
    d = (np.dot(x_tr,a))/m
    return d


#Differentiation function of cross entropy loss function with respect to bias
def del_cross_entr_bias(x,y,theta,bias):
    m = len(y)
    h = hypothesis(x,theta,bias)
    y_1 = y_usable(y)
    a = h-y_1
    d = (np.sum(a,axis=0))/m
    return d


# In[4]:


#upload the data to the program
training_data_raw = (np.genfromtxt('train.csv',delimiter=','))[1:]


# In[5]:


#sklearn just to shuffle the data
import sklearn
x_tr,y = sklearn.utils.shuffle((training_data_raw[:,1:])/255.0,training_data_raw[:,0])



#Subtract the mean from the data and normalizing
x = x_tr

x = x.astype(float)
classes = 10
x_train1 = (x[0:41000]-np.mean(x[0:41000],axis=0))/255.00
y_train = y[0:41000]
x_test1 = (x[41000:42000]-np.mean(x[0:41000],axis=0))/255.00
y_test = y[41000:42000]


##applying Radial Basis Kernel method
##step 1. take suitable number of examples from training set as base points
base_points = 1000
##step 2. choose a free variable lambda.
l = 0.005
##randomly choosing the base points from the training data
point_argument = np.random.randint(len(y_train),size = base_points)

min_val = np.zeros(base_points)
max_val = np.zeros(base_points)
x_train = np.zeros((len(y_train),base_points))


##calculating the kernel for each point in training data, this is our new training data
print("Calculating the kernel")
for i in range(base_points):
    a = x_train1-x_train1[point_argument[i]]
    x_train[:,i] = np.exp(-1*(np.square(np.linalg.norm(a,axis=1)))*l)
print("Kernel Calculalted")

## Calculating the kernel for each point in testing data, this is our new testing data
x_test = np.zeros((len(y_test),base_points))
for i in range(base_points):
    a = x_test1-x_train1[point_argument[i]]
    x_test[:,i] = np.exp(-1*(np.square(np.linalg.norm(a,axis=1)))*l)


# In[74]:


theta,bias = np.random.randn(np.shape(x_train)[1],classes),np.random.randn(1,classes)


# In[75]:


learning_rate = 0.2 ##0.3 was best for l=0.01 and points=500
iteration = 3000
print("Initial Cross Entropy")
print(cross_entropy(x_train,y_train,theta,bias))
batch_size = 50
crs = np.zeros(iteration)
u=int(len(y_train)/batch_size)
for i in range(iteration):
    for j in range(batch_size):
        x_train1 = x_train[j*(u):(j+1)*(u)]
        y_train1 = y_train[j*(u):(j+1)*(u)]
        

        theta = theta -learning_rate*del_cross_entropy(x_train1,y_train1,theta,bias)
        bias = bias-learning_rate*del_cross_entr_bias(x_train1,y_train1,theta,bias)
    k=cross_entropy(x_train,y_train,theta,bias)
    print("iteration:  ",i," cross entropy:",k)
    crs[i] = k/1000

#Lets plot the cross entropy with each iteration and see if its convex or not?
mt.pyplot.plot(np.arange(iteration),crs)



# In[76]:


#Prediction over training data
l = hypothesis(x_train,theta,bias)
train_prediction = np.argmax(l,axis=1)
e= (np.equal(y_train,train_prediction))*1
print("ACCURACY ON TRAINING sET")
print((np.sum(e))/len(y_train))


#Prediction over Testing data
l = hypothesis(x_test,theta,bias)
d = np.argmax(l,axis=1)
e= (np.equal(y_test,d))*1
print("ACCURACY ON Testing sET")
print((np.sum(e))/len(y_test))
# Author : Shubhansh Awasthi
