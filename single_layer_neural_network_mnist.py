
# coding: utf-8

# ### Hand Digit Classification using "1 Layer Neural Network(Softmax Classification)"

# In[12]:


#import required libraries
from random import *
from matplotlib import pyplot as plt
import numpy as np
#%matplotlib inline


# In[13]:


#one hot encoding
def y_usable(y):
    y_1 = np.ones((len(y),1))
    th = np.array([0,1,2,3,4,5,6,7,8,9])
    y_2 = y_1*th
    y_3 = np.reshape(y,(len(y),1))
    y_4= (np.equal(y_2,y_3))*1
    return y_4


# In[15]:


#create the hypothesis function
def hypothesis(x_train,theta,bias):
    a = np.dot(x_train,theta)
    a = np.exp(a+bias)
    b = np.sum(a,axis=1)
    a = a/np.reshape(b,(len(b),1))
    return a


# In[16]:


#cross entropy calculation function
def cross_entropy(x,y,theta,bias):
    h = hypothesis(x,theta,bias)
    e = np.log(h)
    y_1 = y_usable(y)
    return np.sum(np.sum(np.multiply(y_1,e)))


# In[17]:


#derivative of cross entropy with respect to variable theta
def del_cross_entropy(x,y,theta,bias):
    m = len(y)
    h = hypothesis(x,theta,bias)
    y_1 = y_usable(y)
    a = h-y_1
    x_tr = np.transpose(x)
    d = (np.dot(x_tr,a))/m
    return d


# In[18]:


#derivative of cross entropy with respect ot bias
def del_cross_entr_bias(x,y,theta,bias):
    m = len(y)
    h = hypothesis(x,theta,bias)
    y_1 = y_usable(y)
    a = h-y_1
    d = (np.sum(a,axis=0))/m
    return d


# In[19]:


#import the data from csv file contating 42000 labelled examples
training_data_raw = (np.genfromtxt('train.csv',delimiter=','))[1:]


# In[32]:


#sklearn is used just to shuffle the data
import sklearn
x,y = sklearn.utils.shuffle((training_data_raw[:,1:])/255,training_data_raw[:,0])
classes = 10
#subtract the mean from the data
x = x-np.mean(x,axis=0)
#split the data into training and testing
x_train = x[0:41000]
y_train = y[0:41000]
x_test = x[41000:42000]
y_test = y[41000:42000]
theta,bias = np.random.randn(np.shape(x_train)[1],classes),np.random.randn(1,classes)


# In[34]:


#we will fit the data using gradient descent iterating over whole data 500 times
#the data will be devided into 100 batches and parameter update will occur using 1 batch at a time
#loss function used here if cross entropy
learning_rate = 0.3
iteration = 500
cross_entr = np.zeros(iteration)
batch_size = 100
u=int(len(y_train)/batch_size)
print("Initial Cross Entropy")
print(cross_entropy(x_train,y_train,theta,bias))
for i in range(iteration):
    for j in range(batch_size):
        x_train1 = x_train[j*(u):(j+1)*(u)]
        y_train1 = y_train[j*(u):(j+1)*(u)]
        theta = theta -learning_rate*del_cross_entropy(x_train1,y_train1,theta,bias)
        bias = bias-learning_rate*del_cross_entr_bias(x_train1,y_train1,theta,bias)
    a = cross_entropy(x_train,y_train,theta,bias)
    print("iteration:",i,"  cross-entropy :", a)
    cross_entr[i] = a
print("Final cross entropy")
print(cross_entropy(x_train1,y_train1,theta,bias))

plt.plot(np.arange(iteration),cross_entr)


#Prediction over TRAINING DATA
l = hypothesis(x_train,theta,bias)
train_prediction = np.argmax(l,axis=1)
e= (np.equal(y_train,train_prediction))*1
print("ACCURACY ON TRAINING sET")
print((np.sum(e))/len(y_train))


#Prediction over testing dataset
l = hypothesis(x_test,theta,bias)
d = np.argmax(l,axis=1)
e= (np.equal(y_test,d))*1
print("ACCURACY ON Testing sET")
print((np.sum(e))/len(y_test))

