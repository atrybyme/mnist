
# coding: utf-8

# In[4]:


import numpy as np


# In[1]:


##X is input matrix and value is the number of features you want in the data after PCA
def PCA(X,value):
    ##take data as n*m matrix. n is number of example,m id dimension of each example
    mean =np.mean(X,axis=0)
    X_m = X-mean
    ##covariance matrix
    covX = np.dot(np.transpose(X_m),X_m)
    conX = covX/(np.shape(X)[0])
    #get eagen value and corresoponding eagen vector
    featurevec = []
    eagenval,eagenvec = np.linalg.eig(conX)
    a = np.argmax(eagenval)
    featurevec = np.array([eagenvec[:,a]])
    eagenval = np.delete(eagenval,a)
    for i in range(value-1):
        a = np.argmax(eagenval)
        featurevec = np.append(featurevec,[eagenvec[:,a]],axis=0)
        eagenval = np.delete(eagenval,a)
    featurevecT = np.transpose(featurevec)
    return np.dot(X_m,featurevecT)


# In[3]:


##take x_train and amount,if pixel_value>amount make pixel_value = 1 otherwise pixel_value = 0
def pixelthresholdfilter(x_train,amount):
    image_number=1
    mtr = x_train>amount
    mtr = mtr*1
    return mtr

