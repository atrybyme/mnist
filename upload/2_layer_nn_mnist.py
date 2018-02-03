
# coding: utf-8
# Author: Shubhansh Awasthi


##import important libraries
from random import *
import numpy as np
import matplotlib as mt
import preprocessing as pr
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
##we imported sklearn just to shuffle the data
import sklearn



##build your tiny 1 hiddenlayer neural network model
def nn_model():
    model = Sequential()
    model.add(Dense(800, input_dim=784, kernel_initializer='normal',bias_initializer='normal',activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
    ##model will use adam optimize
    ##the lost function used is categorial cross-entropy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



##import data from csv file the data contains 42000 labelled examples
training_data_raw = (np.genfromtxt('train.csv',delimiter=','))[1:]


##shuffle the data
x,y = sklearn.utils.shuffle((training_data_raw[:,1:])/255,training_data_raw[:,0])




##doing one hot encoding
y = np_utils.to_categorical(y)




#splitting the data for training and testing
x_train = x[0:41000]
y_train = y[0:41000]
x_test = x[41000:42000]
y_test = y[41000:42000]



#calling the model
model = nn_model()

#fitting the model over training dataset by splitting the dataset into 200 batches ,
#running 1 batch at a time and iterating 200 times over the whole dataset.
#99% of training data will be used for training and 1% of the training data will be used for validationat each iteration
model.fit(x_train,y_train,batch_size=200,epochs=10,verbose=1,validation_split=0.01)

#predict the output for training data
l = model.predict(x_train,verbose=1,batch_size=100)
train_prediction = np.argmax(l,axis=1)
e= (np.equal(np.argmax(y_train,axis=1),train_prediction))*1
print("ACCURACY ON Training sET",((np.sum(e))/len(y_train))*100)


#predict the output for tesating data
l = model.predict(x_test,verbose=1)
prediction_test= np.argmax(l,axis=1)

#check the accuracy over training data
e= (np.equal(np.argmax(y_test,axis=1),prediction_test))*1
print("ACCURACY ON Testing sET",((np.sum(e))/len(y_test))*100)


# coding: utf-8
# Author: Shubhansh Awasthi
