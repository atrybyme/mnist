
# coding: utf-8
# Author: Shubhansh Awasthi




#import important libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils



#building cnn model for mnist
def cnn_model():
    model = Sequential()
    model.add(Conv2D(30,(5,5),input_shape=(28,28,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model



#import the data from csv file.
#the file contains 42000 labelled examples
_data_raw = (np.genfromtxt('train.csv',delimiter=','))[1:]




#we import sklearn just to shuffle the dataset
import sklearn
x_tr,y = sklearn.utils.shuffle((_data_raw[:,1:])/255,_data_raw[:,0])




#one hot encoding
y = np_utils.to_categorical(y)




#reshape and split the data for training and testing
x= np.reshape(x_tr,(42000,28,28,1))
x_train = x[0:41000]
y_train = y[0:41000]
x_test = x[41000:42000]
y_test = y[41000:42000]




#call the cnn model already build
model = cnn_model()

#fit the data over training dataset, by deviding the into 128 batches and using 1 batch at a time
#the model will fit by iterating 10 times over the whole training data
#at each iteration 1% of the whole training data will be used for validation and rest for training
model.fit(x_train,y_train,batch_size=128,epochs=10,verbose=1,validation_split=0.01)
l = model.predict(x_train,verbose=1,batch_size=100)
train_prediction = np.argmax(l,axis=1)

#predictiong over Training Set
e= (np.equal(np.argmax(y_train,axis=1),train_prediction))*1
print("ACCURACY ON Training sET",(np.sum(e))/len(y_train))




#Prediction over the testing and checking test accuracy

l = model.predict(x_test,verbose=1)
prediction_test= np.argmax(l,axis=1)
e= (np.equal(np.argmax(y_test,axis=1),prediction_test))*1
print("ACCURACY ON Testing sET",((np.sum(e))/len(y_test))*100)



# coding: utf-8
# Author: Shubhansh Awasthi
