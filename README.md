## Classifying MNIST Dataset using Different Classification Algorithms

This repository is about some Classification Algorithms for **MNIST**

![MNIST][1]

All the basic machine learning models are made from scratch except 2 Layer Neural Network and Basic CNN model.

## Requirements

- Python(3.52 or 2.7.14)
- Numpy(for all the matrix operations)
- Matplotlib(to plot graph and for visualization)
- Sklearn(Just to shuffle the data)
- Keras(2.0.8)(Required in 2_layer_neural_network and Simple_CNN_Model)
- tensorflow-gpu(1.3.0)(Required in 2_layer_neural_network and Simple_CNN_Model)

## Algorithm with their Papers
- **K-Nearest Neighbours:**
  - [KNN Model-Based Approach in Classification][2]

- **Multiclass Sofmax Classification(1 Layer Neural Network):**
  - [Gated Softmax Classification][3]
  
- **Radial Basis Analysis Over Linear Classifier:**
  - [Efficient and Accurate Gaussian Image Filtering Using Running Sums][4]

- **2 Layer Neural Network (Relu Activation in all layer except Softmax at the top):**
  - [Convergence Analysis of Two-layer Neural Networks with ReLU Activation][5]
  
- **Simple CNN Model(The model in not exact Le_Net Architecture.):**
  - [Le_Net][6]
  
## Details
  [train.csv][7] contains 42000 labeled images of digits from 0 to 9. The image is 28*28(784) in dimensions.
  Each image is a black and white image.
  For most algorithms we have taken 41000 images for training and rest 1000 for testing.
  Except a simple filter no preprocessing is done because the aim of the project was to understand the different classification techniques.

## Order of Accuracy
  Simple_CNN > 2_Layer_NN > Radial_Basis_Analysis over linear classifier > 1 Layer NN > KNN
  
### Please feel free to contact me if you have any questions or suggestions.
  
[1]: mnist.png
[2]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.2.815&rep=rep1&type=pdf
[3]: http://www.cs.toronto.edu/~fritz/absps/gatedsoftmax.pdf
[4]: http://www.cs.huji.ac.il/~werman/Papers/paper42.pdf
[5]: https://papers.nips.cc/paper/6662-convergence-analysis-of-two-layer-neural-networks-with-relu-activation.pdf
[6]: http://yann.lecun.com/exdb/lenet/
[7]: https://github.com/atrybyme/mnist/blob/master/train.csv
