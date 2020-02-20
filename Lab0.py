#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf

import os
import numpy as np

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import pandas as pd


# In[3]:


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)
# Disable some troublesome logging.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[41]:


# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
#ALGORITHM = "custom_net"
ALGORITHM = "tf_net"


# In[43]:


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        f = self.__sigmoid(x)
        df = f * (1 - f)
        return df

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def __loss(self, Y, Y_hat):
        L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
        m = Y.shape[1]
        L = -(1/m) * L_sum
        return L
    
    def relu(self, x):
        return np.maximum(0,x)
        
    def mse(self, Y, Y_hat):
        diff = (Y_hat - Y) ** 2
        return np.sum(diff)/Y.shape[1]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 1, minibatches = True, mbs = 100):
        for i in range(epochs):
            x_instance = self.__batchGenerator(xVals, mbs)
            y_instance = self.__batchGenerator(yVals, mbs)
            L1a = 0
            L2a = 0
            for j in range(0,xVals.shape[0],mbs):
                x_batch = next(x_instance)
                y_batch = next(y_instance)
                x_batch = x_batch.reshape(x_batch.shape[0], -1)
                layer1, layer2 = self.__forward(x_batch)
                L2e = (y_batch - layer2)
                L2d = L2e * self.__sigmoidDerivative(layer2)
                L1e = np.dot(L2d, self.W2.T)
                L1d = L1e * self.__sigmoidDerivative(layer1)
                L1a = np.dot(x_batch.T,L1d) * self.lr
                L2a = np.dot(layer1.T, L2d) * self.lr
                self.W1 += L1a
                self.W2 += L2a
            loss = self.mse(y_batch, layer2)
            print("Epoch No: ", i, "Loss: ",loss)
            
    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2 

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain, xTest = xTrain/255.0, xTest/255.0
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        net = NeuralNetwork_2Layer(IMAGE_SIZE,NUM_CLASSES,128)
        net.train(xTrain, yTrain)
        print("Building and training Custom_NN.")
        return net
    elif ALGORITHM == "tf_net":
        model = keras.models.Sequential()
        lossType = keras.losses.categorical_crossentropy
        inShape = (IMAGE_SIZE,)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, input_shape= inShape, activation=tf.nn.relu))
        model.add(keras.layers.Dense(10,activation=tf.nn.softmax))
        model.compile(optimizer='adam', loss=lossType)
        model.fit(xTrain,yTrain,epochs=1)
        print("Building and training TF_NN.")
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        data = data.reshape(data.shape[0], -1)
        pred = model.predict(data)
        ans = np.zeros_like(pred)
        ans[np.arange(len(pred)),pred.argmax(1)] = 1
        return ans
    elif ALGORITHM == "tf_net":
        data = data.reshape(data.shape[0], -1)
        pred = model.predict(data)
        ans = np.zeros_like(pred)
        ans[np.arange(len(pred)),pred.argmax(1)] = 1
        print("Testing TF_NN.")
        return ans
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    yTestResult = []
    predsResult = []
    for i in range(preds.shape[0]):
        yTestResult.append(np.argmax(yTest[i]))
        predsResult.append(np.argmax(preds[i]))
    yTestResult = pd.Series(yTestResult)
    predsResult = pd.Series(predsResult)
    df = pd.crosstab(yTestResult,predsResult)
    k = 0
    prec = []
    recall = []
    for i in range(10):
        x = df[i]
        num = x[k]
        summ = 0
        for j in range(10):
            summ += x[j]
        prec.append(num/summ)
        k += 1
    k = 0
    for i in range(10):
        num = df[i][k]
        summ = 0
        for j in range(10):
            summ += df[j][i]
        k += 1
        recall.append(num/summ)
    print(df)
    for i in range(10):
        f1 = (2 * prec[i] * recall[i])/(prec[i] + recall[i])
        print("F1 score of", i, "is:",f1)
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:




