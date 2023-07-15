import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Steps for Perceptron.

# 1. Taking X as input and Y as Target - rows of x should be unique x observations and columns should correcpond to different features of each observation
# 2. Initialising weight vector: nrow = number of features in each observation, ncol = 1, between -1 and 1
# 3. Intiailising bias vecctor: nrow = number of observations, ncol = 1, zeros
# 4. Defining sigmoid activation function = 1/(1+exp(-x))
# 5. Define derivative for sigmoid activation function = x*(1-x)
# 6. Define forward propagation step: Z = X*W + B
# 7. Apply Activation function on Z: sigmoid(Z)
# 8. Calculate Loss using the Binary Cross Entropy loss function
# 9. Find derivative of Loss function L wrt Output a
# 10. Find derivative of Output a wrt input Z
# 11. Find derivative of input Z wrt weight w
# 12. Find derivative of input z wrt bias b
# 13. Find overall derivative of Loss function with respect to weight - dL/dw
# 14. Update weight vector by w = w - learning rate*dL/dw
# 15. Update bias vector by b = b - learning rate*dL/db

class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self,x):
        return np.clip(1/(1+np.exp(-x)), 1e-7, 1-1e-7)
    
    def sigmoid_derivative(self,x):
        sx = self.sigmoid(x)
        return sx * (1-sx)
    
    def loss_function(self,y,a):
        return -1 * np.mean(y*np.log(a) + (1 - y)*np.log(1 - a))
    
    def loss_derivative(self,y,a):
        return (-y/a) + (1-y)/(1-a)

    def initialize_weights(self, feature_size):
        self.weights = 2 * np.random.random((feature_size, 1)) - 1
        self.bias = 0

    def fwd(self, X):
        Z = np.dot(X, self.weights) + self.bias
        A = self.sigmoid(Z)
        return A
    
    def bwd(self, X, y, A):
        loss = self.loss_function(y, A)
        dLdw = np.dot(X.T, (self.sigmoid_derivative(A) * self.loss_derivative(y, A)))
        dLdb = np.sum(self.sigmoid_derivative(A) * self.loss_derivative(y, A))
        return dLdw, dLdb, loss

    def train(self, X, y):
        self.initialize_weights(X.shape[1])

        for _ in range(self.epochs):
            A = self.fwd(X)
            dLdw, dLdb, loss = self.bwd(X, y, A)

            self.weights -= self.lr * dLdw
            self.bias -= self.lr * dLdb

    def predict(self, X):
        A = self.fwd(X)
        return np.round(A)

##########################################################################################

bc = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
print(bc[0].head())
print(bc[1].head())

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(bc[0], bc[1], test_size=0.2, random_state=1) # 80% training and 20% test

# Standardize the features to have mean=0 and variance=1 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ensure the targets are in the correct shape
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1) 

##########################################################################################

p = Perceptron(learning_rate=0.01, epochs=1000)
p.train(X_train, y_train)
predictions = p.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, predictions))

