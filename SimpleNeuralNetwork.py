import numpy as np

# User inputs the number of neurons in each successive hidden layer in the form of a list.

#### If Regression, one output Neuron.
#### If Classification need to specify how many classes. Will one hot encode the target labels according to the number of classes. Classes = 1 implies Regression problem.
########## Also for Classification, if there are just two classes we can either use one sigmoid activation neuron or two softmax activated nuerons. 
########## In this case it is better to use 2 softmax neurons for 2 classes to prevent confusion in the use case of one output neuron for regression.

#### Regularization

class SimpleNeuralNetwork:
    def __init__(self, learning_rate, epochs, activation: str, loss_function: str, layers: list, classes: int):
        self.lr = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.loss_function = loss_function
        # self.losses = []
        
        self.activation_funcs = {
            "relu": self.relu,
            "tanh": self.tanh,
            "sigmoid": self.sigmoid,
            "leaky_relu": self.leaky_relu,
            "identity": self.identity,
            "softmax": self.softmax
        }
        self.activation_derivs = {
            "relu": self.relu_derivative,
            "tanh": self.tanh_derivative,
            "sigmoid": self.sigmoid_derivative,
            "leaky_relu": self.leaky_relu_derivative,
            "identity": self.identity_derivative,
            "softmax_derivative": self.softmax_derivative
        }
        self.loss_funcs = {
            "MSE": self.MSE,
            "RMSE": self.RMSE,
            "MAE": self.MAE,
            "Binary_Cross_Entropy": self.Binary_Cross_Entropy,
            "Categorical_Cross_Entropy": self.Categorical_Cross_Entropy
        }
        self.loss_derivs = {
            "MSE": self.MSE_derivative,
            "RMSE": self.RMSE_derivative,
            "MAE": self.MAE_derivative,
            "Binary_Cross_Entropy": self.Binary_Cross_Entropy_derivative,
            "Categorical_Cross_Entropy": self.Categorical_Cross_Entropy_derivative
        }
        assert self.activation in self.activation_funcs.keys()
        assert self.loss_function in self.loss_funcs.keys()
        self.layers = layers
        assert type(layers) is list
        self.classes = classes
        assert type(classes) is int
        # self.batch_size_percentage = batch_size_percentage
        self.batch_size = None

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
        return e_x / e_x.sum(axis=1, keepdims=True)

    def softmax_derivative(self, x):
        pass
        # # Initialize the 2D Jacobian matrix.
        # jacobian_m = np.zeros((len(x), len(x)))

        # # Fill the diagonal.
        # for i in range(len(jacobian_m)):
        #     for j in range(len(jacobian_m)):
        #         if i == j:
        #             jacobian_m[i][j] = x[i] * (1 - x[i])
        #         else: 
        #             jacobian_m[i][j] = -x[i] * x[j]
        # return jacobian_m
    
    def identity(self,x):
        return x

    def identity_derivative(self,x):
        return 1
    
    def sigmoid(self,x):
        return np.clip(1/(1+np.exp(-x)), 1e-7, 1-1e-7)
    
    def sigmoid_derivative(self,x):
        sx = self.sigmoid(x)
        return sx * (1-sx)

    def tanh(self,x):
        return np.tanh(x)
    
    def tanh_derivative(self,x):
        return 1 - np.square(np.tanh(x))

    def relu(self,x):
        return np.maximum(0, x)

    def relu_derivative(self,x):
        output = (x > 0)*1
        return output

    def leaky_relu(self,x):
        return np.where(x > 0, x, x * 0.01)
    
    def leaky_relu_derivative(self,x):
        return np.where(x>0,1,0.01)
    
    def Binary_Cross_Entropy(self,y,a):
        return -1 * np.mean(y*np.log(a) + (1 - y)*np.log(1 - a))
    
    def Binary_Cross_Entropy_derivative(self,y,a):
        return (-y/a) + (1-y)/(1-a)
    
    def Categorical_Cross_Entropy(self, y, a):
        n_samples = y.shape[0]
        log_likelihood = -np.log(a[np.arange(n_samples), y.argmax(axis=1)])
        loss = np.sum(log_likelihood) / n_samples
        return loss

    def Categorical_Cross_Entropy_derivative(self, y, a):
        a = a.clip(min=1e-8,max=None)
        n_samples = y.shape[0]
        a_copy = a.copy()
        a_copy[np.arange(n_samples), y.argmax(axis=1)] -= 1
        return a_copy / n_samples
    
    def MSE(self,y,a):
        return np.mean(np.square(a-y))
    
    def MSE_derivative(self,y,a):
        n = len(y)
        return (2/n) * (a - y)
    
    def RMSE(self,y,a):
        return np.sqrt(np.mean(np.square(a-y)))
    
    def RMSE_derivative(self,y,a):
        n = len(y)
        mse = np.mean((a - y) ** 2)
        return (a - y) / (n * np.sqrt(mse))

    def MAE(self,y,a):
        return np.mean(np.abs(a-y))
    
    def MAE_derivative(self,y,a):
        n = len(y)
        return (1/n) * np.sign(a - y)

    def initialize_parameters(self, X, list_layers):
        # Find shape of input X - (input array has samples in rows and features in columns).
        # Find length of Layers list - this gives the number of hidden layers to be built.
        # In the layers list, insert the number of features at the start.

            # Hidden Layer 1 size = (num_features,num_neurons_H1) ; Biases = (num_inputs, 1)
            # Hidden Layer 2 size = (num_neurons_H1, num_neurons_H2) ; Biases = (num_inputs, 1)
            # ........
            # Hidden Layer n size = (num_neurons_n-1, num_output_neurons) ; Biases = (num_inputs, 1)
        
        # num_output_neurons = 1 if regression, K if K-class classification
        # Output layer size = (num_columns_layer_n,1) if regression, (num_columns_layer_n,1)

        np.random.seed(1)  
        self.weights = []
        self.biases = []

        layers = [X.shape[1]] + list_layers
        if self.classes > 2:
            layers.append(self.classes)
        else:
            layers.append(1)

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.01)
            self.biases.append(np.zeros((1,layers[i+1])))
            print(f"Layer {i+1}: Weights shape: {self.weights[i].shape}, Biases shape: {self.biases[i].shape}")
    
    def fwd(self, X):
        # Need to store outputs from consecutive layers as they will need to be referred back during backprop
        # Start by setting A as input X
        self.A = [X]
        print(f"input shape: {X.shape}")
        self.Z = []

        # Process all layers except the final one
        for i in range(len(self.weights) - 1):
            Z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            print(f"Layer {i+1} output shape: {Z.shape}")
            self.Z.append(Z)
            A = self.activation_funcs[self.activation](Z)
            print(f"Layer {i+1} activation shape: {A.shape}")
            self.A.append(A)
        
        # For the final layer, choose the activation function based on the loss function
        Z = np.dot(self.A[-1], self.weights[-1]) + self.biases[-1]
        print(f"Final Layer {i+2} output shape: {Z.shape}")
        self.Z.append(Z)
        if self.loss_function in ['MSE', 'MAE', 'RMSE']:
            A = self.activation_funcs['identity'](Z)
            print(f"Final Layer {i+2} activation shape: {A.shape}")
        elif self.loss_function == 'Binary_Cross_Entropy':
            A = self.activation_funcs['sigmoid'](Z)
            print(f"Final Layer {i+2} activation shape: {A.shape}")
        elif self.loss_function == 'Categorical_Cross_Entropy':
            A = self.activation_funcs['softmax'](Z)
            print(f"Final Layer {i+2} activation shape: {A.shape}")
        print(A[:5, :])    
        self.A.append(A)
        
        return self.A[-1]

    def bwd(self, X, y, A):
        # backpropagation
        # Start by calculating the derivative of Loss wrt predicted outputs from final output layer. The activation for the final output layer might be different from the hidden layer activations.
        # Chain rule to last Hidden Layer. The last Hidden layer would be connected 1:1 with the final output layer.
        # Need to account for multiple connections between Hidden Layers which would include an addition component in the calculus.

        m = X.shape[0]
        # loss = self.loss_funcs[self.loss_function](y, A)
        # self.losses.append(loss)

        if self.classes == 1:
            self.dZ = [self.loss_derivs[self.loss_function](y, A) * self.identity_derivative(self.Z[-1])]
        elif self.classes == 2:
            self.dZ = self.loss_derivs[self.loss_function](y, A) * self.sigmoid_derivative(self.Z[-1])
            self.dZ = [dZ_i.reshape(-1,1) for dZ_i in self.dZ]
        else:
            # self.dZ = [self.loss_derivs[self.loss_function](y, A) * self.softmax_derivative(self.Z[-1])]
            self.dZ = [A - y]

        
        print(f"dL/dZ last shape: ",self.dZ[0].shape)

        self.dW = [np.dot(self.A[-2].T,self.dZ[-1]) / m]
        print(f"dL/dW last set shape: ",self.dW[0].shape)
        self.dB = [np.sum(self.dZ[-1], axis=0, keepdims=True) / m]
        print(f"dL/dB last set shape: ",self.dB[0].shape)

        for i in range(len(self.weights) - 2, -1, -1):
            dZ = np.dot(self.dZ[-1], self.weights[i+1].T) * self.activation_derivs[self.activation](self.Z[i])
            # dW = np.dot(dZ.T, self.A[i]) / m
            dW = np.dot(self.A[i].T, dZ) / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            print(f"dZ shape: {dZ.shape}, self.A[i] shape: {self.A[i].shape}, dW shape: {dW.shape}, dB shape: {dB.shape}")
            self.dZ.append(dZ)
            # self.dW.append(dW.T)
            self.dW.append(dW)
            self.dB.append(dB)

    def bwd2(self, X, y, A):
        # 1. Get length of 'Layers' list set up in "initialise_parameters". 
        #    Length 'n' = sets of weights involved. 
        #    Let the output involve k Classes (1 for Regression).
        #    '*' Hadamard Product
        #    '.' Dot Product
        #    '_n' Layer index
        #    'm_n' Number of neurons in layer with index n and similarly 'm_n-1' is the number of neurons in layer with index n-1.
        #    Since we already append the number of output classes (indicated by the user) to the Layers list in the function "initialise_parameters", m_n will be equal to k.

        # 2. Calculate dL/dA_n and dA_n/dZ_n. Both of dimenstions (k,1), where (k,1) = (m_n,1)
        # 3. Delta_n (m_n,1) = dL/dA_n (m_n,1) * dA_n/dZ_n (m_n,1)
            # 4. dL/dW_n (m_n-1,m_n) = Delta_n * dZ_n/dW_n (m_n-1,m_n)
        # 5. Delta_n-1 (m_n-1,1) = dA_n-1/dZ_n-1 (m_n-1,1) * (W_n (m_n-1,m_n) . Delta_n (m_n,1))
            # 6. dL/dW_n-1 (m_n-2,m_n-1) = Delta_n-1 * dZ_n-1/dW_n-1 (m_n-2,m_n-1)
        # 7. Delta_n-2 (m_n-2,1) = dA_n-2/dZ_n-2 (m_n-2,1) * (W_n-1 (m_n-2,m_n-1) . Delta_n-1 (m_n-1,1))
            # 8. dL/dW_n-2 (m_n-3,m_n-2) = Delta_n-2 * dZ_n-2/dW_n-2 (m_n-3,m_n-2)
        # Repeat till we have 'n' Deltas and we get the gradient of Loss wrt 1st set of weights.

        m = X.shape[0]
        self.dZ = []
        self.dW = []
        self.dB = []
        loss = self.loss_funcs[self.loss_function](y, A)
        self.losses.append(loss)

        # Step 2: Calculate dL/dA_n and dA_n/dZ_n
        if self.classes == 1:
            dZ = self.loss_derivs[self.loss_function](y, A) * self.identity_derivative(self.Z[-1])
        elif self.classes == 2:
            dZ = self.loss_derivs[self.loss_function](y, A) * self.sigmoid_derivative(self.Z[-1])
        else:
            dZ = [A - y]

        # Step 3: Delta_n
        self.dZ.append(dZ)
        print(f"dL/dZ_last shape: ",self.dZ[0].shape)

        for i in range(len(self.weights) - 1, -1, -1):
            # Step 4 & 6 & 8: dL/dW_n
            dW = np.dot(self.A[i].T, self.dZ[-1]) / m
            self.dW.append(dW)

            # Step 5 & 7: Delta_n-1
            if i > 0:
                dZ = np.dot(self.dZ[-1], self.weights[i].T) * self.activation_derivs[self.activation](self.Z[i-1])
                self.dZ.append(dZ)

            # Bias update
            dB = np.sum(self.dZ[-1], axis=0, keepdims=True) / m
            self.dB.append(dB)

        self.dW = self.dW[::-1]
        self.dB = self.dB[::-1]


    def train(self, X, y):
        self.check_data(X, y)
        self.initialize_parameters(X, self.layers)
        self.y_encoded = 0
        print(f"Y shape : ", y.shape)
        # Perform one-hot encoding on y
        if self.classes > 2:
            self.y_encoded = np.eye(self.classes)[y]
        elif self.classes == 1:
            self.y_encoded = y.reshape(-1,1)
        else:
            self.y_encoded = y
        
        print(f"Y Encoded shape : ", self.y_encoded.shape)
        
        m = X.shape[0]
        self.losses = []
        #self.batch_size = int(m * self.batch_size_percentage)
        self.count = 0

        for _ in range(self.epochs-1):
            print(f"Epoch: ",self.count+1)
            # Shuffle the data at the beginning of each epoch
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = self.y_encoded[permutation]

            # Forward and backward propagation
            A = self.fwd(X_shuffled)
            loss = self.loss_funcs[self.loss_function](self.y_encoded, A)
            print(f"loss shape : ", loss.shape)
            self.losses.append(loss)
            self.bwd(X_shuffled, y_shuffled, A)

            # Update weights and biases
            for j in range(len(self.weights)):
                self.weights[j] -= self.lr * self.dW[-1 - j]
                print(self.weights[j][:3,:])
                self.biases[j] -= self.lr * self.dB[-1 - j]

            self.count += 1

    def predict(self, X):
        A = self.fwd(X)
        return np.round(A)
    

    def check_data(self, X, y):
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("The input data contains NaN values. Please clean the data before training.")
        
        if np.count_nonzero(X) != np.size(X):
            raise ValueError("The input data contains zero values. Please clean the data before training.")
        
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Inputs X and y must be numpy arrays.")
        
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Input X should be 2D and input y should be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Inputs X and y should have the same number of samples.")
        
        # For classification, check that y contains the correct number of unique values
        if self.classes > 1 and len(np.unique(y)) != self.classes:
            raise ValueError(f"Input y should contain {self.classes} unique values for a {self.classes}-class classification problem.")

##########################################################################