# Neural-Network
# Feed-Forward Neural Network from Scratch in Python and Airflow Implementation

This project aims to implement a simple feed-forward neural network from scratch in Python using only the Numpy library and then use Apache Airflow to orchestrate the process of loading the data, training the model, and generating predictions. 

The purpose of this project was to better understand how Neural Networks are actually built and the mathematics which goes behind libraries such as PyTorch and Tensorflow. I also set the project up in a virtual environment and used Airflow to orchestrate predictions to better model production environments.

## Getting Started
These instructions will help you set up the project on your local machine.

### Prerequisites
You need to have the following installed:

Python 3.8+
Pipenv

### Installation
Clone the repo: git clone https://github.com/username/repo.git
Navigate to the cloned repo: cd repo
Set up the virtual environment: pipenv install
Activate the virtual environment: pipenv shell

Pipenv will be able to set up the correct virtual environment and install the relevant libraries using the Pipfile.

### Dependencies

The script is built using Python and makes use of the following libraries:
- numpy

## Code Details

### Perceptron

The provided code defines a Perceptron class, a basic building block of neural networks. The class includes an initialization method setting learning rate, number of epochs, and initializing weights and bias. The sigmoid method is the activation function, and its derivative is calculated for backpropagation. The loss_function computes the binary cross-entropy loss. The model is trained and tested on the breast cancer dataset from sklearn.datasets module. 

### SimpleNeuralNetwork

The code consists of a class `SimpleNeuralNetwork` that includes initialization function `__init__` and the following methods:

- `initialize_parameters`: Initialize the weights and biases for each layer in the network.

- `fwd`: Perform a forward pass through the network, calculating and storing the outputs for each layer.

- `bwd`: Perform a backward pass through the network, calculating and updating the gradients of the loss function with respect to the network's parameters.

- Activation functions: `relu`, `tanh`, `sigmoid`, `leaky_relu`, `identity`, `softmax` and their derivatives are implemented.

- Loss functions: `MSE`, `RMSE`, `MAE`, `Binary_Cross_Entropy`, `Categorical_Cross_Entropy` and their derivatives are implemented.

- Check datra: Basic checks for NaNs and dimensions.

### AirflowImplementation

To use the project:

Start the Airflow webserver: airflow webserver --port 8080
In a new terminal window, start the Airflow scheduler: airflow scheduler
Navigate to localhost:8080 in your web browser to access the Airflow UI.
Run the simple_neural_network DAG.
Project Structure
The project consists of the following files:

AirflowImplementation.py - The Airflow DAG that orchestrates the data loading, model training, and prediction generation.
SimpleNeuralNetwork.py - The SimpleNeuralNetwork class used to train the model and generate predictions.

### Testing

There are 4 scripts which I used to test the SimpleNeuralNetwrok class for Regression, Binary Classification, and Multi-Class Classification problems. The datasets used are available on Scikit-Learn (https://scikit-learn.org/stable/datasets.html)

## Future Implementations

This project combines elements of Data Science, Deep Learning, and Data Engineering. Over the next few weeks I aim to implement more features such as Stcohastic Gradient Descent and Regularization for the SimpleNeuralNetwork class.

## Acknowledgements and References

- https://towardsdatascience.com/how-to-choose-the-right-activation-function-for-neural-networks-3941ff0e6f9c
- https://towardsdatascience.com/a-data-scientists-guide-to-python-virtual-environments-858841922f14
- https://towardsdatascience.com/building-a-neural-network-from-scratch-8f03c5c50adc
- https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0#:~:text=%E2%80%9CEssentially%2C%20backpropagation%20evaluates%20the%20expression,%E2%80%9Cbackwards%20propagated%20error).%E2%80%9D
- https://towardsdatascience.com/neural-network-the-dead-neuron-eaa92e575748
- https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks
- https://studymachinelearning.com/mathematics-behind-the-neural-network/
