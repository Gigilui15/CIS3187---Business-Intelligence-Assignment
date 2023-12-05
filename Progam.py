import numpy as np
import pandas as pd

# Loading the dataset with no headers
dataset = pd.read_csv('C:\\Users\\luigi\\Desktop\\Third Year\\Business Intelligence\\IrisDataset.csv')

# One-Hot Encoding the Iris Type column
one_hot_encoded_data = pd.get_dummies(dataset, columns=['Type'], dtype=int)

# Save the one-hot encoded data to a CSV file
# This file is then opened in excel, randomised, and split into two csv's for training and testing using an 80/20 split
one_hot_encoded_data.to_csv('C:\\Users\\luigi\\Desktop\\Third Year\\Business Intelligence\\OneHotEncodedData.csv', index=False)

# Loading the training dataset
training_data = pd.read_csv('C:\\Users\\luigi\\Desktop\\Third Year\\Business Intelligence\\training80.csv')
# Loading the testing dataset
testing_data = pd.read_csv('C:\\Users\\luigi\\Desktop\\Third Year\\Business Intelligence\\testing20.csv')

# Sigmoid function for the hidden and output layer neurons
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative to be used in Error Backpropagation formula: error * sigmoid_derivative
def sigmoid_derivative(x):
    return x * (1 - x)

#loading the training inputs (120x4)
inputs = training_data.iloc[:, :-3].values
#loading the training answers
outputs = training_data.iloc[:, -3:].values

#Hyperparameters
learning_rate = 0.2
error_threshold = 0.2
epochs = 1000
input_layer_size = inputs.shape[1]
hidden_layer_size = 4
output_layer_size = 3

#Seeding for reproducability
np.random.seed(10)

#Generating random weights for the input layer (4x4)
input_weights = np.random.uniform(size=(input_layer_size,hidden_layer_size))

#Generating random weights for the hidden layer (4x3)
hidden_weights = np.random.uniform(size=(hidden_layer_size, output_layer_size))

#Fix dataset headers (shouldn't matter)
#Add Dropout Layers
#Hyperparameter Optimization
#Add Softmax to the Output Layer