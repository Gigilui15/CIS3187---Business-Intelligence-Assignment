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

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_propagation(fit, input_weights, hidden_weights):
    hidden_layer_input = np.dot(fit, input_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, hidden_weights)
    output_layer_output = softmax(output_layer_input)
    return hidden_layer_output, output_layer_output

def backpropagation(inputs, input_weights, hidden_weights, output_layer_output, outputs, hidden_layer_output):
    output_error = outputs - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    hidden_error = output_delta.dot(hidden_weights.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    hidden_weights += hidden_layer_output.T.dot(output_delta) * learning_rate
    input_weights += inputs.T.dot(hidden_delta) * learning_rate

    return hidden_weights, input_weights

#loading the training inputs (120x4)
inputs = training_data.iloc[:, :-3].values
#loading the training answers
outputs = training_data.iloc[:, -3:].values

#Hyperparameters
learning_rate = 0.25
error_threshold = 0.2
epochs = 10000
input_layer_size = 4
hidden_layer_size = 4
output_layer_size = 3

# Seeding for reproducibility
np.random.seed(1)

# Generating random weights for the input layer (4x4)
input_weights = np.random.uniform(size=(input_layer_size, hidden_layer_size))
# Generating random weights for the hidden layer (4x3)
hidden_weights = np.random.uniform(size=(hidden_layer_size, output_layer_size))


for epoch in range(epochs):
    print("Epoch: ", epoch + 1)
    bad_facts_count = 0 
    for example in range(len(inputs)):

        fit = inputs[example].reshape(1, -1)
        hidden_layer_output, output_layer_output = forward_propagation(fit, input_weights, hidden_weights)

        error = outputs[example] - output_layer_output

        if any(abs(e) > error_threshold for e in error[0]):
            hidden_weights, input_weights = backpropagation(fit, input_weights, hidden_weights, output_layer_output, outputs[example], hidden_layer_output)
            bad_facts_count += 1  

    print(f"Number of bad facts in epoch {epoch + 1}: {bad_facts_count}")

    # Check if bad facts occurred, and break if not
    if bad_facts_count == 0:
        print("No bad facts in the last epoch. Stopping training.")
        break

