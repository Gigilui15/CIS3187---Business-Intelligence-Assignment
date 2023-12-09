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

# Softmax function for the output layer
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Sigmoid function for the hidden and output layer neurons
def sigmoid(x):
    x = np.clip(x, -500, 500)
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
epochs = 10000
input_layer_size = 4
hidden_layer_size = 4
output_layer_size = 3

#Seeding for reproducability
# Seeding for reproducibility
np.random.seed(10)

# Generating random weights for the input layer (4x4)
input_weights = np.random.uniform(size=(input_layer_size, hidden_layer_size))
# Generating random weights for the hidden layer (4x3)
hidden_weights = np.random.uniform(size=(hidden_layer_size, output_layer_size))

facts = []

def forward_propagation(inputs, input_weights, hidden_weights):
    hidden_layer_input = np.dot(inputs, input_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, hidden_weights)
    output_layer_output = softmax(output_layer_input)
    return hidden_layer_output, output_layer_output

def backpropagation(inputs, input_weights, hidden_weights, output_layer_output, outputs):
    delta_output = (output_layer_output - outputs) / len(inputs)
    hidden_weights += learning_rate * np.dot(hidden_layer_output.T, delta_output)
    
    delta_hidden = sigmoid_derivative(hidden_layer_output) * np.dot(delta_output, hidden_weights.T)
    input_weights += learning_rate * np.dot(inputs.T, delta_hidden)

    return hidden_weights, input_weights

# Loading the training inputs (120x4)
inputs = training_data.iloc[:, :-3].values
# Loading the training answers
outputs = training_data.iloc[:, -3:].values

# Training loop with forward and backward propagation
for epoch in range(epochs):
    hidden_layer_output, output_layer_output = forward_propagation(inputs, input_weights, hidden_weights)

    epoch_facts = []
    correct_predictions = 0

    error = outputs - output_layer_output

    for i in range(len(inputs)):
        
        if np.any(error < error_threshold):
            status = 'Good Fact'
        else:
            status = 'Bad Fact'
            hidden_weights, input_weights = backpropagation(inputs, input_weights, hidden_weights, output_layer_output, outputs)

        fact = {
            'Epoch No:': epoch + 1,
            'Index:': i,
            'Expected Output:': outputs[i],
            'Actual Output:': output_layer_output[i],
            'Error:': error,
            'Status: ': status
        }
        epoch_facts.append(fact)

    accuracy = correct_predictions / len(inputs)

    facts.append(epoch_facts)

facts_df = pd.DataFrame([fact for epoch_facts in facts for fact in epoch_facts])
facts_df.to_csv('C:\\Users\\luigi\\Desktop\\Third Year\\Business Intelligence\\Model Results\\Allfacts.csv', index=False)

print("Done")