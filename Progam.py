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
    # Use the clip function to avoid overflow issues
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


# Sigmoid derivative to be used in Error Backpropagation formula: error * sigmoid_derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Softmax function for the output layer
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

#loading the training inputs (120x4)
inputs = training_data.iloc[:, :-3].values
#loading the training answers
outputs = training_data.iloc[:, -3:].values

#Hyperparameters
learning_rate = 0.2
error_threshold = 0.2
epochs = 1000
input_layer_size = 4
hidden_layer_size = 4
output_layer_size = 3

#Seeding for reproducability
np.random.seed(10)

#Generating random weights for the input layer (4x4)
input_weights = np.random.uniform(size=(input_layer_size,hidden_layer_size))
#Generating random weights for the hidden layer (4x3)
hidden_weights = np.random.uniform(size=(hidden_layer_size, output_layer_size))

facts = []

def forward_propagation(inputs, input_weights, hidden_weights):
    # Calculate values for the hidden layer
    hidden_layer_input = np.dot(inputs, input_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Calculate values for the output layer
    output_layer_input = np.dot(hidden_layer_output, hidden_weights)
    output_layer_output = sigmoid(output_layer_input)
    return hidden_layer_output, output_layer_output

def backpropagation(inputs, input_weights, hidden_weights, output_layer_output, error):
    #Output Layer Weights
    delta_output = error * sigmoid_derivative(output_layer_output)
    hidden_weights += learning_rate * np.dot(hidden_layer_output.T, delta_output)
    #Hidden Layer Weights
    delta_hidden = sigmoid_derivative(hidden_layer_output) * np.dot(delta_output, hidden_weights.T)
    input_weights += learning_rate * np.dot(inputs.T, delta_hidden)
    
    return hidden_weights, input_weights

#Training loop with forward and backward propagation
for epoch in range(epochs):
    # Run Feed Forward Propagation
    hidden_layer_output, output_layer_output = forward_propagation(inputs, input_weights, hidden_weights)
    
    #Checking for Bad Facts to perform Backpropagation
    #Includes Error Calculation and Saving of Good & Bad Facts in a CSV
    epoch_facts = []
    
    # Calculate the error [120 x 3] using crossentropy
    error = outputs - softmax(output_layer_output)

    for i in range(len(inputs)):
        # Check if all errors are within the threshold
        if np.all(error[i] <= error_threshold):
            status = "Good Fact"
        else:
            status = "Bad Fact"
            # Run Backpropagation
            hidden_weights, input_weights = backpropagation(inputs, input_weights, hidden_weights, output_layer_output, error)
            fact = {
                'Epoch No:': epoch + 1,
                'Index:': i,
                'Expected Output:': outputs[i],
                'Actual Output:': output_layer_output[i],
                'Error:': error[i],
                'Fact:': status
            }
            epoch_facts.append(fact)

    # Append the facts for the current epoch to the main facts list
    facts.append(epoch_facts)

facts_df = pd.DataFrame([fact for epoch_facts in facts for fact in epoch_facts])
facts_df.to_csv('C:\\Users\\luigi\\Desktop\\Third Year\\Business Intelligence\\Model Results\\Allfacts.csv', index=False) 
print("Done")