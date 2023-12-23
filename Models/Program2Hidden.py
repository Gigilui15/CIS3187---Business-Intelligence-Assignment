import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# Load the dataset with no headers
dataset = pd.read_csv('C:\\Users\\luigi\\Desktop\\Third Year\\Business Intelligence\\IrisDataset.csv')

# One-Hot Encode the Iris Type column
one_hot_encoded_data = pd.get_dummies(dataset, columns=['Type'], dtype=int)

# Save the one-hot encoded data to a CSV file
# This file is then opened in excel, randomized, and split into two CSVs for training and testing using an 80/20 split
one_hot_encoded_data.to_csv('C:\\Users\\luigi\\Desktop\\Third Year\\Business Intelligence\\OneHotEncodedData.csv', index=False)

# Load the training dataset
training_data = pd.read_csv('C:\\Users\\luigi\\Desktop\\Third Year\\Business Intelligence\\training80.csv')
# Load the testing dataset
testing_data = pd.read_csv('C:\\Users\\luigi\\Desktop\\Third Year\\Business Intelligence\\testing20.csv')

# Creating a folder to save weights
weights_folder = 'Weights'
os.makedirs(weights_folder, exist_ok=True)

# Define sigmoid function for the hidden and output layer neurons
def sigmoid(x):
    # Clip the input to the range [-500, 500] to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Define sigmoid derivative to be used in Error Backpropagation formula: error * sigmoid_derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Define softmax function for the output layer
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Define forward propagation function
def forward_propagation(fit, input_weights, hidden_weights1, hidden_weights2, dropout_rate, seed):
    # Calculate first hidden layer input and output
    hidden_layer1_input = np.dot(fit, input_weights)
    hidden_layer1_output = sigmoid(hidden_layer1_input)

    # Apply dropout in the first hidden layer
    hidden_layer1_output = dropout(hidden_layer1_output, dropout_rate)

    # Calculate second hidden layer input and output
    hidden_layer2_input = np.dot(hidden_layer1_output, hidden_weights1)
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    # Apply dropout in the second hidden layer
    hidden_layer2_output = dropout(hidden_layer2_output, dropout_rate)

    # Calculate output layer input and output
    output_layer_input = np.dot(hidden_layer2_output, hidden_weights2)
    output_layer_output = softmax(output_layer_input)

    return hidden_layer1_output, hidden_layer2_output, output_layer_output

# Define backpropagation function to update weights based on error
def backpropagation(inputs, input_weights, hidden_weights1, hidden_weights2, output_layer_output, outputs, hidden_layer1_output, hidden_layer2_output, dropout_rate):
    # Calculate output layer error and delta
    output_error = outputs - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    # Calculate second hidden layer error and delta
    hidden_error2 = output_delta.dot(hidden_weights2.T)
    hidden_delta2 = hidden_error2 * sigmoid_derivative(hidden_layer2_output)

    # Calculate first hidden layer error and delta
    hidden_error1 = hidden_delta2.dot(hidden_weights1.T)
    hidden_delta1 = hidden_error1 * sigmoid_derivative(hidden_layer1_output)

    # Update weights using the Error Backpropagation formula
    hidden_weights2 += hidden_layer2_output.T.dot(output_delta) * learning_rate
    hidden_weights1 += hidden_layer1_output.T.dot(hidden_delta2) * learning_rate
    input_weights += inputs.T.dot(hidden_delta1) * learning_rate

    return hidden_weights1, hidden_weights2, input_weights

# Define dropout function to apply dropout regularization during training
def dropout(x, dropout_rate):
    if dropout_rate > 0.0:
        mask = (np.random.rand(*x.shape) < dropout_rate) / (1 - dropout_rate)
        return x * mask
    else:
        return x

# Define training accuracy function to display accuracy during training
def training_Accuracy(epoch, bad_facts_count, total_examples, ax, update_frequency,seed):
    accuracy = ((total_examples - bad_facts_count) / total_examples) * 100
    print(f"\rTraining the Model | Epoch: {epoch + 1} | Bad Facts: {bad_facts_count} | Accuracy: {accuracy:.2f}%", end="", flush=True)

    # Update the plot with the total number of bad facts for the current epoch
    epoch_list.append(epoch + 1)
    bad_facts_list.append(bad_facts_count)
    if epoch % update_frequency == 0 or epoch == epochs - 1:
        ax.clear()
        ax.plot(epoch_list, bad_facts_list, 'bo-')
        ax.set_title(f'Bad Facts vs Epochs (Seed {seed})')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Bad Facts')
        plt.pause(0.1)  

# Define testing accuracy function to evaluate accuracy on unseen data
def testing_Accuracy(inputs, outputs, input_weights, hidden_weights1, hidden_weights2, dropout_rate, seed):
    correct_predictions = 0
    print("\nTesting using unseen data:\n")
    for example in range(len(inputs)):
        fit = inputs[example].reshape(1, -1)
        _, _, output_layer_output = forward_propagation(fit, input_weights, hidden_weights1, hidden_weights2, dropout_rate, seed)
        predicted_class = np.argmax(output_layer_output)
        true_class = np.argmax(outputs[example])

        is_correct = predicted_class == true_class

        # Print ID, expected outputs, predicted outputs, and correctness
        print(f"ID: {example + 1} | Expected: {outputs[example]} | Predicted: {output_layer_output[0]} | Correct: {is_correct}")

        if is_correct:
            correct_predictions += 1

    accuracy = (correct_predictions / len(inputs)) * 100
    print(f"\nTesting Accuracy: {accuracy:.2f}%")

# Load the training inputs
inputs = training_data.iloc[:, :-3].values
# Load the training answers
outputs = training_data.iloc[:, -3:].values

# Set hyperparameters
learning_rate = 0.2
error_threshold = 0.2
epochs = 5000
input_layer_size = 4
hidden_layer1_size = 4
hidden_layer2_size = 4
output_layer_size = 3
dropout_rate = 0
# For testing different seeds. Tested [1,10,21,23,30,91]
seed_range = [23]

# Set the update frequency for the graph 
update_frequency = 100

# Loop through different seeds
for seed in seed_range:
    print(f"\nUsing seed {seed}:")
    
    allFacts = []
    epoch_list = []
    bad_facts_list = []

    # Seed for reproducibility
    np.random.seed(seed)

    # Generate random weights for the input layer (4x4)
    input_weights = np.random.uniform(size=(input_layer_size, hidden_layer1_size))
    # Generate random weights for the first hidden layer (4x4)
    hidden_weights1 = np.random.uniform(size=(hidden_layer1_size, hidden_layer2_size))
    # Generate random weights for the second hidden layer (4x3)
    hidden_weights2 = np.random.uniform(size=(hidden_layer2_size, output_layer_size))

    # Initialize the plot outside the training loop
    plt.ion()
    fig, ax = plt.subplots()

    # Training loop
    for epoch in range(epochs):
        bad_facts_count = 0
        for example in range(len(inputs)):
            fit = inputs[example].reshape(1, -1)
            hidden_layer1_output, hidden_layer2_output, output_layer_output = forward_propagation(fit, input_weights, hidden_weights1, hidden_weights2, dropout_rate, seed)
            error = outputs[example] - output_layer_output

            # Update weights if error exceeds the threshold
            if any(abs(e) > error_threshold for e in error[0]):
                hidden_weights1, hidden_weights2, input_weights = backpropagation(fit, input_weights, hidden_weights1, hidden_weights2, output_layer_output, outputs[example], hidden_layer1_output, hidden_layer2_output, dropout_rate)
                bad_facts_count += 1
                status = "Bad Fact"  
            else:
                status = "Good Fact"

            fact = {
                'Epoch': epoch + 1,
                'Index': example + 1,
                'Expected Output': outputs[example],
                'Output': output_layer_output[0],  
                'Status': status
            }
            allFacts.append(fact)
        
        training_Accuracy(epoch, bad_facts_count, len(inputs), ax, update_frequency, seed)

        # Check if bad facts occurred, and break if not
        if bad_facts_count == 0:
            print("\nNo bad facts in the last epoch. Stopping training.")
            print(f"Found seed {seed} with 100% accuracy.")
            break

    # Convert the list of facts to a DataFrame
    facts_df = pd.DataFrame(allFacts)

    # Create the "facts" folder if it doesn't exist
    facts_folder = 'facts'
    os.makedirs(facts_folder, exist_ok=True)

    # Save the DataFrame as a CSV file inside the "facts" folder
    facts_filename = f'facts_seed_{seed}.csv'
    facts_df.to_csv(os.path.join(facts_folder, facts_filename), index=False)

    # Save weights after training
    with open(os.path.join(weights_folder, f'weights_seed_{seed}.pkl'), 'wb') as file:
        weights_dict = {'input_weights': input_weights, 'hidden_weights1': hidden_weights1, 'hidden_weights2': hidden_weights2}
        pickle.dump(weights_dict, file)

    # Evaluate testing accuracy
    testing_Accuracy(testing_data.iloc[:, :-3].values, testing_data.iloc[:, -3:].values, input_weights, hidden_weights1, hidden_weights2, dropout_rate, seed)

# Display the final plots for all seeds
plt.show(block=True)