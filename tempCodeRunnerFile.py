def testing_Accuracy(inputs, outputs, input_weights, hidden_weights):
    correct_predictions = 0

    for example in range(len(inputs)):
        fit = inputs[example].reshape(1, -1)
        _, output_layer_output = forward_propagation(fit, input_weights, hidden_weights, dropout_rate)
        predicted_class = np.argmax(output_layer_output)
        true_class = np.argmax(outputs[example])

        is_correct = predicted_class == true_class

        # Print id, expected outputs, predicted outputs, and correctness
        print(f"ID: {example + 1} | Expected: {outputs[example]} | Predicted: {output_layer_output[0]} | Correct: {is_correct}")

        if is_correct:
            correct_predictions += 1

    accuracy = (correct_predictions / len(inputs)) * 100
    print(f"\nTesting Accuracy: {accuracy:.2f}%")