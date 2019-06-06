import numpy as np

training_set_inputs = np.array([[0, 0, 0, 0, 0],[1, 1, 1, 1, 1]])
training_set_outputs = np.array([[0,1]])

np.random.seed(1)
synaptic_weights = 2 * np.random.random((5, 1)) - 1

def test_ml():
    global synaptic_weights, training_set_inputs, training_set_outputs
    prev_synaptic_weights = np.copy(synaptic_weights)
    print("Enter case values:\n")
    ind = 0
    input_array = [0,0,0,0,0]
    while ind < 5:
        pos = input("value " + str(ind + 1) + ": ")
        input_array[ind] = int(pos)
        ind += 1
    for iteration in range(2000):
        output = 1 / (1 + np.exp(-(np.dot(training_set_inputs, synaptic_weights))))
        synaptic_weights += np.dot(training_set_inputs.T, (training_set_outputs.T - output) * output * (1 - output))
    print ("I predict: " + str(1 / (1 + np.exp(-(np.dot(np.array(input_array), synaptic_weights))))))
    actual = int(input("What should the result have been?"))
    feedback = input("Was I close? Y / N")
    if (feedback == 'N'):
        print(training_set_inputs)
        print(training_set_outputs)
        training_set_inputs = np.append(training_set_inputs, [input_array], axis=0)
        training_set_outputs = np.append(training_set_outputs, np.array([[actual]]),axis=1)
        print('Training sets updated')
        print(training_set_inputs)
        print(training_set_outputs)
        synaptic_weights /= np.ndarray.max(np.absolute(synaptic_weights))
    else:
        synaptic_weights = np.copy(prev_synaptic_weights)
    print(synaptic_weights)
    test_ml()


test_ml()
