import numpy as np
import time

training_set_inputs = np.array([[0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1]])
training_set_outputs = np.array([[0,1]])

np.random.seed(1)

successes = 0
total = 0

def test_ml():
    global training_set_inputs, training_set_outputs, weights, successes, total
    synaptic_weights = 2 * np.random.random((8, 1)) - 1
    prev_synaptic_weights = np.copy(synaptic_weights)
    print("Enter case values:\n")
    ind = 0
    input_array = [0,0,0,0,0,0,0,0]
    while ind < 8:
        input_array[ind] = np.random.choice([0,1])
        ind += 1
    for iteration in range(2000):
        output = 1 / (1 + np.exp(-(np.dot(training_set_inputs, synaptic_weights))))
        synaptic_weights += np.dot(training_set_inputs.T, (training_set_outputs.T - output) * output * (1 - output))
    print ("I predict: " + str(1 / (1 + np.exp(-(np.dot(np.array(input_array), synaptic_weights))))))
    result = 1 / (1 + np.exp(-(np.dot(np.array(input_array), synaptic_weights))))
    '''actual = int(input("What should the result have been?"))'''
    '''feedback = input("Was I close? Y / N\n")'''
    print(int(round(result[0],0)))
    success_value = (input_array[2] == 1) or (input_array[4] == 1) or ((input_array[5] == 1) and (input_array[6] == 1))
    if not (((int(round(result[0])) == 1) and (success_value)) or ((int(round(result[0])) == 0) and (not success_value))):
        print('called it incorrectly\n')
        training_set_inputs = np.append(training_set_inputs, [input_array], axis=0)
        training_set_outputs = np.append(training_set_outputs, np.array([[1-int(round(result[0],0))]]),axis=1)
        print('Training sets updated')
        print(training_set_inputs)
        print(training_set_outputs)
        '''synaptic_weights /= np.ndarray.max(np.absolute(synaptic_weights))'''
    else:
        print('called it correctly\n')
        successes += 1
        '''synaptic_weights = np.copy(prev_synaptic_weights)'''
    total += 1
    print(synaptic_weights)
    print('% success is ' + str(int(100*successes/total)) + '%')
    time.sleep(1)
    test_ml()


test_ml()
