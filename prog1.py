import numpy as np
import random

# function to read in data and scale the data values
def preprocess(filename):
    data = np.genfromtxt(filename, delimiter=',', dtype=int, comments="#")
    labels = data[:, 0]
    N = np.shape(labels)[0]
    targets = []
    for a in labels:
        row = []
        for i in range(10):
            if a == i:
                row.append(0.9)
            else:
                row.append(0.1)
        targets.append(row)
    targets = np.array(targets)
    inputs = (1 / 255) * data
    inputs[:,0] = 1
    return targets, inputs

# sigmoid function for vector computation
def vectorized_sigmoid(m):
    return 1 / (1 + np.exp(-m))

# Print accuracies into a file 
def write_result(train_accuracy, test_accuracy):
    with open("output.txt", 'a') as f:
        f.write(str(train_accuracy))
        f.write("\n")
        f.write(str(test_accuracy))
        f.write("\n")
        f.close()

# compute the accuracy of the network with the frozen weights
def computeAccuracy(inputs, weights_in_hid, weights_hid_out, targets, confmat):
    # activation at the hidden layer
    activations_hid = np.dot(inputs, weights_in_hid)
    activations_hid = vectorized_sigmoid(activations_hid)
    
    # add the bias neuron to the hidden layer
    hid_inputs = np.concatenate((activations_hid, np.ones((np.shape(activations_hid)[0], 1))), axis=1)
    
    # activation at the output layer
    activations_out = np.dot(hid_inputs, weights_hid_out)
    activations_out = vectorized_sigmoid(activations_out)
    
    # count the count predictions made
    correct_pred = 0
    output = np.argmax(activations_out, axis=1)
    labels = np.argmax(targets, axis=1)
    for i in range(len(output)):
        if labels[i] == output[i]:
            correct_pred += 1
            
    # print the confusion matrix
    if confmat:
        confs = np.array([[0] * 10] * 10)
        for i in range(len(output)):
            confs[labels[i]][output[i]] += 1
        print(confs)
        print()
    return correct_pred / np.shape(inputs)[0] * 100

# function to train the network
def train(h_units, alpha, train_accuracy, test_accuracy):
    order = [i for i in range(Ntrain)]
    for epoch in range(50):
        global weights_in_hid
        global weights_hid_out
        global delta_weights_in_hid
        global delta_weights_hid_out
        
        random.shuffle(order)           # randomize the order of training examples used
        
        # for each training example, compute the activation, errors and update the weights
        for idx in order:
            # activation at the hidden layer
            activation = np.dot(train_inputs[idx], weights_in_hid)
            activation = vectorized_sigmoid(activation)
            
            # activation at the output layer
            hid_inputs = np.concatenate((activation, [1]))
            activation_out = np.dot(hid_inputs, weights_hid_out)
            activation_out = vectorized_sigmoid(activation_out)
            
            # calculate errors at the output and hidden layer
            output_error = activation_out * (([1] * np.shape(activation_out)[0]) - activation_out) * (train_target[idx] - activation_out)
            hidden_error = activation * (([1] * np.shape(activation)[0]) - activation) * (np.dot(weights_hid_out[:-1], output_error))
            
            # update the change in weights from the previous update  
            delta_weights_hid_out = 0.1 * hid_inputs.reshape(h_units + 1, 1) * output_error + alpha * delta_weights_hid_out
            delta_weights_in_hid = 0.1 * train_inputs[idx].reshape(785, 1) * hidden_error + alpha * delta_weights_in_hid
            
            # update the weights
            weights_hid_out += delta_weights_hid_out
            weights_in_hid += delta_weights_in_hid
        
        # keep track of the accuracies for each epoch
        train_accuracy.append(computeAccuracy(train_inputs, weights_in_hid, weights_hid_out, train_target, False))
        if epoch == 49:
            test_accuracy.append(computeAccuracy(test_inputs, weights_in_hid, weights_hid_out, test_target, True))
        else:
            test_accuracy.append(computeAccuracy(test_inputs, weights_in_hid, weights_hid_out, test_target, False))

# Set up the network with a number of hidden units, h_units, and momentum, alpha.
def network(h_units,alpha):
    train_accuracy = []
    test_accuracy = []
    
    # training and testing accuracies at epoch 0
    initial_accuracy = computeAccuracy(train_inputs, weights_in_hid, weights_hid_out, train_target, False)
    train_accuracy.append(initial_accuracy)
    initial_accuracy = computeAccuracy(test_inputs, weights_in_hid, weights_hid_out, test_target, False)
    test_accuracy.append(initial_accuracy)
   
    # train the network
    train(h_units, alpha, train_accuracy, test_accuracy)
    
    # output the results
    write_result(train_accuracy, test_accuracy)

# initialize the training inputs and targets
train_target, train_inputs = preprocess("mnist_train.csv")
Ntrain = np.shape(train_target)[0]
# initialize the training inputs and targets
test_target, test_inputs = preprocess("mnist_test.csv")
Ntest = np.shape(test_target)[0]

n = [20, 50, 100]           # hidden units
momentum = [0, 0.25, 0.5]   # momentum

print("Experiment 1") # Varying the number of hidden units
for h_units in n:
    # initialize weights for the hidden layer
    weights_in_hid = np.random.uniform(-0.05, 0.05, (785, h_units))
    # initialize weights for the output layer
    weights_hid_out = np.random.uniform(-0.05, 0.05, (h_units + 1, 10))
    # initialize the weights difference
    delta_weights_hid_out = np.zeros((h_units + 1, 10))
    delta_weights_in_hid = np.zeros((785, h_units))
    # initialize the network
    network(h_units, 0.9)


print("Experiment 2") # Varying the momentum of the network
for alpha in momentum:
    # initialize weights for the hidden layer
    weights_in_hid = np.random.uniform(-0.05, 0.05, (785, 100))
    # initialize weights for the output layer
    weights_hid_out = np.random.uniform(-0.05, 0.05, (100 + 1, 10))
    # initialize the weights difference
    delta_weights_hid_out = np.zeros((100 + 1, 10))
    delta_weights_in_hid = np.zeros((785, 100))
    # initialize the network
    network(100, alpha)

print("Experiment 3") # Varying the size of the training set
order = [i for i in range(Ntrain)]

# to ensure that the 10 classes are well represented in the reduced data set, we seperate the dataset. The reduced sets will be shuffled before training, so this is fine.
distr = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
for idx in order:
    label = np.argmax(train_target[idx])
    distr[label].append(train_inputs[idx])
network1 = []
network2 = []
for key in distr:
    network1 = network1 + distr[key][:3000]
    network2 = network2 + distr[key][:1500]
network1 = np.array(network1)
network2 = np.array(network2)
network1_targets = np.array([[0.1] * 10]* 30000)
network2_targets = np.array([[0.1] * 10]* 15000)
a = 0
for i in range(10):
    network1_targets[a:a+3000, i] = 0.9
    a = a + 3000
a = 0
for i in range(10):
    network2_targets[a:a+1500, i] = 0.9
    a += 1500


# initialize the dataset of 30,000 training examples and its weights
train_target = network1_targets
train_inputs = network1
Ntrain = np.shape(train_target)[0]
weights_in_hid = np.random.uniform(-0.05, 0.05, (785, 100))
weights_hid_out = np.random.uniform(-0.05, 0.05, (100 + 1, 10))
delta_weights_hid_out = np.zeros((100 + 1, 10))
delta_weights_in_hid = np.zeros((785, 100))
network(100,0.9)                # train the network


# initialize the dataset of 15,000 training examples and its weights
train_target = network2_targets
train_inputs = network2
Ntrain = np.shape(train_target)[0]
weights_in_hid = np.random.uniform(-0.05, 0.05,(785, 100))
weights_hid_out = np.random.uniform(-0.05, 0.05,(100+1,10))
delta_weights_hid_out = np.zeros((100+1,10))
delta_weights_in_hid = np.zeros((785, 100))
network(100,0.9)            # train the network


