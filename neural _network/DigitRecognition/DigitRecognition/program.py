import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
import functions as fn
from RandomSample import Rand


# Define the multi-layer model using all the helper functions we wrote before
def L_layer_model(
        X, y, layers_dims, learning_rate=0.01, num_iterations=3000,
        print_cost=True, hidden_layers_activation_fn="relu"):
    np.random.seed(1)

    # initialize parameters
    parameters = fn.initialize_parameters(layers_dims)

    # intialize cost list
    cost_list = []

    # iterate over num_iterations
    for i in range(num_iterations):
        # iterate over L-layers to get the final output and the cache
        AL, caches = fn.L_model_forward(
            X, parameters, hidden_layers_activation_fn)

        # compute cost to plot it
        cost = fn.compute_cost(AL, y)

        # iterate over L-layers backward to get gradients
        grads = fn.L_model_backward(AL, y, caches, hidden_layers_activation_fn)

        # update parameters
        parameters = fn.update_parameters(parameters, grads, learning_rate)

        # append each 100th cost to the cost list
        if (i + 1) % 100 == 0 and print_cost:
            print(f"The cost after {i + 1} iterations is: {cost:.4f}")

        if i % 100 == 0:
            cost_list.append(cost)

    # plot the cost curve
    plt.figure(figsize=(10, 6))
    plt.plot(cost_list)
    plt.xlabel("Iterations (per hundreds)")
    plt.ylabel("Loss")
    plt.title(f"Loss curve for the learning rate = {learning_rate}")

    return parameters


def accuracy(X, parameters, y, activation_fn="relu"):
    probs, caches = fn.L_model_forward(X, parameters, activation_fn)
    labels = (probs >= 0.5) * 1
    accuracy = np.mean(labels == y) * 100
    return f"The accuracy rate is: {accuracy:.2f}%."



s = Rand(60)
if s.notEmpty():
    image, num = s.next()
    X1, y1 = fn.take_a_pic(image, num)
    X_train = np.array([X1]).T
    y_train = np.array([y1]).T
    #print (np.shape(X), np.shape(y))

while(s.notEmpty()):
    image, num = s.next()
    X1, y1 = fn.take_a_pic(image, num)
    X2 = np.array([X1]).T
    y2 = np.array([y1]).T
    X_train = np.concatenate((X_train, X2), axis = 1)
    y_train = np.concatenate((y_train, y2), axis = 1)
    print (np.shape(X_train), np.shape(y_train))

t = Rand(10, 'test')
if t.notEmpty():
    image, num = t.next()
    X1, y1 = fn.take_a_pic(image, num)
    X_test = np.array([X1]).T
    y_test = np.array([y1]).T
    #print (np.shape(X), np.shape(y))

while(t.notEmpty()):
    image, num = t.next()
    X1, y1 = fn.take_a_pic(image, num)
    X2 = np.array([X1]).T
    y2 = np.array([y1]).T
    X_test = np.concatenate((X_test, X2), axis = 1)
    y_test = np.concatenate((y_test, y2), axis = 1)
    print (np.shape(X_test), np.shape(y_test))

X_train /= 255
X_test /= 255

#print (X_train/255)

layers_dims = [X_train.shape[0], 12, 12, 10]
parameters = L_layer_model( X_train, y_train, layers_dims, learning_rate=0.03, num_iterations=3000, hidden_layers_activation_fn="tanh")
fn.input_parameters(parameters)

print (accuracy(X_test, parameters, y_test))



    