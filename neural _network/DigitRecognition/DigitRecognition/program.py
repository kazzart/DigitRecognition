import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
import functions as fn
from RandomSample import Rand

def L_layer_model(
        X, y, layers_dims, learning_rate = 0.01, num_iterations = 3000, 
        print_cost = True):
    np.random.seed(1)
    parameters = fn.init_parameters(layers_dims)
    cost_list = []

    for i in range(num_iterations):
        AL, caches = fn.L_model_forward(X, parameters)
        cost = fn.compute_cost(AL,y)
        grads = fn.L_model_backward(AL, y, caches)
        parameters = fn.update_parameters(parameters, grads, learning_rate)
        if (i + 1) % 100 == 0 and print_cost:
            print(f"The cost after {i + 1} iterations is: {cost:.4f}")
        if i % 100 == 0:
            cost_list.append(cost)
    plt.figure(figsize = (10, 6))
    plt.plot(cost_list)
    plt.xlabel("Iterations (per hundreds)")
    plt.ylabel("Loss")
    plt.title(f"Loss curve for the learning rate = {learning_rate}")
    return parameters

def accuracy(X, parameters, y):
    probs, caches = fn.L_model_forward(X, parameters)
    labels = (probs >= 0.5) * 1
    accuracy = np.mean(labels == y) * 100
    return f"The accuracy rate is: {accuracy:.2f}%."



s = Rand(10)
if s.notEmpty():
    image, num = s.next()
    X1, y1 = fn.take_a_pic(image, num)
    X = np.array([X1]).T
    y = np.array([y1]).T
    #print (np.shape(X), np.shape(y))

while(s.notEmpty()):
    image, num = s.next()
    X1, y1 = fn.take_a_pic(image, num)
    X2 = np.array([X1]).T
    y2 = np.array([y1]).T
    X_train = np.concatenate((X, X2), axis = 1)
    y_train = np.concatenate((y, y2), axis = 1)
print (np.shape(X_train), np.shape(y_train))


layers_dims = [X_train.shape[0], 30, 10]
parameters = L_layer_model(
    X_train, y_train, layers_dims,
    learning_rate = 0.03, num_iterations = 3000)
fn.input_parameters(parameters)

accuracy(X_train, parameters, y_train)

    