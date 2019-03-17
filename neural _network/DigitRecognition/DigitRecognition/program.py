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
        #print (AL)
        cost = fn.compute_cost(AL, y)
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
    image, num = s.next()
    X1, y1 = fn.take_a_pic(image, num)
    X_test = np.array([X1]).T
    y_test = np.array([y1]).T
    #print (np.shape(X), np.shape(y))

while(s.notEmpty()):
    image, num = s.next()
    X1, y1 = fn.take_a_pic(image, num)
    X2 = np.array([X1]).T
    y2 = np.array([y1]).T
    X_test = np.concatenate((X_test, X2), axis = 1)
    y_test = np.concatenate((y_test, y2), axis = 1)
    print (np.shape(X_test), np.shape(y_test))

#print (X_train/255)

layers_dims = [X_train.shape[0], 30, 10]
parameters = L_layer_model(
    X_train/255, y_train, layers_dims,
    learning_rate = 0.03, num_iterations = 3000)
fn.input_parameters(parameters)

print (accuracy(X_test/255, parameters, y_test))

    