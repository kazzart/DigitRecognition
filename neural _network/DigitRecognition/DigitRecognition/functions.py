import numpy as np
import random
#import scipy
from PIL import Image
import os

'''
Функции активации
'''

#Сигмоида
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

#Гиперболический тангенс
def tanh(Z):
    A = np.tanh(Z)
    return A, Z

#ReLU
def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

#Leaky ReLU
def leaky_relu(Z):
    A = np.maximum(0.1 * Z, Z)
    return A, Z


'''
Работа с файлами
'''

#Преобразование изображения 10x10 в одномерный массив длиной в 100 элементов
def take_a_pic(im, rnd):
    img_temp = np.asarray(im)
    img = np.zeros((100))
    #print (img_temp, rnd)
    count = 0
    for i in img_temp:
        for j in i:
            img[count] = j
            count += 1

    answer = np.zeros((10))
    answer[rnd] = 1
    return img, answer

#Инициализация случайных весов и сдвигов
def initialize_parameters(layers_dims):
    random.seed(version =2)
    np.random.seed(random.randint(0,1000))              
    parameters = {}
    L = len(layers_dims)            


    
    for l in range(1, L):
   
        parameters["W" + str(l)] = np.random.randn(
        layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))


    assert parameters["W" + str(l)].shape == (
    layers_dims[l], layers_dims[l - 1])
    assert parameters["b" + str(l)].shape == (layers_dims[l], 1)
 

    return parameters

#Функция записи параметров в файл
def input_parameters(parameters,path=""):
    count = 1
    string = ''
    while np.all(parameters.get("W" + str(count))) != None:
        if path != "":
            file = open(os.path.join(os.path.realpath(path),'W' + str(count) + '.txt'), 'w')
        else:
            file = open('W' + str(count) + '.txt', 'w')
        for i in parameters["W" + str(count)]:
            for j in i:
                string += str(j) + ' '
            file.write(string + '\n')
            string = ''
        file.close()
        count += 1
    count = 1
    string = ''
    while np.all(parameters.get("b" + str(count))) != None:
        if path != "":
            file = open(os.path.join(os.path.realpath(path),'b' + str(count) + '.txt'), 'w')
        else:
            file = open('b' + str(count) + '.txt', 'w')
        for i in parameters["b" + str(count)]:
            for j in i:
                string += str(j) + ' '
            file.write(string + '\n')
            string = ''
        file.close()
        count += 1
    return 0
#print(inputWB(initWB([10, 4, 2]))) #пример использования последних двух функций

#Функция считывания параметров из файла
def outputWB(layers_dims,path=""):
    parameters = {}
    count = 1
    L = len(layers_dims)
    for l in range(1, L):
        matrix_W = np.zeros((layers_dims[l], layers_dims[l-1]))
        matrix_b = np.zeros((layers_dims[l], 1))
        if path != "":
            file_W = open(os.path.join(os.path.realpath(path),'W' + str(count) + '.txt'), 'r')
            file_b = open(os.path.join(os.path.realpath(path),'b' + str(count) + '.txt'), 'r')
        else:
            file_W = open('W' + str(count) + '.txt', 'r')
            file_b = open('b' + str(count) + '.txt', 'r')
        for i in range(layers_dims[l]):
            matrix_W[i] = file_W.readline().strip().split(' ')
            matrix_b[i] = file_b.readline().strip().split(' ')
        file_W.close()
        file_b.close()
        parameters["W" + str(count)] = matrix_W
        parameters["b" + str(count)] = matrix_b
        count += 1    
    return parameters

'''
Feed Forward
'''

def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation_fn):
    assert activation_fn == "sigmoid" or activation_fn == "tanh" or \
        activation_fn == "relu"

    if activation_fn == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation_fn == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    elif activation_fn == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    #assert A.shape == (W.shape[0], A_prev.shape[1])

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters, hidden_layers_activation_fn="relu"):
    A = X                           
    caches = []                     
    L = len(parameters) // 2        

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
            activation_fn=hidden_layers_activation_fn)
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)],
        activation_fn="sigmoid")
    caches.append(cache)

    #assert AL.shape == (10, X.shape[1])
    return AL, caches


'''
Cross-Entropy cost
'''

def compute_cost(AL, y):
    m = y.shape[1]              
    cost = - (1 / m) * np.sum(
        np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))
    return cost


'''
Back-Propagation
'''

def sigmoid_gradient(dA, Z):
    A, Z = sigmoid(Z)
    dZ = dA * A * (1 - A)

    return dZ


def tanh_gradient(dA, Z):
    A, Z = tanh(Z)
    dZ = dA * (1 - np.square(A))

    return dZ


def relu_gradient(dA, Z):
    A, Z = relu(Z)
    dZ = np.multiply(dA, np.int64(A > 0))

    return dZ



def linear_backword(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation_fn):
    linear_cache, activation_cache = cache

    if activation_fn == "sigmoid":
        dZ = sigmoid_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    elif activation_fn == "tanh":
        dZ = tanh_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    elif activation_fn == "relu":
        dZ = relu_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, y, caches, hidden_layers_activation_fn="relu"):
    
    L = len(caches)
    grads = {}

    dAL = np.divide(AL - y, np.multiply(AL, 1 - AL))

    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads[
        "db" + str(L)] = linear_activation_backward(
            dAL, caches[L - 1], "sigmoid")

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads[
            "db" + str(l)] = linear_activation_backward(
                grads["dA" + str(l)], current_cache,
                hidden_layers_activation_fn)

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters[
            "W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters[
            "b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters