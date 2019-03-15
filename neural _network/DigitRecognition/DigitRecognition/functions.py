import numpy as np
import random
import scipy
from PIL import Image

#Сигмоида
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z)), Z

#Преобразование изображения 10x10 в одномерный массив длиной в 100 элементов
def take_a_pic(im, rnd):
    img_temp = np.asarray(im)
    img = np.zeros((100))
    count = 0
    for i in img_temp:
        for j in i:
            img[count] = j
            count += 1
    answer = np.zeros((10))
    answer[rnd] = 1
    return img, answer

#Инициализация случайных весов и сдвигов
def initWB(layers_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(
            layers_dims[l], layers_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

#Функция записи параметров в файл
def inputWB(parameters):
    count = 1
    string = ''
    while np.all(parameters.get("W" + str(count))) != None:
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
def outputWB(layers_dims):
    parameters = {}
    count = 1
    L = len(layers_dims)
    for l in range(1, L):
        matrix_W = np.zeros((layers_dims[l], layers_dims[l-1]))
        matrix_b = np.zeros((layers_dims[l], 1))
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
#print(outputWB([10, 4, 2]))

'''
Feed Forward
'''

def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b):
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = sigmoid(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    A = X
    caches = []
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters["W" + str(l)], parameters["b" + str(l)])
        caches.append(cache)
    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)])
    caches.append(cache)
    return AL, caches



'''
Cross-Entropy cost
'''

def compute_cost(AL, y):
    m = y.shape[1]
    cost = - (1 / m) * np.sum(np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))
    return cost


'''
Back-Propagation
'''

