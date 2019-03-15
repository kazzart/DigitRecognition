import numpy as np
import random
import scipy
from PIL import Image

#Сигмоида
def sigmoid(x, deriv = False):
    if deriv:
        return sigmoid(x) * (1 - sigmoid(x)) #Производная сигмоиды
    return 1 / (1 + np.exp(-x))

#Преобразование изображения 10x10 в одномерный массив длиной в 100 элементов
def take_a_pic():
    rndFolder, rndSample = someFunc() #Нужно доделать функцию рандомайзера выбора файла с отсеиванием использованных файлов
    im = Image.open("test_{}_{}.jpg".format(rndrndFolder, rndSample)) #Нужно разобраться как указывать путь относительно корневой папки а не диска Д
    im.show()
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

#print(initWB([10, 4, 2]))
print(inputWB(initWB([10, 4, 2])))