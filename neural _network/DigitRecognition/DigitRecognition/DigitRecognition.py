import numpy as np
import random
import scipy
from PIL import Image

#Сигмоида
def nonlin (x, deriv = False):
    if deriv:
        return nonlin(x) * (1 - nonlin(x)) #Производная сигмоиды
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

