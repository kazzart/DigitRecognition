import random as r
import os
from PIL import Image
class Rand:
    used = []
    #инициализация, quantity - количество примеров
    def __init__(self,quantity):
        r.seed(version =2)
        for i in range(10):
            li = []
            for i in range(quantity):
                li.append(i+1)
            self.used.append(li)
    #проверка, не осталось ли неиспользованных примеров
    def notEmpty(self):
        for line in self.used:
            if len(line) > 0:
                return True
        return False
    #так, хуйня для дебага
    def show(self):
        for i in self.used:
            print(i)
    #выдаёт следующее рандомное изображение
    def next(self):
        li = self.used
        a = r.randint(0,len(li)-1)
        while len(li[a]) == 0:
            a = r.randint(0,len(li)-1)
        #print(str(a) + " , " + str(li[a]))
        li2 = li[a]
        im = Image.open(os.path.realpath(r"..\..\..\samples\learn\{}\test_{}_{}.jpg".format(a, a, li2[0])))
        li2.remove(li2[0])
        li[a] = li2
        self.used = li
        return im,a

'''
s = Rand(1)
while(s.notEmpty()):
    image,num = s.next()
    image.show()
'''
