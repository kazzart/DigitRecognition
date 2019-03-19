import numpy as np
#import matplotlib.pyplot as plt
import random
from PIL import Image
import os
#import scipy
import functions as fn
from RandomSample import Rand



def L_layer_model(
        X, y, layers_dims, learning_rate=0.01, num_iterations=3000,
        print_cost=True, hidden_layers_activation_fn="relu"):
    random.seed(version =2)
    np.random.seed(random.randint(0,1000))

    
    parameters = fn.initialize_parameters(layers_dims)

    
    cost_list = []

    
    for i in range(num_iterations):
    
        AL, caches = fn.L_model_forward(
            X, parameters, hidden_layers_activation_fn)

    
        cost = fn.compute_cost(AL, y)

    
        grads = fn.L_model_backward(AL, y, caches, hidden_layers_activation_fn)

    
        parameters = fn.update_parameters(parameters, grads, learning_rate)

    
        if (i + 1) % 100 == 0 and print_cost:
            print(f"The cost after {i + 1} iterations is: {cost:.4f}")

        if i % 100 == 0:
            cost_list.append(cost)

    
    #plt.figure(figsize=(10, 6))
    #plt.plot(cost_list)
    #plt.xlabel("Iterations (per hundreds)")
    #plt.ylabel("Loss")
    #plt.title(f"Loss curve for the learning rate = {learning_rate}")
    return parameters


def accuracy(X, parameters, y, q, activation_fn="relu"):
    probs, caches = fn.L_model_forward(X, parameters, activation_fn)
    probs = probs.T
    #print(probs)    
    for i in range(np.shape(probs)[0]):
        flag = True
        li = np.asarray(probs[i])
        Max = max(li)
               
        
        for j in range(np.shape(probs)[1]):
            if probs[i][j]==Max and flag:
                probs[i][j] = 1
                flag = False
            else:
                probs[i][j] = 0
                
       
    labels = probs
    y = y.T
    #print(labels, y)
    count = 0
    acc = {}
    
    for i in range(10):
        acc[i] = 0
    for i in range(np.shape(probs)[0]):
        flag = True
        for j in range(np.shape(probs)[1]):
            if labels[i][j] != y[i][j]:
                flag = False
                break
            elif labels[i][j] == 1:
                acc[j] +=1
        if flag:
            count +=1
    for i in range(10):
        acc[i] *= np.shape(probs)[0]/10
    accuracy = count
    print(acc)
    print()
    
    return f"The accuracy rate is: {accuracy:.2f}%."

mode = ""
mode = input("Введите 0, если хотите использовать старые веса, или же 1, если хотите расчитать их заного: ")
while (str(mode) != str(0)) and (str(mode)!= str(1)):
    mode = input("Повторите ввод: ")
s = Rand(60)
if s.notEmpty():
    image, num = s.next()
    X1, y1 = fn.take_a_pic(image, num)
    X_train = np.array([X1]).T
    y_train = np.array([y1]).T
    #print (np.shape(X), np.shape(y))
if mode == "1":
    while(s.notEmpty()):
        image, num = s.next()
        X1, y1 = fn.take_a_pic(image, num)
        X2 = np.array([X1]).T
        y2 = np.array([y1]).T
        X_train = np.concatenate((X_train, X2), axis = 1)
        y_train = np.concatenate((y_train, y2), axis = 1)
        #print (np.shape(X_train), np.shape(y_train))

coun = 0
coun2 = 1
q = 0
t = Rand(10, 'test')
if t.notEmpty():
    image, num = t.next()
    if num == 2:
        coun +=1
    X1, y1 = fn.take_a_pic(image, num)
    X_test = np.array([X1]).T
    y_test = np.array([y1]).T
    #print (np.shape(X), np.shape(y))

while(t.notEmpty()):
    coun2 += 1
    image, num = t.next()
    if num == 2:
        coun+= 1
    if coun == 6:
        q = coun2
    X1, y1 = fn.take_a_pic(image, num)
    X2 = np.array([X1]).T
    y2 = np.array([y1]).T
    X_test = np.concatenate((X_test, X2), axis = 1)
    y_test = np.concatenate((y_test, y2), axis = 1)
    #print (np.shape(X_test), np.shape(y_test))

X_train /= 255
X_test /= 255

layers_dims = [X_train.shape[0], 50, 10]
if mode == "1":
    parameters = L_layer_model( X_train, y_train, layers_dims, learning_rate=0.05, num_iterations=9000, hidden_layers_activation_fn="tanh")
    fn.input_parameters(parameters)

print (accuracy(X_test, fn.outputWB(layers_dims), y_test, q))
print()

#Подсчёт ответа на конкретном файле
print("Доступные файлы:")
dirName = os.path.join(os.path.realpath(r"..\..\..\ "),"samples","working")
names = os.listdir(dirName)
for name in names:
    fullname = os.path.join(dirName, name)
    if os.path.isfile(fullname) and name.split('.')[1] == "jpg":
        print(name)
print()
path = input("Введите название файла в папке working, включая расширение, введите \" exit \" для завершения : ")
print()
while(str(path)!="exit"):
    path = os.path.join(dirName,path)
    try:
        im = Image.open(path)
        img_temp = np.asarray(im)
        img = np.zeros((100))
        #print (img_temp, rnd)
        count = 0
        for i in img_temp:
            for j in i:
                img[count] = j
                count += 1

        img = np.concatenate((np.array([img]).T, np.array([img]).T), axis = 1)
        #print (img)
        X_test = img

        X_test /= 255




        #print ((img/255).T)
        probs, caches = fn.L_model_forward(X_test, fn.outputWB(layers_dims), "relu")
        print("answer:")
        #print(probs)
        ans = np.asarray(probs.T[0])
        Max = max(ans)
        for i in range(len(ans)):
            if np.all(ans[i] == Max):
                print(i)
                break;
        #print (probs.T)
    except IOError:
        print("Can't open image")
    print()
    print("Доступные файлы:")
    names = os.listdir(dirName)
    for name in names:
        fullname = os.path.join(dirName, name)
        if os.path.isfile(fullname) and name.split('.')[1] == "jpg":
            print(name)
    print()
    path = input("Введите название файла в папке working, включая расширение, введите \" exit \" для завершения : ")
    print()



    