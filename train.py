import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
import os
import pandas as pd
import sys
import os
import dlib
import glob
import cv2
import re
import numpy as np
from rotate import pickle_loader
from pylab import imshow, show
from os import path
import pickle


var1,var2,labels=pickle_loader()

var1=var1/255
var2=var2/255

var1=np.stack(var1)
var2=np.stack(var2)


print(var1.shape[0])
print(var2.shape[0])

var1=np.reshape(var1,(var1.shape[0],114,40,40,1))
var2=np.reshape(var2,(var2.shape[0],114,40,40,3))

print("\nAfter\n")

print(var1.shape)
print(var2.shape)


from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils

print(labels)

labels=np_utils.to_categorical(labels)
print("AFTERRRRR.............")
print(labels)

epochs=100
from sklearn.model_selection import train_test_split

Xtrain1, Xtest1,Ytrain1,Ytest1 = train_test_split(var1,labels,test_size=0.2)
Xtrain2, Xtest2,Ytrain2,Ytest2 = train_test_split(var2,labels, test_size=0.2)

print(Xtrain1)
print(Xtrain1.shape)

#architecture
from model import *

model1=CNN_LSTM((114,40,40,1))

from tensorflow.keras.callbacks import ModelCheckpoint
#saving the model
checkpoint=ModelCheckpoint("Trained_Model/new_model_final.h5",monitor="accuracy",save_best_only=True,verbose=1)

history=model1.fit(Xtrain1,Ytrain1,epochs=epochs,batch_size=2,callbacks=[checkpoint])

#plotting
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Accuracy plot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'train_loss'], loc='upper left')
plt.savefig('Project_Extra/new_plot_final.png')
plt.show()