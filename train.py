# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:15:07 2019

@author: Adrian
"""

### choo choo

import cv2
import os
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split
from model import unet

from tensorflow.keras.models import load_model


def pad_to_square(X, Y):
    
    i = np.argmin(X.shape[0:2])

    d = X.shape[1-i] - X.shape[i]
    
    return np.pad(X,tuple([(int(d/2), int(d/2)) if j==i else (0,0) for j in range(0,2)]), mode='constant', constant_values=np.nan)

def maskread(file):

    M = np.asarray(Image.open(file).convert('L'))
    
    Mv = list(set(M.flatten()))
    Mc = np.zeros((M.shape[0],M.shape[1],3), dtype=np.float32)
    
    for i, v in enumerate(Mv):
        Mc[M[:,:]==v,i] = 1
        
    return Mc

def data_generator(X_files, Y_files, batch_size = 2, size = (512, 512)):

    while True:
        
        X, Y = [], []
        
        inds = np.random.randint(len(X_files), size=batch_size)
        
        X_Frames = [F for i, F in enumerate(X_files) if i in inds]
        Y_Frames = [F for i, F in enumerate(Y_files) if i in inds]
        
        for (X_Frame, Y_Frame) in zip(X_Frames, Y_Frames):
            
            M = maskread(Y_Frame)
            
            I = cv2.imread(X_Frame, cv2.IMREAD_GRAYSCALE)

            Imax = np.iinfo(I.dtype).max
            I = np.float32(I)
            I /= Imax
            
            i = np.argmin(I.shape)
            d = I.shape[1-i] - I.shape[i]
            
            I = np.pad(I,tuple([(int(d/2), int(d/2)) if j==i else (0,0) for j in range(0,2)]), mode='constant', constant_values=np.mean(I[M[:,:,1]==1]))
            M = np.pad(M,tuple([(int(d/2), int(d/2)) if j==i else (0,0) for j in range(0,3)]), mode='constant', constant_values=np.nan)

            y, x = np.where(np.isnan(M[:,:,1]))
            M[y,x,1] = 1
            M[np.isnan(M)] = 0

            I = cv2.resize(I, size, interpolation=cv2.INTER_LINEAR)
            M = cv2.resize(M, size, interpolation=cv2.INTER_NEAREST)
            
            I -= I.mean()
            I /= I.std()

            # I = np.stack([I for i in range(3)],axis=-1)
            
            p = np.random.rand()
            
            if p < 0.25:
                I = cv2.flip(I, 0)
                M = cv2.flip(M, 0)
            elif p > 0.25 and p < 0.5:
                I = cv2.flip(I, 1)
                M = cv2.flip(M, 1)
            elif p > 0.5 and p < 0.75:
                I = cv2.flip(I, -1)
                M = cv2.flip(M, -1)
            elif p > 0.75:
                pass
                    
            I = np.reshape(I, I.shape+(1,))
            
            X.append(I)
            Y.append(M)

        yield (np.float16(np.asarray(X)), np.float16(np.asarray(Y)))
          


depth = 5

batch_size = 2
size = (512, 512)
epochs = 10
         
data_path = 'C:/Users/Adrian/Desktop/SlimeNet'
images = [File for File in os.listdir(os.path.join(data_path,'images')) if '.jpeg' in File or '.jpg' in File or '.tif' in File or '.png' in File]

X_files = sorted([os.path.join(data_path,'images',file) for file in images])
Y_files = sorted([os.path.join(data_path,'labels',file) for file in images])

X_train, X_test, Y_train, Y_test = train_test_split(X_files,Y_files,test_size=0.1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.1)

data_gen = data_generator(X_train, Y_train, batch_size=batch_size, size=size)
val_gen = data_generator(X_val, Y_val, batch_size=batch_size, size=size)
test_gen = data_generator(X_test, Y_test, batch_size=1, size=(512, 512))

model = load_model(os.path.join('C:/Users/Adrian/Desktop/SlimeNet','model512_depth5_epoch5.h5'))

# model = unet(depth=depth, size=(None,None,1), nc=3, nf=64, ks=3, bn=True, do=0.25, LR = 1e-4)

model.fit_generator(data_gen, steps_per_epoch = len(X_train)/batch_size, epochs=epochs, verbose = 1, validation_data=val_gen, validation_steps=len(X_val)/batch_size)
model.save(os.path.join('C:/Users/Adrian/Desktop/SlimeNet','model512_depth5_epoch15.h5'))

import matplotlib.pyplot as plt
plt.close()

I = model.predict_generator(test_gen, steps = 1)

plt.imshow(I[0,:,:,:], cmap='gray')


# pickle.dump(std, open(os.path.join('D:/Seafile/PhaseProject/segmentation','std512_binning2.pkl'), "wb" ))