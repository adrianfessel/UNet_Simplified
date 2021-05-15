import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split
from model import unet

from tensorflow.keras.models import load_model

def maskread(file, binning=None):
    """
    Function to read and resize label/mask data. Resizing binning is performed
    using nearest neighbor interpolation to preserve discrete label values.
    
    Currently fixed to three different labels
    
    Parameters
    ----------
        file : string
            system path to mask file
        binning : int or tuple or None 
            binning of input data. If int, original size is reduced by factor binning,
            if tuple, resize to size binning
        
    Returns
    -------
        M : numpy array
            third dimension equals number of different labels (currently 3)
    
    """

    M = np.asarray(Image.open(file,mode='r').convert('L'))

    if binning:

        if isinstance(binning, tuple):
            size = binning
        else:
            size = (np.int(M.shape[1]/binning), np.int(M.shape[0]/binning))

        M = cv2.resize(M, size, interpolation=cv2.INTER_NEAREST)


    Mv = list(np.unique(M))
    Mc = np.zeros((M.shape[0], M.shape[1], len(Mv)), dtype=np.float32)

    for i, v in enumerate(Mv):
        Mc[M[:, :] == v, i] = 1

    return Mc


def data_generator(X_files, Y_files, batch_size=2, size=(512, 512)):
    """
    Data generator to obtain resized batches of input images and labels. 
    Images are padded to square shape and resized to specified size. Padding
    values are selected according to mask background.
    Images and labels are are randomly flipped along one or two axes for
    data augmentation.
    
    Parameters
    ----------
        X_files : list
            input data paths
        Y_files : list
            input label paths
        batch_size : int
            output batch size
        size : (int, int)
            output image size
            
            
    Returns
    -------
        output : tuple of float16 arrays
            output batch
            
    """      

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

            I = np.pad(I, tuple([(int(d/2), int(d/2)) if j == i else (0, 0) 
                                 for j in range(0, 2)]), mode='constant', constant_values=np.mean(I[M[:, :, 1] == 1]))
            M = np.pad(M, tuple([(int(d/2), int(d/2)) if j == i else (0, 0)
                                 for j in range(0, 3)]), mode='constant', constant_values=np.nan)

            y, x = np.where(np.isnan(M[:, :, 1]))
            M[y, x, 1] = 1
            M[np.isnan(M)] = 0

            I = cv2.resize(I, size, interpolation=cv2.INTER_LINEAR)
            M = cv2.resize(M, size, interpolation=cv2.INTER_NEAREST)

            I -= I.mean()
            I /= I.std()

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


if __name__ == '__main__':

    # training parameters
    depth = 4
    batch_size = 2
    size = (256, 256)
    epochs = 5
    
    # Paths
    Path_Data = 'E:\Seadrive\Adrian F\Meine Bibliotheken\Phasenwellen-Projekt\codes_unsorted\SlimeNet'
    Path_Model = os.path.join('C:/Users/Adrian/Desktop', 'model{}_depth{}_epoch{}.h5',format(size[0], depth, epochs))
    
    # training data
    images = [File for File in os.listdir(os.path.join(
        Path_Data, 'images')) if '.jpeg' in File or '.jpg' in File or '.tif' in File or '.png' in File]
    
    X_files = sorted([os.path.join(Path_Data, 'images', file) for file in images])
    Y_files = sorted([os.path.join(Path_Data, 'labels', file) for file in images])
    
    # train / test / validation split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_files, Y_files, test_size=0.1)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.1)
    
    # data generators
    data_gen = data_generator(X_train, Y_train, batch_size=batch_size, size=size)
    val_gen = data_generator(X_val, Y_val, batch_size=batch_size, size=size)
    test_gen = data_generator(X_test, Y_test, batch_size=1, size=(512, 512))

    # generate new unet model (& specify some hyperparameters)
    model = unet(depth=depth, size=(None,None,1), nc=3, nf=64, ks=3, bn=True, do=0.25, LR = 1e-4)
    
    # optional: load pretrained model for transfer learning or to continue training
    # model = load_model(os.path.join(
    #     'C:/Users/Adrian/Desktop/SlimeNet', 'model512_depth5_epoch5.h5'))
    
    # train unet model
    model.fit(data_gen, steps_per_epoch=len(X_train)/batch_size, epochs=epochs,
                        verbose=1, validation_data=val_gen, validation_steps=len(X_val)/batch_size)
    
    # save model
    model.save(Path_Model)
    
    # optional: segment & show a random single example from the test set
    # plt.close()
    # I = model.predict(test_gen, steps=1)
    # plt.imshow(I[0, :, :, :], cmap='gray')
