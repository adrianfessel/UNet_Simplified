import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.keras.models import load_model   

class segmentation():
    
    def __init__(self, Path, Parameters, model):
        """
        Class for multilevel semantic segmentation with a pretrained unet.
        Provided grayscale images are resized to a uniform size during
        segmentation and results are restored to their original size via
        linear upsampling afterwards. Segmentation results are of shape
        (x_dim, y_dim, n), where n is the number of levels distinguished
        by the segmentation. Each pixel givel carries information on the 
        probability of belonging to any of the n levels.
        Intended for iterating over a directory
        of grayscale images. Upon instantiation, a folder for the output data
        is generated in the parent directory to the input path.
        

        Parameters
        ----------
        Path : string
            Directory containing input data.
        Parameters : dict
            dictionary of segmentation parameters. Possible parameters are
        'first_frame' : int
            number of first frame
        'last_frame' : int
            number of last frame
        'increment' : int
            process only every ith frame
        'size' : (int, int)
            uniform size images are resized to during segmentation
        model : tensorflow/keras model
            pretrained unet model

        Returns
        -------
        None.

        """
        
        self.Path = Path
        self.Path_Output = Path + '_Probability'
        
        if not os.path.isdir(self.Path_Output):
            os.mkdir(self.Path_Output)
        
        self.model = model

        
        self.Parameters = Parameters
        
        self.Parameters['first_frame'] = None if 'first_frame' not in self.Parameters else self.Parameters['first_frame']
        self.Parameters['last_frame'] = None if 'last_frame' not in self.Parameters else self.Parameters['last_frame']
        self.Parameters['increment'] = None if 'increment' not in self.Parameters else self.Parameters['increment']
        
        self.Parameters['size'] = (512, 512) if 'size' not in self.Parameters else self.Parameters['size']
        
        Frames = [Frame for Frame in os.listdir(os.path.join(self.Path)) if '.jpeg' in Frame or '.jpg' in Frame or '.tif' in Frame or '.png' in Frame]
        Frames = sorted(Frames)
        
        self.Frames = Frames[self.Parameters['first_frame']:self.Parameters['last_frame']:self.Parameters['increment']]
                

    def imread(self, Frame):
        """
        Function for reading and equalizing gray-scale images.

        Parameters
        ----------
        Frame : string
            name of the current image file

        Returns
        -------
        I : numpy array
            equalized image

        """
        
        Path = os.path.join(self.Path, Frame)
        
        I = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
        
        Imax = np.iinfo(I.dtype).max
        I = np.float32(I)
        I /= Imax
                
        I -= I.mean()
        I /= I.std()
        
        return I


    def segment_frame(self, Frame):
        """
        Function to segment a single gray-scale image.

        Parameters
        ----------
        Frame : string
            name of the current image file

        Returns
        -------
        R : numpy array
            segmented image

        """

        I = self.imread(Frame)

        size0 = I.shape
        
        i = np.argmin(size0)
        d = I.shape[1-i] - I.shape[i]
        
        I = np.pad(I,tuple([(int(d/2), int(d/2)) if j==i else (0,0) for j in range(0,2)]), mode='constant', constant_values=0)
        
        size1 = I.shape
        
        I = cv2.resize(I, self.Parameters['size'], interpolation=cv2.INTER_LINEAR)

        R = np.squeeze(np.array(self.model.predict_on_batch(np.float16(np.reshape(I,(1,)+I.shape+(1,))))))
        R = cv2.resize(R, size1, interpolation=cv2.INTER_LINEAR)
        
        if i == 0:
            R = R[int(d/2):-(int(d/2)), :, :]
        if i == 1:
            R = R[:, int(d/2):-(int(d/2)), :]

        return R
    
    
    def segment_and_show(self, Frame):
        """
        Function to segment a single image & display an overlay of the original
        data and the segmentation result

        Parameters
        ----------
        Frame : string
            name of the current image file

        Returns
        -------
        fig : handle
            handle to the figure

        """
        
        I = self.imread(Frame)
        R = self.segment_frame(Frame)
        
        
        fig = plt.figure(figsize=plt.figaspect(1/3)) 
        
        gs = gridspec.GridSpec(1, 3)
        
        plt.subplot(gs[0, 0])
        plt.imshow(I, cmap='gray')
        plt.axis('off')
        
        plt.subplot(gs[0, 1])
        plt.imshow(I, cmap='gray')
        plt.imshow(R, alpha=0.25)
        plt.axis('off')
        
        plt.subplot(gs[0, 2])
        plt.imshow(I, cmap='gray')
        plt.imshow(R)
        plt.axis('off')
        
        plt.tight_layout()
        
        return fig
    

    def run(self):
        """
        Function to run segmentation over a set of images.

        Returns
        -------
        None.

        """
        
        for Frame in tqdm(self.Frames):

            p = self.segment_frame(Frame)
    
            Img = Image.fromarray(np.uint8(p*255))
            Img.save(os.path.join(self.Path_Output,Frame.split('.')[0]+'.png'))
    
 
if __name__ == '__main__':

    # paths
    Path_Data = 'E:\Seadrive\Adrian F\Meine Bibliotheken\Phasenwellen-Projekt\codes_unsorted\SlimeNet\images'
    Path_Model = os.path.join('C:/Users/Adrian/Desktop', 'model512_depth4_epoch5.h5')

    # load pretrained model
    model = load_model(Path_Model)

    # specify segmentation parameters
    Parameters = {'first_frame':None,\
                  'last_frame':None,\
                  'increment':None,\
                  'size':(256, 256)
                    }
        
    # instantiate segmentation object
    obj = segmentation(Path_Data, Parameters, model)
    
    # run segmentation
    # obj.run()
    
    # show random examples
    frame = os.path.join(obj.Path, np.random.choice(obj.Frames))
    fig = obj.segment_and_show(frame)