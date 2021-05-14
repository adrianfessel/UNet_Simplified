
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

class segmentation():
    
    def __init__(self, Path, Parameters, model):
        
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
                

    def imread(self,Frame):
        Path = os.path.join(self.Path, Frame)
        
        I = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
        
        Imax = np.iinfo(I.dtype).max
        I = np.float32(I)
        I /= Imax
                
        I -= I.mean()
        I /= I.std()
        
        return I


    def segment_frame(self,Frame):
        
        # plt.figure() 
        

        I = self.imread(Frame)

            # O = I.copy()
            
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
                
            
                
            # plt.clf()
            # plt.imshow(O, cmap='gray')
            # plt.imshow(R, alpha=0.25)
            # plt.axis('off')
            # plt.savefig(os.path.join(self.Path_Output,Frame.split('.')[0]+'.png'), dpi=300)
            # plt.pause(0.01)
            # plt.draw()
        
        
        return R

    def run(self):
        
        for Frame in tqdm(self.Frames):

            p = self.segment_frame(Frame)
    
            Img = Image.fromarray(np.uint8(p*255))
            Img.save(os.path.join(self.Path_Output,Frame.split('.')[0]+'.png'))
 
from tensorflow.keras.models import load_model           
model = load_model(os.path.join('C:/Users/Adrian/Desktop/SlimeNet','model512_depth5_epoch15.h5'))
    

Path = 'C:/Users/Adrian/Desktop/Crop2'

Parameters = {'first_frame':None,\
              'last_frame':None,\
              'size':(512, 512)
                }
    
obj = segmentation(Path, Parameters, model)
obj.run()