from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, concatenate, UpSampling2D, BatchNormalization, Dropout #, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def conv_block(i, nf, f, ks = 3, bn = False, do = 0):
    c = Conv2D(filters = nf*f, kernel_size = (ks, ks), kernel_initializer = 'he_normal', padding = 'same', activation = 'relu')(i)
    c = BatchNormalization()(c) if bn else c
    c = Conv2D(filters = nf*f, kernel_size = (ks, ks), kernel_initializer = 'he_normal', padding = 'same', activation = 'relu')(c)
    c = BatchNormalization()(c) if bn else c
    m = MaxPooling2D((2, 2))(c)
    return [Dropout(do)(m),c] if do else [m,c]
    
### UpSampling2D + Conv against Checkerboard-pattern

def deconv_block(i, c2, nf, f, ks = 3, bn = False, do = 0):
    c1 = UpSampling2D(size=(2,2))(i)
    c1 = Conv2D(filters = nf*f, kernel_size=(2,2), padding = 'same')(c1)
    c1 = BatchNormalization()(c1) if bn else c1
    c = concatenate([c1, c2])
    c = Conv2D(filters = nf*f, kernel_size = (ks, ks), kernel_initializer = 'he_normal', padding = 'same', activation = 'relu')(c)
    c = BatchNormalization()(c) if bn else c
    c = Conv2D(filters = nf*f, kernel_size = (ks, ks), kernel_initializer = 'he_normal', padding = 'same', activation = 'relu')(c)
    c = BatchNormalization()(c) if bn else c
    return Dropout(do)(c) if do else c
    
### using actual deconvolution

#def deconv_block(i, c2, nf, f, ks = 3, bn = False, do = 0):
#    c1 = Conv2DTranspose(nf * f, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(i)
#    c1 = BatchNormalization()(c1) if bn else c1
#    c = concatenate([c1, c2])
#    c = Conv2D(filters = nf*f, kernel_size = (ks, ks), kernel_initializer = 'he_normal', padding = 'same', activation = 'relu')(c)
#    c = BatchNormalization()(c) if bn else c
#    c = Conv2D(filters = nf*f, kernel_size = (ks, ks), kernel_initializer = 'he_normal', padding = 'same', activation = 'relu')(c)
#    c = BatchNormalization()(c) if bn else c
#    return Dropout(do)(c) if do else c

def unet(depth=4, size=(None,None,1), nc=1, nf=64, ks=3, bn=False, do=0, LR = 1e-4):
    
    """
    Implementation of the UNet CNN model for semantic segmentation as 
    specified by Ronneberger et al. 2015 [1]. An additional hyperparameter
    has been added to procedurally control the number of convolution
    blocks in the contracting path and deconvolution blocks in the
    expanding path. The number of layers and the number of weights
    increases with depth. As implemented here, UpSampling2D + Conv2D
    is used for upsampling in the expanding path instead of Conv2DTranspose
    in order to reduce checkerboard-artifacts [2].
    Uses 'Adam' optimizer with binary crossentropy as loss metric.
    
    
    Parameters
    ----------
        depth : int
            controls number of conv/deconv-blocks
        size : (int, int, int)
            dimensions of input data
        nc : int
            number of output channels (eg. 2 => binary)
        nf : int
            number of filters
        ks : int
            kernel size of convolution filters
        do : float
            dropout (0..1)
        bn : bool
            batch normalization
        LR : float
            learning rate
        
    
    Returns
    -------
        model : keras/tensorflow cnn model
            untrained model
    
    
    References
    ----------
        [1] : https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28
        [2] : https://distill.pub/2016/deconv-checkerboard/
    
    """
    
    i = Input(size)
        
    ### contracting path
    
    c = {}
    
    for d in range(0,depth):
        
        c[d] = conv_block(i if d==0 else c[d-1][0],nf,2**d,ks,bn,do)
        
    ### bottleneck
    
    b = Conv2D(filters = nf*2**depth, kernel_size = (ks, ks), kernel_initializer = 'he_normal', padding = 'same', activation = 'relu')(c[depth-1][0])
    b = BatchNormalization()(b) if bn else b
    b = Conv2D(filters = nf*2**depth, kernel_size = (ks, ks), kernel_initializer = 'he_normal', padding = 'same', activation = 'relu')(b)
    b = BatchNormalization()(b) if bn else b
        
    ### expanding path

    e = {}

    for d in reversed(range(0,depth)):

        e[d] = deconv_block(b if d==depth-1 else e[d+1],c[d][1],nf,2**d,ks,bn,do)
        
    ### output
    
    o = Conv2D(filters = nc, kernel_size = (1, 1), activation='softmax')(e[0])
        
    model = Model(inputs = i, outputs = o)
    
    model.compile(optimizer = Adam(lr = LR), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

if __name__ == '__main__':
    
    model = unet()
    model.summary()