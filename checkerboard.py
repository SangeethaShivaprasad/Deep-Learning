import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def Checkerboard(N,n):
    """N: size of board; n=size of each square; N/(2*n) must be an integer """
    if (N%(2*n)):
        print('Error: N/(2*n) must be an integer')
        return False
    a = np.concatenate((np.zeros(n),np.ones(n)))
    b = np.pad(a,int((N**2)/2-n),'wrap').reshape((N,N))


    return (b+b.T == 1).astype(int)

B = Checkerboard(200,5)

plt.imshow(B, cmap='gray')
plt.show()