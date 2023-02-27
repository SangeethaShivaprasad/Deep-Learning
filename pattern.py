
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Checker :

 def __init__(self,N,n):
    self.N = N
    self.n = n
    self.output = np.array([])

 def draw(self):
    """N: size of board; n=size of each square; N/(2*n) must be an integer """
    if (self.N%(2*self.n)):
        print('Error: N/(2*n) must be an integer')
        return False
    a = np.concatenate((np.zeros(self.n),np.ones(self.n)))
    b = np.pad(a,int((self.N**2)/2-self.n),'wrap').reshape((self.N,self.N))

    self.output = (b+b.T == 1).astype(int)
    copy = np.copy(self.output)
    return(copy)

 def show(self):
    plt.imshow(self.output , cmap='gray')
    plt.show()


class Spectrum:
    def __init__(self, resolution):
        # initialization
        self.resolution = resolution

        self.output = np.array([])

    def draw(self):
        print(self.resolution)
        spectrum = np.zeros([(self.resolution) , (self.resolution) , 3], dtype=float)  # init the array

        r = np.array(np.linspace(0, (self.resolution-1) , (self.resolution) ))

        np.tile(r, ((self.resolution) , 1))

        g = np.array(np.linspace(0, (self.resolution - 1) ,(self.resolution) ))
        g = np.reshape(g, ((self.resolution), 1))
        np.tile(g, (1, (self.resolution)))

        b = np.array(np.linspace((self.resolution - 1), 0, (self.resolution)))
        np.tile(b, ((self.resolution), 1))

        # fill the array with rgb values to create the spectrum without the use of loops
        spectrum[:, :, 0] = r
        spectrum[:, :, 1] = g
        spectrum[:, :, 2] = b
        self.output = spectrum/(self.resolution - 1)
        print(self.output.shape)
        copy = np.copy(self.output )
        return (copy)



    def show(self):
      #  data =  self.output * (self.resolution - 1)
      #  img = Image.fromarray( self.output , 'RGB')
        plt.imshow(self.output)
        plt.show()

class Circle:
 def __init__(self,resolution,radius,position):

    # initialization
    self.resolution=resolution
    self.radius=radius
    self.position=position
    self.output=np.array([])


 def draw(self):
     self.reference_img = np.load('reference_arrays/circle.npy')
     xx, yy = np.mgrid[:self.resolution, :self.resolution]
     print(xx,yy)
     circle = (yy - self.position[0]) ** 2 + (xx - self.position[1]) ** 2  # circles contains the squared distance to the (position[0], position[1]) point
     #circle = (self.position[0]) ** 2 + (self.position[1]) ** 2

     self.output = (circle <= (self.radius**2))  # true points will be circle
   #  self.output = self.output.astype(int)

     copy = np.copy(self.output)
   #  self.output = self.reference_img
     return copy


 def show(self):
     # self.resolution = 50
     # plt.figure(figsize=(10, 10), dpi=self.resolution)
     color_map = plt.imshow(self.output)
     color_map.set_cmap("gray")
     # plt.imshow(self.output,cmap=plt.cm.gray, interpolation='nearest')
     plt.axis('off')
     plt.show()

#
# c = Circle(1024, 200, (512, 256))
# circ = c.draw()
# c.show()
# # print(circ)
