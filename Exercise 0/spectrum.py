import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

spectrum = np.zeros([256,256, 3], dtype=np.uint8) # init the array

r  = np.array(np.linspace(0,255,256))
np.tile(r,(256,1))


g  = np.array(np.linspace(0,255,256))
g = np.reshape(g,(256,1))
np.tile(g,(1,256))


b  = np.array(np.linspace(255,0,256))
np.tile(b,(256,1))

# fill the array with rgb values to create the spectrum without the use of loops
spectrum[:,:,0] = r
spectrum[:,:,1] = g
spectrum[:,:,2] = b
img = Image.fromarray(spectrum, 'RGB')

plt.imshow(img)
plt.show()

