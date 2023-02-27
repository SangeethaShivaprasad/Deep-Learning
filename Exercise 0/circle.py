import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

spectrum = np.zeros([256,256, 3], dtype=np.uint8) # init the array

(x, y) = (125, 60)

# Create a figure. Equal aspect so circles look circular
fig,ax = plt.subplots(1)
ax.set_aspect('equal')

# Show the image
ax.imshow(spectrum)

# Now, loop through coord arrays, and create a circle at each x,y pair
circ = Circle((x,y),50,color="white")
ax.add_patch(circ)

ax.set_yticklabels([])
ax.set_xticklabels([])

# Show the image
plt.show()