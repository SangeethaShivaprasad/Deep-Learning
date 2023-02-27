import numpy as np
import pattern
from generator import ImageGenerator

c = pattern.Checker(250, 25)
c.draw()
c.show()
s = pattern.Spectrum(255)
s.draw()
s.show()
circ = pattern.Circle(1024, 200, (512, 256))
circ.draw()
circ.show()

data_dir = "exercise_data/"
file_path = "Labels.json"
batch_size = 12
image_size = [32,32,3]
obj = ImageGenerator(data_dir,file_path,batch_size,image_size,False,False,True)
obj.show()
# b1 = obj.next()[0]
# b2 = obj.next()[0]