import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from skimage.transform import rotate
import matplotlib
import glob
import os
import json
import random
class ImageGenerator:
    # Directory to store the downloaded data.
    image_retrieve_index = 0

    def __init__(self,data_dir,file_path,batch_size,image_size,rotation,mirroring,shuffle):
        self.file_path = file_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle


    def next(self):


        self.data_dir = "exercise_data/"
        dataarray = np.zeros(shape=(self.batch_size,self.image_size[0] ,self.image_size[1] ,self.image_size[2] ))

        filekey =  np.zeros(shape=(self.batch_size))
        filekey_all = np.zeros(shape=(100))
        label_array = np.zeros(100)
        labels_num = np.zeros(self.batch_size)
        filepath_array = ["" for x in range(100)]
        with open(self.file_path) as f:
            data = json.load(f)



        # Test folder contains all my numpy file traces
        traces = os.listdir(self.data_dir)

        it = 0
        for j, trace in enumerate(traces):
                 # Find the path of the file
                 filepath = os.path.join(self.data_dir, trace)
                 file_index_ver_1 = filepath.split("/")
                 file_index =  file_index_ver_1[1].split(".")

                 #print(trace)
                 label_array[int(file_index[0])] = data[str(int(file_index[0]))]
                 filepath_array[int(file_index[0])] = filepath
                 filekey_all[j] = j

        if self.shuffle == True:
            random.shuffle(filekey_all)
        
        for i in range(self.batch_size) :
                load_index = (self.image_retrieve_index + filekey_all[i])
                if(load_index > 99):
                    load_index = load_index - 100
                print(load_index)
                labels_num[i] = label_array[int(load_index)]
                image = np.load(filepath_array[int(load_index)])
                image = rotate(image, 360)
                #print(load_index)
                image = np.resize(image,(self.image_size[0],self.image_size[1],self.image_size[2]))
                if self.mirroring == True:
                     image = image[:,::-1]  # flipping colomns to get horizontal mirror effect
                if self.rotation:
                     image = rotate(image, 90)

                filekey[i] = i

                dataarray[i, :, :, :] = image

        #filekeylistall = list(filekey_all)
    #    filekeylist = list(filekey)`

          #  print(filekeylist)

       # for k in filekeylistall:
       #     it = it + 1
        #    print(it)
       #     if (it == self.batch_size):
       #         break
        #    photos = [dataarray[int(k)]]

       # it = 0
       # photos = np.array(photos)

       # for k in filekeylistall:
        #    it = it + 1
        #    print(it)
        #    if (it == self.batch_size):
         #       break

          #  labels_num = [label_array[int(k)]]


      #  np.reshape(labels_num,self.batch_size)

        #print(type(photos))

        self.image_retrieve_index = self.image_retrieve_index + (self.batch_size )
        photos_copy = np.copy(dataarray)
        labels_num_copy = np.copy(labels_num)
        return photos_copy,labels_num_copy

    def class_name(self,labels_num):
            class_label = {'0': 'airplane', '1': 'automobile', '2': 'bird', '3': 'cat', '4': 'deer', '5': 'dog','6': 'frog', '7': 'horse', '8': 'ship', '9': 'truck'}
            labels_string = [class_label[str(int(k))] for k in labels_num]
            return labels_string

    def show(self):
        fig = plt.figure()
        col = 3
        (photos,labels_num) = self.next()

        labels_string = self.class_name(labels_num)
        row = int(self.batch_size/col)
        remainder = self.batch_size % col
        if (remainder > 0) :
            row = row + 1
        count = 1

        for r in range(1, row + 1):
            for c in range(1, col + 1):
                ax = fig.add_subplot(row, col, count)
                ax.set_title(labels_string[count - 1])
                plt.axis('off')



                plt.imshow(photos[count - 1])
                count = count + 1
                if (count - 1 == self.batch_size):
                    break


        plt.show()



#
# data_dir = "exercise_data/"
# file_path = "Labels.json"
# batch_size = 60
# image_size = [50,50,3]
# obj = ImageGenerator(data_dir,file_path,batch_size,image_size,False,False,False)
# b1 = obj.next()[0]
# b2 = obj.next()[0]

# data_dir = "exercise_data/"
# file_path = "Labels.json"
# batch_size = 12
# image_size = [32,32,3]
# obj = ImageGenerator(data_dir,file_path,batch_size,image_size,False,False,True)
# obj.show()