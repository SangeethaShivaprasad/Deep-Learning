from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):

    def __init__(self, mode_flag, csv_path, split_param, compose_obj):
        self.mode_flag = mode_flag
        self.csv_path = csv_path
        self.split_param = split_param
        self.compose_obj = compose_obj
    
def get_train_dataset():
    #TODO
    pass

# this needs to return a dataset *without* data augmentation!
def get_validation_dataset():
    #TODO
    pass


