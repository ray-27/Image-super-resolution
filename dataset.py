import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os


class Div2kDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]

        if self.transform:
            data = self.transform(data)
            target = self.transform(target)

        return data, target
    


class EdgeDataset(Dataset):
    # we are not going to store the edge transformed images in the disk
    # as it is easier to generate them on the fly as they are less computational heavy.
    def __init__(self,sample_size=0,small_sample=False):

        #setting the path to the high resolution images
        if small_sample:
            high_res_folder_path = os.path.join('data', 'data/small set of div2k (20)')
        else:
            high_res_folder_path = os.path.join('data', 'div2k/DIV2K_train_HR/DIV2K_train_HR')

        if sample_size != 0:
            # for i in range(sample_size):
                
                #TODO: implement the edge detection and store the images in the disk
        

        pass
