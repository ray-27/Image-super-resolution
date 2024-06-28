import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from scipy.ndimage import convolve
import torch.nn.functional as F
import cv2 as cv


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
    def __init__(self,hr_dir='data/small set of div2k (20)',sample_size=-1,hr_size=2040,scale=4):

        self.sample_size = sample_size
        #setting the path to the high resolution images

        high_res_folder_path = hr_dir


        # Define a simple edge detection kernel
        kernel = np.array([
                        [1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

        self.hr_edge_tensor = torch.tensor([])
        self.lr_edge_tensor = torch.tensor([])
        count = 0
        for img_name in os.listdir(high_res_folder_path):
            img_path = os.path.join(high_res_folder_path,img_name)
            hr_img = cv.imread(img_path,cv.IMREAD_GRAYSCALE) #grey scale image 
            
            #applying edge kernal and resizing the image to 2040x2040
            kernal = np.array([
                    [1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]
                ])
            hr_img = cv.filter2D(hr_img, -1, kernal)
            # lr_img = cv.resize(hr_img,hr_img.shape, interpolation=cv.INTER_BICUBIC)
            hr_img = torch.tensor(hr_img).unsqueeze(0)
            w = hr_size

            padding_left = (w - hr_img.shape[2]) // 2
            padding_right = (w - hr_img.shape[2]) - padding_left
            padding_top = (w - hr_img.shape[1]) // 2
            padding_bottom = (w - hr_img.shape[1]) - padding_top

            hr_pad_tensor = F.pad(hr_img, (padding_left, padding_right, padding_top, padding_bottom))

            hr_pad_tensor_t = hr_pad_tensor.unsqueeze(0).to(torch.float32)
            ss = 2040//4
            lr_img_tensor = F.interpolate(hr_pad_tensor_t, size=(ss,ss), mode='bicubic', align_corners=False)

            # lr_img_tensor = lr_img_tensor.squeeze(0)
            
            self.hr_edge_tensor = torch.cat((self.hr_edge_tensor,hr_pad_tensor_t),0)
            self.lr_edge_tensor = torch.cat((self.lr_edge_tensor,lr_img_tensor),0)

            if count == sample_size:
                break
            else: count += 1

    def __len__(self):
        return self.hr_edge_tensor.shape[0]

    def __getitem__(self, idx):
        return self.hr_edge_tensor[idx], self.lr_edge_tensor[idx]


if __name__ == "__main__":

    dataset = EdgeDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, (x,y) in enumerate(loader):
        print(f'Batch {idx+1}')
        print(f'hr shape : {x.shape}')
        print(f'lr shape : {y.shape}')
        