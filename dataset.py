import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from scipy.ndimage import convolve
import torch.nn.functional as F
import cv2 as cv
import matplotlib.pyplot as plt
import config

class Div2kDataset(Dataset):
    def __init__(self, hr_dir='local_data/small set of div2k (20)',sample_size=-1,hr_size=2040,scale=4):
        
        ## printing the confif of dataset
        print(f'HR directory : \t {hr_dir}')
        print(f'scale : \t {scale}')
        print(f'Sample size : \t {sample_size}')
        self.sample_size = sample_size
        self.hr_size = hr_size
        self.hr_edge_tensor = torch.tensor([])
        self.lr_edge_tensor = torch.tensor([])

        count = 0
        for img_name in os.listdir(hr_dir):
            img_path = os.path.join(hr_dir,img_name)
            hr_img = cv.imread(img_path,cv.IMREAD_COLOR)
            hr_img = cv.cvtColor(hr_img,cv.COLOR_BGR2RGB)
            hr_img = torch.tensor(hr_img).permute(2,0,1).to(torch.float32)
            w = hr_size

            padding_left = (w - hr_img.shape[2]) // 2
            padding_right = (w - hr_img.shape[2]) - padding_left
            padding_top = (w - hr_img.shape[1]) // 2
            padding_bottom = (w - hr_img.shape[1]) - padding_top

            hr_pad_tensor = F.pad(hr_img, (padding_left, padding_right, padding_top, padding_bottom))

            hr_pad_tensor_t = hr_pad_tensor.unsqueeze(0).to(torch.float32)
            ss = hr_size//scale
            lr_img_tensor = F.interpolate(hr_pad_tensor_t, size=(ss,ss), mode='bicubic', align_corners=False)

            self.hr_edge_tensor = torch.cat((self.hr_edge_tensor,hr_pad_tensor_t),0)
            self.lr_edge_tensor = torch.cat((self.lr_edge_tensor,lr_img_tensor),0)

            if count == sample_size:
                break
            else: count += 1



    def __len__(self):
        return self.hr_edge_tensor.shape[0]

    def __getitem__(self, idx):
        return self.hr_edge_tensor[idx], self.lr_edge_tensor[idx]
    


class EdgeDataset(Dataset):
    # we are not going to store the edge transformed images in the disk
    # as it is easier to generate them on the fly as they are less computational heavy.
    def __init__(self,hr_dir='local_data/small set of div2k (20)',sample_size=-1,hr_size=2040,scale=4):

        self.sample_size = sample_size
        #setting the path to the high resolution images

        high_res_folder_path = hr_dir


        # Define a simple edge detection kernel
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
            hr_img = cv.filter2D(hr_img, -1, kernal.T)
            # lr_img = cv.resize(hr_img,hr_img.shape, interpolation=cv.INTER_BICUBIC)
            hr_img = torch.tensor(hr_img).unsqueeze(0)
            w = hr_size

            padding_left = (w - hr_img.shape[2]) // 2
            padding_right = (w - hr_img.shape[2]) - padding_left
            padding_top = (w - hr_img.shape[1]) // 2
            padding_bottom = (w - hr_img.shape[1]) - padding_top

            hr_pad_tensor = F.pad(hr_img, (padding_left, padding_right, padding_top, padding_bottom))

            hr_pad_tensor_t = hr_pad_tensor.unsqueeze(0).to(torch.float32)
            ss = hr_size//scale
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


class Edge_3_dataset(Dataset):
    def __init__(self,hr_dir='local_data/small set of div2k (20)',sample_size=-1,hr_size=2040,scale=4):

        self.sample_size = sample_size
        #setting the path to the high resolution images

        high_res_folder_path = hr_dir


        # Define a simple edge detection kernel
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
            hr_img_x = cv.filter2D(hr_img, -1, kernal)
            hr_img_y = cv.filter2D(hr_img, -1, kernal.T)
            hr_img_c = cv.filter2D(hr_img, -1, kernal*0.5+kernal.T*0.5)
            # lr_img = cv.resize(hr_img,hr_img.shape, interpolation=cv.INTER_BICUBIC)
            hr_img_x = torch.tensor(hr_img_x).unsqueeze(0)
            hr_img_y = torch.tensor(hr_img_y).unsqueeze(0)
            hr_img_c = torch.tensor(hr_img_c).unsqueeze(0)

            w = hr_size

            hr_img = torch.cat((hr_img_x,hr_img_y,hr_img_c),0)

            padding_left = (w - hr_img.shape[2]) // 2
            padding_right = (w - hr_img.shape[2]) - padding_left
            padding_top = (w - hr_img.shape[1]) // 2
            padding_bottom = (w - hr_img.shape[1]) - padding_top

            hr_pad_tensor = F.pad(hr_img, (padding_left, padding_right, padding_top, padding_bottom))

            hr_pad_tensor_t = hr_pad_tensor.unsqueeze(0).to(torch.float32)
            ss = hr_size//scale
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

    dataset = Edge_3_dataset(sample_size=3)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, (x,y) in enumerate(loader):
        print(f'Batch {idx+1}')
        print(f'hr shape : {x.shape}')
        print(f'lr shape : {y.shape}')
        # data = x.squeeze(0).squeeze(0).numpy()
        # target = y.squeeze(0).squeeze(0).numpy()
        data = x.squeeze(0).numpy().transpose(1,2,0).astype(np.uint8)
        target = y.squeeze(0).numpy().transpose(1,2,0).astype(np.uint8)
        print(data.shape,target.shape)
        data = cv.cvtColor(data,cv.COLOR_BGR2RGB)
        target = cv.cvtColor(target,cv.COLOR_BGR2RGB)
        break

    cv.imwrite('output/hr_div_edge.jpg',data)
    cv.imwrite('output/lr_div_edge.jpg',target)
        