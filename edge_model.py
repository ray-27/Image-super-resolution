import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from dataset import Div2kDataset, EdgeDataset
from torchsummary import summary

class Gen_ResudialBlock(nn.Module):
    def __init__(self, in_channels,kernel_size=3,stride=1,num_channels=64):
        super(Gen_ResudialBlock,self).__init__()

        self.conv = nn.Conv2d(in_channels,num_channels,kernel_size,stride,padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.prelu = nn.PReLU(num_channels)

    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.conv(out)
        out = self.bn(out)

        return x + out

class UpscaleBlock(nn.Module):
    
    def __init__(self,in_channels,scale_factor=4,kernel_size=3,stride=1):
        super(UpscaleBlock,self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, kernel_size, stride,padding=1)
        self.ps = nn.PixelShuffle(scale_factor)  # in_c * 4, H, W --> in_c, H*2, W*2
        self.prelu = nn.PReLU(num_parameters=in_channels)
    
    def forward(self,x):
        out = self.conv(x)
        # print(f'upscale conv shape : {out.shape}')
        out = self.ps(out)
        # print(f'pixel shuffle shape : {out.shape}')
        out = self.prelu(out)
        # print(f'prelu shape : {out.shape}')
        return out

class Edge_Generator(nn.Module):
    def __init__(self,in_channels,num_channels=64,num_res_blocks=16):
        super(Edge_Generator,self).__init__()

        self.conv_1 = nn.Conv2d(in_channels,num_channels,kernel_size=9,stride=1,padding=4)
        self.prelu = nn.PReLU(num_parameters = num_channels)
        self.resuidal_blocks = nn.Sequential(*[Gen_ResudialBlock(num_channels,num_channels=num_channels) for _ in range(num_res_blocks)])
        self.conv_2 = nn.Conv2d(num_channels,num_channels,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        # self.upscale_blocks = nn.Sequential(*[UpscaleBlock(num_channels) for _ in range(1)])
        self.upscale_blocks = nn.Sequential(
            UpscaleBlock(num_channels,scale_factor=2),
            UpscaleBlock(num_channels,scale_factor=2)
        )
        self.conv_3 = nn.Conv2d(num_channels,in_channels,kernel_size=3,stride=1,padding=1)



    def forward(self,x):
        x = self.conv_1(x)
        x = self.prelu(x)
        # print(f'first block success : {x.shape}')
        x1 = self.resuidal_blocks(x)
        # print(f'resudial block success : {x1.shape}')
        x1 = self.conv_2(x1)
        x1 = self.bn(x1)

        x = x + x1 # skip connection conv2d + resudial block
        # print(f'x + x1 shape : {x.shape}')
        # print(f'skip connection success : {x.shape}')

        x = self.upscale_blocks(x)
        x = self.conv_3(x)


        return x

### now code for discriminator
class Dis_ResudialBlock(nn.Module):
    def __init__(self,in_channels,num_channels=64,kernel_size=3,stride=1,padding=1,lekey_relu=0.2):
        super(Dis_ResudialBlock,self).__init__()

        self.conv = nn.Conv2d(in_channels,num_channels,kernel_size,stride,padding)
        self.bn = nn.BatchNorm2d(num_channels)
        self.lrelu = nn.LeakyReLU(lekey_relu)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)

        return x

class Edge_Discriminator(nn.Module):
    def __init__(self,in_channels=1,num_channels=[64,128,256,512],lekey_relu=0.2):
        super(Edge_Discriminator,self).__init__()

        self.conv_1 = nn.Conv2d(in_channels,num_channels[0],kernel_size=3,stride=1,padding=1)
        self.lrelu = nn.LeakyReLU(lekey_relu)
        self.resudial_blocks = nn.Sequential(
            Dis_ResudialBlock(num_channels[0],num_channels[0],kernel_size=3,stride=2,padding=1),
            Dis_ResudialBlock(num_channels[0],num_channels[1],kernel_size=3,stride=1,padding=1),
            Dis_ResudialBlock(num_channels[1],num_channels[1],kernel_size=3,stride=2,padding=1),
            Dis_ResudialBlock(num_channels[1],num_channels[2],kernel_size=3,stride=1,padding=1),
            Dis_ResudialBlock(num_channels[2],num_channels[2],kernel_size=3,stride=2,padding=1),
            Dis_ResudialBlock(num_channels[2],num_channels[3],kernel_size=3,stride=1,padding=1),
            Dis_ResudialBlock(num_channels[3],num_channels[3],kernel_size=3,stride=2,padding=1)
        )


        self.dense_1 = nn.Linear(num_channels[3]*128*128,10,bias=True)
        self.lrelu = nn.LeakyReLU(lekey_relu)
        self.dense_2 = nn.Linear(10,1,bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):

        x = self.conv_1(x)
        x = self.lrelu(x)
        # print(f'before resudial block : {x.shape}')
        x = self.resudial_blocks(x)
        # print(x.shape)
        x = x.view(x.size(0),-1)
        x = self.dense_1(x)
        x = self.lrelu(x)
        x = self.dense_2(x)
        x = self.sigmoid(x)

        return x



if __name__ == "__main__":

    x = torch.randn(1,1,510,510)
    # model = Edge_Generator(1,num_channels=8,num_res_blocks=8)
    gen = Edge_Generator(1,num_channels=8,num_res_blocks=8)
    dis = Edge_Discriminator()
        

    summary(gen,(1,510,510))
    summary(dis,(1,2040,2040))
    



        
