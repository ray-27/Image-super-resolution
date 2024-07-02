import torch

device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

# Configuration file for the project
dataset_path = 'data/div2k-sample-50'
sample_size = 1 #number of samples to load
batch_size = 1 #batch size
shuffle = False #shuffle the dataset
num_workers = 2 #number of workers for the dataloader
pin_memory = False #pin memory for the dataloader

# Confuguration for the model training
epochs = 1 #number of epochs
gen_lr = 0.0002 #learning rate for the generator
dis_lr = 0.0002 #learning rate for the discriminator

# Configuration for the Generator
num_channels=1
num_res_blocks=1

# model details
gen_model_name = 'Edge_Generator'
dis_model_name = 'Edge_Discriminator'
