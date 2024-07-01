
# Configuration file for the project
dataset_path = 'data/div2k-sample-50'
sample_size = 2 #number of samples to load
batch_size = 4 #batch size
shuffle = True #shuffle the dataset
num_workers = 4 #number of workers for the dataloader
pin_memory = True #pin memory for the dataloader

# Confuguration for the model training
epochs = 10 #number of epochs
gen_lr = 0.0002 #learning rate for the generator
dis_lr = 0.0002 #learning rate for the discriminator

# Configuration for the Generator
num_channels=8
num_res_blocks=8
