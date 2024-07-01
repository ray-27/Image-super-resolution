
import config

from dataset import EdgeDataset
from edge_model import Edge_Generator, Edge_Discriminator
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm


def train(dataset,generator, discriminator, criterion, optimizer_gen, optimizer_dis, device, epochs=config.epochs):
    #loading the dataset
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers, pin_memory=config.pin_memory)

    generator = generator.to(device)
    discriminator = discriminator.to(device)


    #training loop
    for epoch in tqdm(range(epochs)):
        for idx, (hr, lr) in enumerate(loader):
            hr = hr.to(device)
            lr = lr.to(device)

            #training the generator
            optimizer_gen.zero_grad()
            fake_hr = generator(lr)
            fake_hr = fake_hr.to(device)
            gen_loss = criterion(fake_hr, hr)
            gen_loss.backward()
            optimizer_gen.step()

            #training the discriminator
            optimizer_dis.zero_grad()
            real_pred = discriminator(hr)
            fake_pred = discriminator(fake_hr.detach())
            dis_loss = (criterion(real_pred, torch.ones_like(real_pred)) + criterion(fake_pred, torch.zeros_like(fake_pred))) / 2
            dis_loss.backward()
            optimizer_dis.step()

            if idx % 4 == 0:
                tqdm.write(f'Epoch {epoch + 1}/{epochs}, Batch {idx + 1}/{len(loader)}, Gen Loss : {gen_loss.item()}, Dis Loss : {dis_loss.item()}')



if __name__ == "__main__":
    dataset = EdgeDataset(config.dataset_path,sample_size=config.sample_size)
    generator = Edge_Generator(1,num_channels=config.num_channels,
                               num_res_blocks=config.num_res_blocks)
    discriminator = Edge_Discriminator()
    criterion = nn.MSELoss()
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=config.gen_lr)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=config.dis_lr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    print("------------------------")
    epochs = 2

    train(dataset, generator, discriminator, criterion, optimizer_gen, optimizer_dis, device,epochs=epochs)
        
    