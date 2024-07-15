
import config

from dataset import EdgeDataset, Div2kDataset,Edge_3_dataset
from edge_model import Edge_Generator, Edge_Discriminator
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from loss import Edge_Gradient_loss, VGGLoss

import wandb

wandb.init(project='super_resolution',)



def train(dataset,generator, discriminator, opt_gen, opt_disc, mse, bce, vgg_loss, device, epochs=config.epochs):
    #loading the dataset
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers, pin_memory=config.pin_memory)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    wandb.watch(generator, log='all')


    #training loop
    for epoch in tqdm(range(epochs)):
        for idx, (high_res, low_res) in enumerate(loader):
            # hr = hr.to(device)
            # lr = lr.to(device)

            # #training the generator
            # optimizer_gen.zero_grad()
            # fake_hr = generator(lr)
            # fake_hr = fake_hr.to(device)
            # gen_loss = criterion(fake_hr, hr)
            # gen_loss.backward()
            # optimizer_gen.step()

            # #training the discriminator
            # optimizer_dis.zero_grad()
            # real_pred = discriminator(hr)
            # fake_pred = discriminator(fake_hr.detach())
            # dis_loss = (criterion(real_pred, torch.ones_like(real_pred)) + criterion(fake_pred, torch.zeros_like(fake_pred))) / 2
            # dis_loss.backward()
            # optimizer_dis.step()

            high_res = high_res.to(device)
            low_res = low_res.to(device)
            
            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            fake = generator(low_res)
            disc_real = discriminator(high_res)
            disc_fake = discriminator(fake.detach())
            disc_loss_real = bce(
                disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
            )
            disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = disc_loss_fake + disc_loss_real

            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            disc_fake = discriminator(fake)
            #l2_loss = mse(fake, high_res)
            adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
            loss_for_vgg = 0.006 * mse(fake, high_res)
            gen_loss = loss_for_vgg + adversarial_loss
            gen_loss = vgg_loss(fake, high_res)

            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            wandb.log({"gen_loss": gen_loss.item(), "dis_loss": loss_disc.item()})
            if idx % 4 == 0:
                tqdm.write(f'Epoch {epoch + 1}/{epochs}, Batch {idx + 1}/{len(loader)}, Gen Loss : {gen_loss.item()}, Dis Loss : {loss_disc.item()}')#, Dis Loss : {dis_loss.item()}')

    
    # saving the generator and discriminator after training
    gen_model_name = config.gen_model_name
    dis_model_name = config.dis_model_name
    for name, module in generator.named_modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()

    torch.save(generator, "gen_model.pt")
    torch.save(discriminator, "dis_model.pt")

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(f'{config.gen_model_name}.pth')
    wandb.log_artifact(artifact)
    wandb.finish()




if __name__ == "__main__":
    # dataset = EdgeDataset(config.dataset_path,sample_size=config.sample_size)
    # dataset = Div2kDataset(config.dataset_path,sample_size=config.sample_size)
    dataset = Edge_3_dataset(config.dataset_path,sample_size=config.sample_size)
    
    generator = Edge_Generator(3,num_channels=config.num_channels,
                               num_res_blocks=config.num_res_blocks)
    discriminator = Edge_Discriminator(in_channels=3)
    # criterion = nn.MSELoss()
    mse = nn.MSELoss()
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=config.gen_lr)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=config.dis_lr)
    device = config.device#'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    print("------------------------")
    epochs = config.epochs

    opt_gen = optim.Adam(generator.parameters(), lr=config.gen_lr, betas=(0.9, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=config.dis_lr, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    train(dataset, generator, discriminator, opt_gen, opt_disc, mse, bce, vgg_loss, device,epochs=epochs)
        
    