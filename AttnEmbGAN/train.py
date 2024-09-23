import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.generator import Generator
from models.discriminator import Discriminator
from data.dataset import EmbroideryDataset

# Hyperparameters
batch_size = 1
lr = 0.0002
n_epochs = 200
input_nc = 3
output_nc = 3

# Data
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = EmbroideryDataset(root_dir='path/to/dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(input_nc, output_nc).to(device)
discriminator = Discriminator(input_nc).to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss()

# Training Loop
for epoch in range(n_epochs):
    for i, imgs in enumerate(dataloader):
        real_imgs = imgs.to(device)

        # Train Generator
        optimizer_G.zero_grad()

        # Generate a batch of images
        fake_imgs = generator(real_imgs)

        # Loss measures generator's ability to fool the discriminator
        loss_GAN = criterion_GAN(discriminator(fake_imgs), torch.ones_like(discriminator(fake_imgs)))
        loss_identity = criterion_identity(fake_imgs, real_imgs)

        loss_G = loss_GAN + loss_identity
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        # Real loss
        loss_real = criterion_GAN(discriminator(real_imgs), torch.ones_like(discriminator(real_imgs)))
        # Fake loss
        loss_fake = criterion_GAN(discriminator(fake_imgs.detach()), torch.zeros_like(discriminator(fake_imgs)))

        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")