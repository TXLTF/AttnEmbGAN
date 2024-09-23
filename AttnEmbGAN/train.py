import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.attnembgan import Generator, Discriminator
from data.dataset.py import EmbroideryDataset

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
generator = Generator(input_nc, output_nc)
discriminator = Discriminator(input_nc)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss()

# Training Loop
for epoch in range(n_epochs):
    for i, imgs in enumerate(dataloader):
        # Training code here
        pass
