import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from printer import intToSelfies

class VAE(nn.Module):
    def __init__(self, in_dimension, layer_1d, layer_2d, layer_3d, latent_dimension):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),
            nn.ReLU(),
            nn.Linear(layer_3d, latent_dimension*2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dimension, layer_3d),
            nn.ReLU(),
            nn.Linear(layer_3d, in_dimension),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    ### borrowed
    def forward(self, x):
        # Encoding
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)

        # Reparameterization
        z = self.reparameterize(mu, logvar/4)

        # Decoding
        x_recon = self.decoder(z)

        return x_recon, mu, logvar
    ###

# Check if GPU is available and if not, fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

num_epochs = 10

# Load the data
data = torch.load('./Data/train_combined_repr_tensor.pt')
print(data.shape)
dataset = TensorDataset(data)
batch_size = 32  # adjust the batch size as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# in_dimension, layer_1d, layer_2d, layer_3d, latent_dimension
model = VAE(in_dimension=2516, layer_1d=1500, layer_2d=1000, layer_3d=250, latent_dimension=80).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

def loss_function(x_decoded, x, mu, logvar):
    xent_loss = F.mse_loss(x_decoded, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (0.1 * kl_loss) + xent_loss

# Training loop
for epoch in range(num_epochs):
    ### borrowed
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        optimizer.zero_grad()

        batch = batch[0].to(device)  # move batch data to the device
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar)
        total_loss += loss.item()
        num_batches += 1
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / num_batches
    print(f'Epoch {epoch+1}, Loss: {avg_loss}')
    
    single_batch_example = batch[0][0: 1236] # Remember that the first 1236 elements correspons to the small molecule
    recon_example = recon_batch[0][0: 1236]
    print('Label molecule: ', intToSelfies(single_batch_example * 105)) # Remeber that 111 is the number of unique selfies characters we have 

    print('Recon example: ', intToSelfies(recon_example * 105))
    ### end borrowed

