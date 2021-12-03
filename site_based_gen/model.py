import torch
from torch import nn
from pytorch_lightning import Trainer, LightningModule
from molpro.models.bicycle_gan import Generator, Discriminator, Encoder
from data import BicycleGANDataModule
from argparse import ArgumentParser
import pytorch_lightning as pl

pl.seed_everything(1234)

class BicycleGAN(LightningModule):
    def __init__(self, z_dim=8, lambda_kl=0.01, lambda_grid=10, lambda_z=0.5, lr=1e-3,
                 batch_size=2, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.G = Generator()
        self.D_cVAE = Discriminator()
        self.D_cLR = Discriminator()
        self.E = Encoder()
        self.z_dim = z_dim
        self.lambda_kl = lambda_kl
        self.lambda_grid = lambda_grid
        self.lambda_z = lambda_z
        self.lr = lr
        self.batch_size = batch_size
        self.L1_loss = nn.L1Loss()
        self.automatic_optimization = False

    def mse_loss(self,score, target=1):
        if target == 1:
            labels = torch.ones(score.size(),device=self.device)
        elif target == 0:
            labels = torch.zeros(score.size(),device=self.device)
        criterion = nn.MSELoss()
        loss = criterion(score, labels)
        return loss

    def forward(self, x, z) :
        return self.G(x, z)

    def get_z_random(self, batch_size):
        z = torch.randn(batch_size, self.z_dim, device = self.device)
        return z.detach()

    def encode(self, input_grid):
        mu, logvar = self.E(input_grid)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(1)
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def training_step(self, batch, batch_idx):

        g_opt, e_opt, d_cVAE_opt, d_cLR_opt = self.optimizers()

        protein_grid, ligand_grid = batch

        real_A_encoded = protein_grid[0].unsqueeze(dim=0)
        real_B_encoded = ligand_grid[0].unsqueeze(dim=0)
        real_A_random = protein_grid[1].unsqueeze(dim=0)
        real_B_random = ligand_grid[1].unsqueeze(dim=0)

        ############################################################
        # Optimize Conditional Variational Autoencoder Discriminator #
        ############################################################

        z_encoded, mu, logvar = self.encode(real_B_encoded)
        # get random z
        z_random = self.get_z_random(1)
        # generate fake_B_encoded
        fake_B_encoded = self.G(real_A_encoded, z_encoded)
        # generate fake_B_random
        fake_B_random = self.G(real_A_encoded, z_random)

        fake_data_encoded = torch.cat([real_A_encoded, fake_B_encoded], 1)
        real_data_encoded = torch.cat([real_A_encoded, real_B_encoded], 1)
        fake_data_random = torch.cat([real_A_encoded, fake_B_random], 1)
        real_data_random = torch.cat([real_A_random, real_B_random], 1)

        mu2, logvar2 = self.E(fake_B_random)

        self.set_requires_grad([self.D_cVAE, self.D_cLR], False)

        e_opt.zero_grad()
        g_opt.zero_grad()

        fake_d_cVAE_1, fake_d_cVAE_2 = self.D_cVAE(fake_data_encoded)
        GAN_loss_cVAE_1 = self.mse_loss(fake_d_cVAE_1, 1)
        GAN_loss_cVAE_2 = self.mse_loss(fake_d_cVAE_2, 1)

        fake_d_cLR_1, fake_d_cLR_2 = self.D_cLR(fake_data_random)

        GAN_loss_cLR_1 = self.mse_loss(fake_d_cLR_1, 1)
        GAN_loss_cLR_2 = self.mse_loss(fake_d_cLR_2, 1)

        G_GAN_loss = GAN_loss_cVAE_1 + GAN_loss_cVAE_2 + GAN_loss_cLR_1 + GAN_loss_cLR_2
        KL_div = torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * (-0.5 * self.lambda_kl)
        grid_recon_loss = self.lambda_grid * self.L1_loss(fake_B_encoded, real_B_encoded)

        G_loss = G_GAN_loss + KL_div + grid_recon_loss
        
        G_alone_loss = self.lambda_z * self.L1_loss(mu2, z_random)
        
        if torch.isnan(G_loss) == False:

            self.manual_backward(G_loss, retain_graph=True)

            self.set_requires_grad([self.E], False)

            self.manual_backward(G_alone_loss)
            self.set_requires_grad([self.E], True)

            e_opt.step()
            g_opt.step()

            #######################################################
            # Optimize Conditional Latent Regressor Discriminator #
            #######################################################

        self.set_requires_grad([self.D_cVAE, self.D_cLR], True)
        # Get scores and loss
        d_cVAE_opt.zero_grad()
        real_d_cVAE_1, real_d_cVAE_2 = self.D_cVAE(real_data_encoded)
        fake_d_cVAE_1, fake_d_cVAE_2 = self.D_cVAE(fake_data_encoded.detach())

        D_loss_cVAE_1 = self.mse_loss(real_d_cVAE_1, 1) + self.mse_loss(fake_d_cVAE_1, 0)
        D_loss_cVAE_2 = self.mse_loss(real_d_cVAE_2, 1) + self.mse_loss(fake_d_cVAE_2, 0)

        D_loss_cVAE = D_loss_cVAE_1 + D_loss_cVAE_2
        
        
        if torch.isnan(D_loss_cVAE) == False:
            self.manual_backward(D_loss_cVAE)
            d_cVAE_opt.step()

        d_cLR_opt.zero_grad()
        real_d_cLR_1, real_d_cLR_2 = self.D_cLR(real_data_random)
        fake_d_cLR_1, fake_d_cLR_2 = self.D_cLR(fake_data_random.detach())

        D_loss_cLR_1 = self.mse_loss(real_d_cLR_1, 1) + self.mse_loss(fake_d_cLR_1, 0)
        D_loss_cLR_2 = self.mse_loss(real_d_cLR_2, 1) + self.mse_loss(fake_d_cLR_2, 0)

        D_loss_cLR = D_loss_cLR_1 + D_loss_cLR_2
        
        if torch.isnan(D_loss_cLR) == False:
            self.manual_backward(D_loss_cLR)
            d_cLR_opt.step()

        self.log_dict({'g_loss': G_alone_loss, 'd_loss_cVAE': D_loss_cVAE, 'd_loss_cLR': D_loss_cLR, 'g_e_loss' : G_loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        e_opt = torch.optim.Adam(self.E.parameters(), lr=self.lr)
        d_cVAE_opt = torch.optim.Adam(self.D_cVAE.parameters(), lr=self.lr)
        d_cLR_opt = torch.optim.Adam(self.D_cLR.parameters(), lr=self.lr)
        return g_opt, e_opt, d_cVAE_opt, d_cLR_opt


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--max_dist", type=int, default=23, help="maximum distance for atoms from the center of grid")
    parser.add_argument("--grid_resolution", type=int, default=2, help="resolution for the grid")
    parser.add_argument("--augment", type=bool, default=False, help="perform augmentation of dataset")
    parser.add_argument("--hdf_path", type=str, default="train.hdf",
                        help="path where dataset is stored")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size for pytorch dataloader")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for pytorch dataloader")
    parser.add_argument("--pin_memory", type=bool, default=True, help="whether to pin memory for pytorch dataloader")
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus to be used for training")
    hparams = parser.parse_args()
    return hparams


if __name__ == '__main__':
    hparams = parser_args()
    model = BicycleGAN()
    print(model.hparams)
    data_module = BicycleGANDataModule(hparams.hdf_path, hparams.max_dist, hparams.grid_resolution,
                                       'train.txt', hparams.augment, hparams.batch_size,
                                       hparams.num_workers, hparams.pin_memory)
    trainer = Trainer(gpus=hparams.gpus, max_epochs=hparams.epochs, terminate_on_nan=True)
    trainer.fit(model, data_module)
