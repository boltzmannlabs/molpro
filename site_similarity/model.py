import torch
from pytorch_lightning import LightningModule, Trainer
from molpro.models.voxel_network import VoxelNetwork
from data import VoxelNetDataModule
from argparse import ArgumentParser
import torchmetrics
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

pl.seed_everything(123)

class VoxelNetModel(LightningModule):
    def __init__(self, hyper_parameters):
        super(ResNetModel, self).__init__()
        """Lightning module for binding site similarity prediction
            Parameters
            ----------
            hyper_parameters: argparse.Namespace,
                Dictionary containing parse parameters
        """
        self.save_hyperparameters()
        self.net = VoxelNetwork(36)
        self.lr = hyper_parameters.lr
        self.batch_size = hyper_parameters.batch_size
        self.mse = nn.MSELoss()
        self.epochs = hyper_parameters.epochs
        self.train_correlation = torchmetrics.R2Score()
        self.val_correlation = torchmetrics.R2Score()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(),lr = self.lr)
        return  optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.mse(y_pred, y)
        if torch.isnan(loss) == False:
            correlation = self.train_correlation(y_pred,y)
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_correlation', correlation, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            return loss
        else:
            return None
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.mse(y_pred, y)
        if torch.isnan(loss) == False:
            correlation = self.val_correlation(y_pred,y)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_correlation', correlation, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            
def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5, help="adam: learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=1000, help="epochs")
    parser.add_argument("--max_dist", type=int, default=24, help="maximum distance for atoms from the center of grid")
    parser.add_argument("--grid_resolution", type=int, default=2, help="resolution for the grid")
    parser.add_argument("--augment", type=bool, default=True, help="perform augmentation of dataset")
    parser.add_argument("--hdf_path", type=str, default="multitask.hdf",
                        help="path where dataset is stored")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for pytorch dataloader")
    parser.add_argument("--pin_memory", type=bool, default=True, help="whether to pin memory for pytorch dataloader")
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus to be used for training")
    hparams = parser.parse_args()
    return hparams

if __name__ == '__main__':
    hparams = parser_args()
    model = VoxelNetModel(hparams)
    print(model.hparams)
    data_module = VoxelNetDataModule(hparams.hdf_path, hparams.max_dist, hparams.grid_resolution,
                                     hparams.train_ids_path, hparams.augment, hparams.batch_size,
                                     hparams.num_workers, hparams.pin_memory)
    checkpoint_callback = ModelCheckpoint(monitor="val_correlation", mode="max")
    trainer = Trainer(gpus=hparams.gpus, max_epochs=hparams.epochs,callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)