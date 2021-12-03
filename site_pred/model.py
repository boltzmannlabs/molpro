import torch
from pytorch_lightning import LightningModule, Trainer
from molpro.models.unet import UNet, DiceLoss, compute_per_channel_dice
from data import SitepredDataModule
from argparse import ArgumentParser


class SitepredModel(LightningModule):
    def __init__(self, hyper_parameters):
        super(SitepredModel, self).__init__()
        """Lightning module for binding site prediction
            Parameters
            ----------
            hyper_parameters: argparse.Namespace,
                Dictionary containing parse parameters
        """
        self.save_hyperparameters()
        self.net = UNet(34, 1)
        self.lr = hyper_parameters.lr
        self.batch_size = hyper_parameters.batch_size
        self.criterion = DiceLoss()
        self.epochs = hyper_parameters.epochs

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        dice = torch.mean(compute_per_channel_dice(y_pred, y, 1e-6))
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('train_dice', dice, on_epoch=False, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        dice = torch.mean(compute_per_channel_dice(y_pred, y, 1e-6))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        dice = torch.mean(compute_per_channel_dice(y_pred, y, 1e-6))
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_dice', dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-2, help="adam: learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=3000, help="epochs")
    parser.add_argument("--max_dist", type=int, default=35, help="maximum distance for atoms from the center of grid")
    parser.add_argument("--grid_resolution", type=int, default=2, help="resolution for the grid")
    parser.add_argument("--augment", type=bool, default=True, help="perform augmentation of dataset")
    parser.add_argument("--hdf_path", type=str, default="data.h5",
                        help="path where dataset is stored")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for pytorch dataloader")
    parser.add_argument("--pin_memory", type=bool, default=True, help="whether to pin memory for pytorch dataloader")
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus to be used for training")
    hparams = parser.parse_args()
    return hparams


if __name__ == '__main__':
    hparams = parser_args()
    model = SitepredModel(hparams)
    print(model.hparams)
    data_module = SitepredDataModule(hparams.hdf_path, hparams.max_dist, hparams.grid_resolution,
                                     'train.txt', 'valid.txt', 'test.txt', hparams.augment, hparams.batch_size,
                                     hparams.num_workers, hparams.pin_memory)
    trainer = Trainer(gpus=hparams.gpus, max_epochs=hparams.epochs)
    trainer.fit(model, data_module)
