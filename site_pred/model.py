import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from models.unet import UNet, DiceLoss, compute_per_channel_dice
from data import SitePredDataModule
from argparse import ArgumentParser


class SitePredModel(LightningModule):
    def __init__(self, input_channel: int, output_channel: int, learning_rate: float, **kwargs):
        super(SitePredModel, self).__init__()
        """Lightning module for binding site prediction
            Parameters
            ----------
            input_channel: int,
                Input features for the 3D grid
            output_channel: int,
                Output features for the 3D grid
            learning_rate: float,
                Learning rate for training the model
        """
        self.save_hyperparameters("input_channel", "output_channel", "learning_rate")
        self.net = UNet(input_channel, output_channel)
        self.lr = learning_rate
        self.criterion = DiceLoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser_model = parent_parser.add_argument_group("SitePredModel")
        parser_model.add_argument("--input_channel", type=int, default=18)
        parser_model.add_argument("--output_channel", type=int, default=1)
        parser_model.add_argument("--learning_rate", type=float, default=1e-4)
        return parent_parser

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
        if not torch.isnan(loss):
            self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train_dice', dice, on_epoch=False, on_step=True, prog_bar=True, logger=True)
            return loss
        else:
            return None

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        dice = torch.mean(compute_per_channel_dice(y_pred, y, 1e-6))
        if not torch.isnan(loss):
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        dice = torch.mean(compute_per_channel_dice(y_pred, y, 1e-6))
        self.log('test_dice', dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--max_dist", type=int, default=35, help="maximum distance for atoms from the center of grid")
    parser.add_argument("--grid_resolution", type=int, default=2, help="resolution for the grid")
    parser.add_argument("--augment", type=bool, default=True, help="perform augmentation of dataset")
    parser.add_argument("--hdf_path", type=str, default="data.h5",
                        help="path where dataset is stored")
    parser.add_argument("--train_ids_path", type=str, default="sample_data/site_pred/splits/train.txt",
                        help="path where list of train ids is stored")
    parser.add_argument("--val_ids_path", type=str, default="sample_data/site_pred/splits/valid.txt",
                        help="path where list of validation ids is stored")
    parser.add_argument("--test_ids_path", type=str, default="sample_data/site_pred/splits/test.txt",
                        help="path where list of test ids is stored")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for pytorch dataloader")
    parser.add_argument("--pin_memory", type=bool, default=True, help="whether to pin memory for pytorch dataloader")
    parser = SitePredModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model = SitePredModel(**vars(args))
    print(model.hparams)
    data_module = SitePredDataModule(args.hdf_path, args.max_dist, args.grid_resolution,
                                     args.train_ids_path, args.val_ids_path, args.test_ids_path, args.augment,
                                     args.batch_size, args.num_workers, args.pin_memory)
    trainer = Trainer.from_argparse_args(args, callbacks=[ModelCheckpoint(monitor='val_dice', mode='max')])
    trainer.fit(model, data_module)
