import torch
from torch import nn
from pytorch_lightning import LightningModule, Trainer
from molpro.models.resnet import ResNet
from molpro.affinity_pred.data import AffinityPredDataModule
from argparse import ArgumentParser
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

pl.seed_everything(123)


class AffinityPredModel(LightningModule):
    def __init__(self, model_version: str, input_channel: int, output_channel: int, intermediate_channel: int,
                 learning_rate: float, **kwargs):
        super(AffinityPredModel, self).__init__()
        """Lightning module for affinity prediction
            Parameters
            ----------
            model_version: int,
                Version of resnet to train for affinity prediction
            input_channel: int,
                Input features for the 3D grid
            output_channel: int,
                Output features for the 3D grid
            intermediate_channel: int,
                Intermediate filters in convolution block
            learning_rate: float,
                Learning rate for training the model
        """
        self.save_hyperparameters("input_channel", "output_channel", "intermediate_channel", "learning_rate",
                                  "model_version")
        if model_version == 'resnet10':
            layers = [1, 1, 1, 1]
        elif model_version == 'resnet18':
            layers = [2, 2, 2, 2]
        elif model_version == 'resnet50':
            layers = [3, 4, 6, 3]
        elif model_version == 'resnet101':
            layers = [3, 4, 23, 3]
        elif model_version == 'resnet152':
            layers = [3, 8, 36, 3]
        elif model_version == 'resnet200':
            layers = [3, 24, 36, 3]
        else:
            print('Not a valid model name. Using default model resnet10')
            layers = [1, 1, 1, 1]
        self.net = ResNet(input_channel, output_channel, intermediate_channel, layers)
        self.lr = learning_rate
        self.criterion = nn.MSELoss()
        self.train_correlation = torchmetrics.R2Score()
        self.val_correlation = torchmetrics.R2Score()
        self.test_correlation = torchmetrics.R2Score()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser_model = parent_parser.add_argument_group("AffinityPredModel")
        parser_model.add_argument("--model_version", type=str, default='resnet10')
        parser_model.add_argument("--input_channel", type=int, default=8)
        parser_model.add_argument("--intermediate_channel", type=int, default=64)
        parser_model.add_argument("--output_channel", type=int, default=1)
        parser_model.add_argument("--learning_rate", type=float, default=1e-4)
        return parent_parser

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        correlation = self.train_correlation(y_pred, y)
        if not torch.isnan(loss):
            self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train_correlation', correlation, on_epoch=False, on_step=True, prog_bar=True, logger=True)
            return loss
        else:
            return None

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        correlation = self.val_correlation(y_pred, y)
        if not torch.isnan(loss):
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_correlation', correlation, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        correlation = self.test_correlation(y_pred, y)
        self.log('test_correlation', correlation, on_step=False, on_epoch=True, prog_bar=True, logger=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--max_dist", type=int, default=24, help="maximum distance for atoms from the center of grid")
    parser.add_argument("--grid_resolution", type=int, default=1, help="resolution for the grid")
    parser.add_argument("--augment", type=bool, default=True, help="perform augmentation of dataset")
    parser.add_argument("--hdf_path", type=str, default="data.h5",
                        help="path where dataset is stored")
    parser.add_argument("--train_ids_path", type=str, default="sample_data/affinity_pred/splits/train.txt",
                        help="path where list of train ids is stored")
    parser.add_argument("--val_ids_path", type=str, default="sample_data/affinity_pred/splits/valid.txt",
                        help="path where list of validation ids is stored")
    parser.add_argument("--test_ids_path", type=str, default="sample_data/affinity_pred/splits/test.txt",
                        help="path where list of test ids is stored")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for pytorch dataloader")
    parser.add_argument("--pin_memory", type=bool, default=True, help="whether to pin memory for pytorch dataloader")
    parser = AffinityPredModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model = AffinityPredModel(**vars(args))
    print(model.hparams)
    data_module = AffinityPredDataModule(args.hdf_path, args.max_dist, args.grid_resolution,
                                         args.train_ids_path, args.val_ids_path, args.test_ids_path, args.augment,
                                         args.batch_size, args.num_workers, args.pin_memory)
    trainer = Trainer.from_argparse_args(args, callbacks=[ModelCheckpoint(monitor='val_correlation', mode='max')])
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
