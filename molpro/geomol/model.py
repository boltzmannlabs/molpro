from argparse import ArgumentParser, Namespace
import argparse
import csv
import time
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from molpro.models.geomol_model import GeoMol
from molpro.geomol.data import GeomolDataModule
from molpro.geomol.geomol_utils import construct_conformers

def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default='drugs')
    parser.add_argument('--seed', type=int, default=0)

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--verbose', action='store_true', default=False)

def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    return args


def set_hyperparams():
    """
    return: A dictionary containing hyperparams.
    """

    hyperparams = {'model_dim': 25,
                   'random_vec_dim': 10,
                   'random_vec_std': 1,
                   'global_transformer': False,
                   'n_true_confs': 10,
                   'n_model_confs': 10,
                   'gnn1': {'depth': 3,
                            'n_layers': 2},
                   'gnn2': {'depth': 3,
                            'n_layers': 2},
                   'encoder': {'n_head': 2},
                   'coord_pred': {'n_layers': 2},
                   'd_mlp': {'n_layers': 1},
                   'h_mol_mlp': {'n_layers': 1},
                   'alpha_mlp': {'n_layers': 2},
                   'c_mlp': {'n_layers': 1},
                   'loss_type': "ot_emd",
                   'teacher_force': False,
                   'random_alpha': False}

    return hyperparams





class GeomolModelModule(pl.LightningModule):
    """Lightning trainer module to handle training/validation for dataset"""
    def __init__(self,hyper_parameters,num_node_features,num_edge_features):
        super().__init__()
        self.save_hyperparameters()
        self.model = GeoMol(hyper_parameters,num_node_features,num_edge_features)
    
    def forward(self, data,batch_idx):
        data = data
        #result = self.model(data) if batch_idx > 8 else self.model(data, ignore_neighbors=True) # To train on sampe_data
        result = self.model(data) if batch_idx > 128 else self.model(data, ignore_neighbors=True)
        return result


    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        train_loss = self(batch,batch_idx)
        # clip the gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
        self.log("train_loss", train_loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self,data,batch_idx):
        val_loss = self(data,batch_idx)        
        self.log("val_loss", val_loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self,data,batch_idx):
        test_loss = self(data,batch_idx)        
        self.log("test_loss", test_loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3)
        return optimizer
    def prediction(self,data,n_model_confs):
        self.model(data,inference=True,n_model_confs = n_model_confs)
        n_atoms = data.x.size(0)
        model_coords = construct_conformers(data, self.model)
        return model_coords




def main(params):
    print("Starting of main Func")
    st = time.perf_counter()

    print(params)
    geomol_data = GeomolDataModule(dataset_path=params.data_dir,dataset=params.dataset,
                          batch_size=params.batch_size)


    geomol_data.prepare_data()
    geomol_data.setup()
    hyperparams = set_hyperparams()
    if params.dataset == "drugs":
        model=GeomolModelModule(hyperparams,num_node_features = 74,num_edge_features = 4)
    else :
        model=GeomolModelModule(hyperparams,num_node_features=44,num_edge_features=4)
    print("Starting of trainers...")
    trainer = pl.Trainer(max_epochs=int(params.n_epochs),
                         progress_bar_refresh_rate=20,
                         gpus = -1 if torch.cuda.is_available() else None)


    trainer.fit(model, geomol_data)
    print("Training completed... Testing start....")
    trainer.test(model, geomol_data)


if __name__ == "__main__":
    st = time.perf_counter()
    configs = parse_train_args()
    print("config :",configs)
    main(configs)
    print(f"Total time taken: {time.perf_counter() - st}")


