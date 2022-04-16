import csv
import time
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from molpro.models.geomol_model import GeoMol
from molpro.geomol.data import GeomolDataModule
from molpro.geomol.geomol_utils import construct_conformers


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






def train_geomol(data_dir:str = "/",dataset:str = "drugs", seed:int=0, n_epochs:int = 2, batch_size:int = 16,
                                      lr:float = 1e-3, num_workers:int = 6, num_gpus:int = 1, verbose:bool = False):

    """ This function trains the geomol model.

    Parameters:
    -------------
    data_dir : str 
             directory path where data is stored for training
    dataset : str
             on which dataset you want to train the model "drugs" or "qm9"
    seed : int
           the seed you want to fix
    n_epochs : int 
           number of epochs you want to train the model
    bath_size : int
           bath size for model training
    lr : float
           learning rate for optimizers
    num_workers : int
           number of workers to be used
    num_gpus : int
           on how many gpus you want to train then model
    verbose : bool
           verbose "True" or "False"

    """
    
    st = time.perf_counter()
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
    trainer = pl.Trainer(max_epochs=params.n_epochs,
                         progress_bar_refresh_rate=20,
                         gpus = params.num_gpus if torch.cuda.is_available() else None)

    trainer.fit(model, geomol_data)
    print("Training completed... Testing start....")
    trainer.test(model, geomol_data)
    print(f"Total time taken: {time.perf_counter() - st}")
