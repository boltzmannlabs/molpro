from argparse import ArgumentParser, Namespace
import argparse
import csv
import time
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from molpro.models.geomol_model import GeoMol
from data import GeomolDataModule
from geomol_utils import construct_conformers

def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--split_path', type=str)
    parser.add_argument('--dataset', type=str, default='drugs')
    parser.add_argument('--seed', type=int, default=0)

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--verbose', action='store_true', default=False)

    # Model arguments
    parser.add_argument('--model_dim', type=int, default=25)
    parser.add_argument('--random_vec_dim', type=int, default=10)
    parser.add_argument('--random_vec_std', type=float, default=1)
    parser.add_argument('--random_alpha', action='store_true', default=False)
    parser.add_argument('--n_true_confs', type=int, default=10)
    parser.add_argument('--n_model_confs', type=int, default=10)

    parser.add_argument('--gnn1_depth', type=int, default=3)
    parser.add_argument('--gnn1_n_layers', type=int, default=2)
    parser.add_argument('--gnn2_depth', type=int, default=3)
    parser.add_argument('--gnn2_n_layers', type=int, default=2)
    parser.add_argument('--encoder_n_head', type=int, default=2)
    parser.add_argument('--coord_pred_n_layers', type=int, default=2)
    parser.add_argument('--d_mlp_n_layers', type=int, default=1)
    parser.add_argument('--h_mol_mlp_n_layers', type=int, default=1)
    parser.add_argument('--alpha_mlp_n_layers', type=int, default=2)
    parser.add_argument('--c_mlp_n_layers', type=int, default=1)

    parser.add_argument('--global_transformer', action='store_true', default=False)
    parser.add_argument('--loss_type', type=str, default='ot_emd')
    parser.add_argument('--teacher_force', action='store_true', default=False)

def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    return args


def set_hyperparams(args):
    """
    Converts ArgumentParser args to hyperparam dictionary.

    :param args: Namespace containing the args.
    :return: A dictionary containing the args as hyperparams.
    """

    hyperparams = {'model_dim': args.model_dim,
                   'random_vec_dim': args.random_vec_dim,
                   'random_vec_std': args.random_vec_std,
                   'global_transformer': args.global_transformer,
                   'n_true_confs': args.n_true_confs,
                   'n_model_confs': args.n_model_confs,
                   'gnn1': {'depth': args.gnn1_depth,
                            'n_layers': args.gnn1_n_layers},
                   'gnn2': {'depth': args.gnn2_depth,
                            'n_layers': args.gnn2_n_layers},
                   'encoder': {'n_head': args.encoder_n_head},
                   'coord_pred': {'n_layers': args.coord_pred_n_layers},
                   'd_mlp': {'n_layers': args.d_mlp_n_layers},
                   'h_mol_mlp': {'n_layers': args.h_mol_mlp_n_layers},
                   'alpha_mlp': {'n_layers': args.alpha_mlp_n_layers},
                   'c_mlp': {'n_layers': args.c_mlp_n_layers},
                   'loss_type': args.loss_type,
                   'teacher_force': args.teacher_force,
                   'random_alpha': args.random_alpha}

    return hyperparams





class GeomolModelModule(pl.LightningModule):
    """Lightning trainer module to handle training/validation for dataset"""
    def __init__(self,hyper_parameters,node_features,edge_features):
        super().__init__()
        self.save_hyperparameters()
        self.model = GeoMol(hyper_parameters,node_features,edge_features)
    
    def forward(self, data,batch_idx):
        data = data
        result = self.model(data) if batch_idx > 8 else self.model(data, ignore_neighbors=True)
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
    geomol_data = GeomolDataModule(smiles_path=params.data_dir,split_path=params.split_path,dataset=params.dataset,
                          batch_size=params.batch_size)


    geomol_data.prepare_data()
    geomol_data.setup()
    hyperparams = set_hyperparams(params)
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

# args --data_dir /home/uttu/bayes_labs/my_geomol_new_versions/sample_data/drugs --split_path /home/uttu/bayes_labs/my_geomol_new_versions/sample_data/drugs_split.npy --n_epochs 3 --dataset drugs

