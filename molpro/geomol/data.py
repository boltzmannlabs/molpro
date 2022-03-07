import os
import os.path as osp
import numpy as np
import glob
import pickle
import random
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Dataset,Data,DataLoader
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType as BT, ChiralType
from molpro.geomol.geomol_utils import one_k_encoding , get_dihedral_pairs , dihedral_pattern,chirality,qm9_types,drugs_types
import pytorch_lightning as pl
from molpro.utils.dataset import geomol_drugs_confs_dataset,geomol_qm9_confs_dataset

class GeomolDataModule(pl.LightningDataModule):
    """Lightning datamodule to handle dataprep for dataloaders 
        
        Parameters :
        ---------------------

        dataset_path : str 
                    path of the dataset
        batch_size : int
                  batch_size for model training
        nworkers: int,
                number of workers for pytorch dataloader """

    def __init__(self,dataset_path: str = './',dataset:str="drugs",batch_size: int = 1,
                                                 nworkers: int = 6):


        super().__init__()
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.batch_size = batch_size
        self.nworkers = nworkers

    def prepare_data(self):
        if not osp.exists(self.split_path):
            raise FileNotFoundError(f"file doesn't exist: {self.split_path}")
        np.random.seed(0)
        indexes = np.array(list(range(len(os.listdir(self.smiles_path)))))
        np.random.shuffle(indexes)
        self.train_indexes = indexes[:int(len(indexes)*80/100)]
        self.val_indexes = indexes[int(len(indexes)*80/100):int(len(indexes)*90/100)]
        self.test_indexes = indexes[int(len(indexes)*90/100):]
        print("Train_data_len:",len(self.train_indexes),"Val_data_len:", len(self.val_indexes),"Test_data_len:", len(self.test_indexes))

    def setup(self, stage=None):
        if self.dataset == "qm9":
            self.train_loader = geomol_qm9_confs_dataset(self.dataset_path,self.train_indexes,"train")
            self.val_loader = geomol_qm9_confs_dataset(self.dataset_path,self.val_indexes,"val")
            self.test_loader = geomol_qm9_confs_dataset(self.dataset_path,self.test_indexes,"test")
        

        if self.dataset == "drugs":
            self.train_loader = geomol_drugs_confs_dataset(self.dataset_path,self.train_indexes,"train")
            self.val_loader = geomol_drugs_confs_dataset(self.dataset_path,self.train_indexes,"train")
            self.test_loader = geomol_drugs_confs_dataset(self.dataset_path,self.train_indexes,"train")
        
    def train_dataloader(self):
        return DataLoader(self.train_loader, batch_size=self.batch_size,
                                 num_workers=self.nworkers)

    def val_dataloader(self):
        return DataLoader(self.val_loader, batch_size=self.batch_size, 
                               num_workers=self.nworkers)

    def test_dataloader(self):
        return DataLoader(self.test_loader, batch_size=self.batch_size,
                                  num_workers=self.nworkers)
