import argparse
import csv
import os
import numpy as np
from rdkit import Chem
from typing import Callable, List
from pytorch_lightning import profiler
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import utils.preprocess as bup

VOCAB_LIST = [
                "pad", "start", "end",
                "C", "c", "N", "n", "S", "s", "P", "O", "o",
                "B", "F", "I",
                #  "Cl", "[nH]", "Br",
                "X", "Y", "Z",
                "1", "2", "3", "4", "5", "6",
                "#", "=", "-", "(", ")"  # Misc
            ]

vocab_c2i_v1 = {x: i for i, x in enumerate(VOCAB_LIST)}
vocab_i2c_v1 = {i: x for i, x in enumerate(VOCAB_LIST)}


def read_smi(smiles_path: str) -> List[str]:
    """Utility method to read smiles tokens from .smi file"""

    assert smiles_path.endswith(".smi"), "smiles file should end in .smi"
    smiles_tokens = None
    with open(smiles_path, 'r') as wf:
        smiles_tokens = wf.read().split('\n')
    return smiles_tokens


def read_csv(csv_path: str) -> List[str]:
    """Utility method to read smiles tokens from .csv file"""

    assert csv_path.endswith(".csv"), "smiles file should end in .csv"
    smiles_tokens = []
    with open(csv_path, 'r') as wf:
        data = csv.reader(wf, delimiter=',')
        next(data, None)
        for i in data:
            smiles_tokens.append(i[0])
        return smiles_tokens


def custom_collate(in_data):
    """
    Collects and creates a batch.
    """
    # Sort a data list by smiles length (descending order)
    in_data.sort(key=lambda x: x[2], reverse=True)
    images, smiles, lengths = zip(*in_data)

    images = torch.stack(images, 0)  # Stack images

    # Merge smiles (from tuple of 1D tensor to 2D tensor).
    # lengths = [len(smile) for smile in smiles]
    targets = torch.zeros(len(smiles), max(lengths)).long()
    for i, smile in enumerate(smiles):
        end = lengths[i]
        targets[i, :end] = smile[:end]
    return images, targets, lengths



class SmilesDataset(Dataset):
    """pytorch dataset for generating smiles tokens"""

    def __init__(self,smiles_tokens: List[str],file_type:str="smi"):

        self.smiles_tokens = smiles_tokens
        self.file_type = file_type

    def __len__(self):
        return len(self.smiles_tokens)

    def __getitem__(self, idx: int):
        smiles_token = self.smiles_tokens[idx]
        featurizer = bup.Featurizer(smiles_token)
        featurizer.generate_conformer()
        coords = featurizer.get_coords()
        centroid = coords.mean(axis=0)
        coords -= centroid
        afeats = featurizer.atom_features()
        vox = bup.make_3dgrid(coords, afeats, 23, 2)
        vox = np.squeeze(vox, 0).transpose(3, 0, 1, 2)
        mol = Chem.MolFromSmiles(smiles_token)
        if not mol:
            raise ValueError(f"Failed to parse molecule '{mol}'")

        sstring = Chem.MolToSmiles(mol)  # Make the SMILES canonical.
        sstring = sstring.replace("Cl", "X").replace("[nH]", "Y") \
                                            .replace("Br", "Z")
        try:
            vals = [1] + \
                   [vocab_c2i_v1[xchar] for xchar in sstring] + \
                   [2]
        except KeyError:
            raise ValueError(
                ("Unkown SMILES tokens: {} in string '{}'."
                 .format(", ".join([x for x in sstring if
                                    x not in vocab_c2i_v1]),
                         sstring)))
        end_token = vals.index(2)
        # vox = torch.rand(8, 24, 24, 24)
        return torch.Tensor(vox), torch.Tensor(vals), end_token + 1


class BpDataModule(pl.LightningDataModule):
    
    """Lightning datamodule to handle dataprep for dataloaders
        ---------------------
        smiles_path : str 
                    path of the smile dataset
        train_pc  : float 
                  % of data should use for training of model
        val_pc  : float 
                  % of data should use for validation of model
        batch_size : int
                  batch_size for model training
        read_func  : callable function
                   read_smi if input smile dataset is in '.smi' file or read_csv for '.csv' file 
        num_workers: int,
                number of workers for pytorch dataloader """
        

    def __init__(self,smiles_path: str = './', train_pc: float = 0.9,
                               val_pc: float = 0.1, batch_size: int = 16,
                               read_func: Callable = read_smi,nworkers: int = 6):


        super().__init__()
        self.smiles_path = smiles_path
        self.train_pc = train_pc
        self.val_pc = val_pc
        self.batch_size = batch_size
        self.smiles_tokens = None
        self.read_func = read_func
        self.nworkers = nworkers

    def prepare_data(self):
        if not os.path.exists(self.smiles_path):
            raise FileNotFoundError(f"file doesn't exist: {self.smiles_path}")
        self.smiles_tokens = self.read_func(self.smiles_path)
        self.train_len = int(self.train_pc * len(self.smiles_tokens))
        self.test_len = len(self.smiles_tokens) - self.train_len
        self.val_len = max(1, int(self.val_pc * self.train_len))
        print("Train_data_len:",self.train_len,"Val_data_len:", self.val_len,"Test_data_len:", self.test_len)

    def setup(self, stage=None):
        
        if stage == 'fit' or stage is None:
            bpdata = SmilesDataset(self.smiles_tokens[:self.train_len])
            self.bptrain, self.bpval = \
                random_split(bpdata,
                             [self.train_len-self.val_len, self.val_len])
            print(f'len: {len(self.bptrain)}, {len(self.bpval)}')
        self.bptest = SmilesDataset(self.smiles_tokens[-self.test_len:])
        print(f'len_self.bptest: {len(self.bptest)}')

    def train_dataloader(self):
        return DataLoader(self.bptrain, batch_size=self.batch_size,
                          collate_fn=custom_collate, num_workers=self.nworkers)

    def val_dataloader(self):
        return DataLoader(self.bpval, batch_size=self.batch_size,
                          collate_fn=custom_collate, num_workers=self.nworkers)

    def test_dataloader(self):
        #print("model in test_dataloader")
        return DataLoader(self.bptest, batch_size=self.batch_size,
                          collate_fn=custom_collate, num_workers=self.nworkers)



