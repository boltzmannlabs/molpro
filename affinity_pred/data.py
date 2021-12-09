
from random import choice
from skimage.draw import ellipsoid
from scipy import ndimage
import numpy as np
import torch
import h5py
from boltpro.utils.preprocess import rotate_grid, make_3dgrid, Featurizer
from math import pi
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import os
from typing import Tuple
from tqdm.auto import tqdm
from argparse import ArgumentParser

pl.seed_everything(123)


def rotation_matrix(axis, theta):        
    axis = np.asarray(axis, dtype=np.float)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

ROTATIONS = [rotation_matrix([1, 1, 1], 0)]

for a1 in range(3):
    for t in range(1, 4):
        axis = np.zeros(3)
        axis[a1] = 1
        theta = t * pi / 2.0
        ROTATIONS.append(rotation_matrix(axis, theta))

for (a1, a2) in combinations(range(3), 2):
    axis = np.zeros(3)
    axis[[a1, a2]] = 1.0
    theta = pi
    ROTATIONS.append(rotation_matrix(axis, theta))
    axis[a2] = -1.0
    ROTATIONS.append(rotation_matrix(axis, theta))

for t in [1, 2]:
    theta = t * 2 * pi / 3
    axis = np.ones(3)
    ROTATIONS.append(rotation_matrix(axis, theta))
    for a1 in range(3):
        axis = np.ones(3)
        axis[a1] = -1
        ROTATIONS.append(rotation_matrix(axis, theta))

def rotate(coords, rotation):
        global ROTATIONS
        return np.dot(coords, ROTATIONS[rotation])
        

class TrainDataset(Dataset):

    def __init__(self, hdf_path: str, max_dist: int, grid_resolution: int, id_file_path: str, augment: bool) -> None:
        """Pytorch dataset class for preparing 3d grid and labels
            Parameters
            ----------
            hdf_path: str,
                Path to save the HDF5 file
            grid_resolution: float
                Resolution of a grid (in Angstroms)
            max_dist: float
                Maximum distance between atom and box center. Resulting box has size of
                2*`max_dist`+1 Angstroms and atoms that are too far away are not
                included.
            id_file_path: str,
                Path to text file containing pdb ids
            augment: bool,
                Whether to augment the 3d grid or not
        """

        self.transform = augment
        self.max_dist = max_dist
        self.grid_resolution = grid_resolution
        self.hdf_path = hdf_path
        self.data_handle = None
        f = open(id_file_path)
        ids = f.read().splitlines()
        f.close()
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.data_handle is None:
            self.data_handle = h5py.File(self.hdf_path, 'r')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pdbid = self.ids[idx]
        pkd = self.data_handle[pdbid]['pkd'][:]
        activity = self.data_handle[pdbid]['activity'][:]
        if self.transform:
            rot = choice(range(24))
            tr = 10 * np.random.rand(1, 3)
        else:
            rot = 0
            tr = (0,0,0)
        rec_grid, pocket_dens = self.prepare_complex(pdbid,rotation = rot,translation = tr)

        return np.concatenate((rec_grid, pocket_dens)), pkd, int(activity)
    
    def prepare_complex(self,pdb_id,rotation=0, translation=(0, 0, 0),vmin=0, vmax=1):
        
        prot_coords = self.data_handle[pdb_id]['prot_coords'][:]
        poc_coords = self.data_handle[pdb_id]['ligand_coords'][:]
        poc_features = self.data_handle[pdb_id]['ligand_features'][:]
        prot_features = self.data_handle[pdb_id]['prot_features'][:]
        prot_coords = rotate(prot_coords,rotation)
        prot_coords += translation
        poc_coords = rotate(poc_coords,rotation)
        poc_coords += translation
        rec_grid = make_grid(prot_coords,prot_features ,
                                            max_dist=self.max_dist)
        pocket_dens = make_grid(poc_coords,poc_features,
                                      max_dist=self.max_dist)
        rec_grid = rec_grid.transpose(3,0,1,2)
        pocket_dens = pocket_dens.transpose(3,0,1,2)
        return rec_grid, pocket_dens
        
class ValidDataset(Dataset):

    def __init__(self, hdf_path: str, max_dist: int, grid_resolution: int, id_file_path: str) -> None:
        """Pytorch dataset class for preparing 3d grid and labels
            Parameters
            ----------
            hdf_path: str,
                Path to save the HDF5 file
            grid_resolution: float
                Resolution of a grid (in Angstroms)
            max_dist: float
                Maximum distance between atom and box center. Resulting box has size of
                2*`max_dist`+1 Angstroms and atoms that are too far away are not
                included.
            id_file_path: str,
                Path to text file containing pdb ids
            augment: bool,
                Whether to augment the 3d grid or not
        """

        self.max_dist = max_dist
        self.grid_resolution = grid_resolution
        self.hdf_path = hdf_path
        self.data_handle = None
        f = open(id_file_path)
        ids = f.read().splitlines()
        f.close()
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.data_handle is None:
            self.data_handle = h5py.File(self.hdf_path, 'r')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pdbid = self.ids[idx]
        pkd = self.data_handle[pdbid]['pkd'][:]
        rot = 0
        tr = (0,0,0)
        rec_grid, pocket_dens = self.prepare_complex(pdbid,rotation = rot,translation = tr)

        return np.concatenate((rec_grid, pocket_dens)), pkd
    
    def prepare_complex(self,pdb_id,rotation=0, translation=(0, 0, 0),vmin=0, vmax=1):
        
        prot_coords = self.data_handle[pdb_id]['prot_coords'][:]
        poc_coords = self.data_handle[pdb_id]['ligand_coords'][:]
        poc_features = self.data_handle[pdb_id]['ligand_features'][:]
        prot_features = self.data_handle[pdb_id]['prot_features'][:]
        prot_coords = rotate(prot_coords,rotation)
        prot_coords += translation
        poc_coords = rotate(poc_coords,rotation)
        poc_coords += translation
        rec_grid = make_grid(prot_coords,prot_features ,
                                            max_dist=self.max_dist)
        pocket_dens = make_grid(poc_coords,poc_features,
                                      max_dist=self.max_dist)
        rec_grid = rec_grid.transpose(3,0,1,2)
        pocket_dens = pocket_dens.transpose(3,0,1,2)
        return rec_grid, pocket_dens
        
class ResNetDataModule(LightningDataModule):

    def __init__(self, hdf_path: str, max_dist: int, grid_resolution: int, train_ids_path: str, 
                 augment: bool, batch_size: int, num_workers: int, pin_memory: bool):
        super().__init__()

        """Pytorch lightning datamodule for preparing train, validation and test dataloader
            Parameters
            ----------
            hdf_path: str,
                Path to save the HDF5 file
            grid_resolution: float
                Resolution of a grid (in Angstroms)
            max_dist: float
                Maximum distance between atom and box center. Resulting box has size of
                2*`max_dist`+1 Angstroms and atoms that are too far away are not
                included.
            augment: bool,
                Whether to augment the 3d grid or not
            train_ids_path: str,
                Path to text file containing train dataset pdb ids
            valid_ids_path: str,
                Path to text file containing validation dataset pdb ids
            test_ids_path: str,
                Path to text file containing test dataset pdb ids
            batch_size: int,
                Batch size to be used for train and validation dataloader
            num_workers: int,
                number of workers for pytorch dataloader
            pin_memory: bool,
                Whether to pin memory for pytorch dataloader
        """

        self.transform = augment
        self.max_dist = max_dist
        self.grid_resolution = grid_resolution
        self.hdf_path = hdf_path
        self.train_ids_path = train_ids_path
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, **kwargs):
        """define train, test and validation datasets """
        self.train_dataset = TrainDataset(self.hdf_path, self.max_dist, self.grid_resolution,
                                           self.train_ids_path, self.transform)
        self.val_dataset = ValidDataset('valid_multitask.hdf', self.max_dist, self.grid_resolution,
                                           'valid_multitask.txt')

    def train_dataloader(self):
        """returns train dataloader"""
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last = True)
        return loader
    
    def val_dataloader(self):
        """returns train dataloader"""
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last = True)
        return loader
