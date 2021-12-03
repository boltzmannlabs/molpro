
from random import choice
from skimage.draw import ellipsoid
from scipy import ndimage
import numpy as np
import torch
import h5py
from molpro.utils.preprocess import Featurizer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import os
from typing import Tuple, Dict, List
from tqdm.auto import tqdm
from argparse import ArgumentParser
import torch.nn.functional as F
from math import sin, cos, sqrt, pi
from itertools import combinations


def prepare_gan_dataset(data_path: str, hdf_path: str, smiles_dict: Dict[str,List[str]]) -> None:
    """Prepare a HDF5 grouped dataset for I/O from mol2 files

        Parameters
        ----------
        data_path: str,
            Path containing mol2 files
        hdf_path: str,
            Path to save the HDF5 file
        smiles_dict: Dict[str, int],
           Smiles strings with keys containing pdb_ids
    """

    ids = os.listdir(data_path)

    with h5py.File(hdf_path, mode='w') as f:
        for structure_id in tqdm(ids):
            try:
                protein_featurizer = Featurizer(os.path.join(data_path, structure_id, 'protein.mol2'), 'mol2', False,
                                                False, True, True, True)
                prot_coords = protein_featurizer.get_coords()
                prot_features = protein_featurizer.atom_features()
                centroid = prot_coords.mean(axis=0)
                prot_coords -= centroid

                smiles_list = smiles_dict[structure_id]
                for j in range(len(smiles_list)):
                    ligand_featurizer = Featurizer(smiles_list[j], 'smi', False, False, True, True, False)
                    ligand_featurizer.generate_conformer()
                    ligand_coords = ligand_featurizer.get_coords()
                    ligand_coords -= centroid
                    ligand_features = ligand_featurizer.atom_features()

                    group_id = structure_id + '_' + smiles_list[j]
                    group = f.create_group(group_id)
                    for key, data in (('prot_coords', prot_coords),
                                      ('prot_features', prot_features),
                                      ('ligand_coords', ligand_coords),
                                      ('ligand_features', ligand_features),
                                      ('centroid', centroid)):
                        group.create_dataset(key, data=data, shape=data.shape, dtype='float32', compression='lzf')
            except:
                print('Rdkit could not parse file skipping %s' % structure_id)
                continue
                

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

def make_grid(coords, features, grid_resolution=1.0, max_dist=10.0):

        coords = np.asarray(coords, dtype=np.float)
        features = np.asarray(features, dtype=np.float)
        f_shape = features.shape
        num_features = f_shape[1]
        max_dist = float(max_dist)
        grid_resolution = float(grid_resolution)

        box_size = int(np.ceil(2 * max_dist / grid_resolution + 1))


        grid_coords = (coords + max_dist) / grid_resolution
        grid_coords = grid_coords.round().astype(int)


        in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
        grid = np.zeros((1, box_size, box_size, box_size, num_features),dtype = np.float32)
        for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
            grid[0, x, y, z] += f

        return grid

def rotate(coords, rotation):
        global ROTATIONS
        return np.dot(coords, ROTATIONS[rotation])                


class BicycleGANDataset(Dataset):

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
        if self.transform:
            rot = choice(range(24))
            tr = 5 * np.random.rand(1, 3)
        else:
            rot = 0
            tr = (0,0,0)
        rec_grid, pocket_dens = self.prepare_complex(pdbid,rotation = rot,translation = tr)

        return rec_grid, pocket_dens
    
    def prepare_complex(self,pdbid,rotation=0, translation=(0, 0, 0),vmin=0, vmax=1):
        
        prot_coords = self.data_handle[pdbid]['prot_coords'][:]
        poc_coords = self.data_handle[pdbid]['ligand_coords'][:]
        poc_features = self.data_handle[pdbid]['ligand_features'][:]
        prot_features = self.data_handle[pdbid]['prot_features'][:]
        prot_coords = rotate(prot_coords,rotation)
        prot_coords += translation
        poc_coords = rotate(poc_coords,rotation)
        poc_coords += translation
        footprint = ellipsoid(2, 2, 2)
        footprint = footprint.reshape((1, *footprint.shape, 1))
        rec_grid = make_grid(prot_coords,prot_features ,
                                            max_dist=self.max_dist,
                                            grid_resolution=self.grid_resolution)
        pocket_dens = make_grid(poc_coords,poc_features,
                                      max_dist=self.max_dist)
        margin = ndimage.maximum_filter(pocket_dens,footprint=footprint)
        pocket_dens += margin
        pocket_dens = pocket_dens.clip(vmin, vmax)

        zoom = rec_grid.shape[1] / pocket_dens.shape[1]
        pocket_dens = np.stack([ndimage.zoom(pocket_dens[0, ..., i],
                                                 zoom)
                                    for i in range(poc_features.shape[1])], -1)
        rec_grid = np.squeeze(rec_grid)
        rec_grid = rec_grid.transpose(3,0,1,2)
        pocket_dens = pocket_dens.transpose(3,0,1,2)
        return rec_grid, pocket_dens


class BicycleGANDataModule(LightningDataModule):

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
        self.train_dataset = BicycleGANDataset(self.hdf_path, self.max_dist, self.grid_resolution,
                                           self.train_ids_path, self.transform)
    
    def my_collate(self,batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        target = torch.FloatTensor(target)
        data = torch.FloatTensor(data)
        return F.normalize(data), F.normalize(target)

    def train_dataloader(self):
        """returns train dataloader"""
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)
        return loader


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="path where mol2 files are stored")
    parser.add_argument("--hdf_path", type=str, default="data.h5",
                        help="path where dataset is stored")
    hparams = parser.parse_args()
    return hparams


if __name__ == '__main__':
    hparams = parser_args()
    smiles = {'4wno': '"C1CC2=C3C(=CC=C2)C(=CN3C1)[C@H]4[C@@H](C(=O)NC4=O)C5=CNC6=CC=CC=C65"'}
    prepare_gan_dataset(hparams.data_path, hparams.hdf_path, smiles)
