import os
from argparse import ArgumentParser
from random import choice
from typing import Tuple
import h5py
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from scipy import ndimage
from skimage.draw import ellipsoid
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from molpro.utils.preprocess import rotate_grid, make_3dgrid, Featurizer


def prepare_dataset(data_path: str, hdf_path: str, df_path: str) -> None:
    """Prepare a HDF5 grouped dataset for I/O from mol2 files

        Parameters
        ----------
        data_path: str,
            Path containing mol2 files
        hdf_path: str,
            Path to save the HDF5 file
        df_path: str,
           Path to csv file containing pdb ids and associated smiles
    """
    df = pd.read_csv(df_path)
    ids = df['pdb']
    ligand_path = df['ligand_path']

    with h5py.File(hdf_path, mode='w') as f:
        for i, structure_id in enumerate(tqdm(ids)):
            try:
                protein_featurizer = Featurizer(os.path.join(data_path, structure_id, 'site.pdb'),
                                                'pdb', named_props=['partialcharge', 'heavydegree'],
                                                smarts_labels=['hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring'],
                                                metal_halogen_encode=False)
                ligand_featurizer = Featurizer(os.path.join(data_path, structure_id, '%s' % ligand_path[i]),
                                               'mol2', named_props=['heavydegree'],
                                               smarts_labels=['hydrophobic', 'aromatic', 'acceptor', 'donor'],
                                               metal_halogen_encode=False)
                prot_coords, prot_features = protein_featurizer.coords, protein_featurizer.features
                ligand_coords, ligand_features = ligand_featurizer.coords, ligand_featurizer.features
            except StopIteration:
                print('openbabel could not parse file skipping %s' % structure_id)
                continue

            centroid = prot_coords.mean(axis=0)
            prot_coords -= centroid
            ligand_coords -= centroid
            group = f.create_group(structure_id)
            for key, data in (('prot_coords', prot_coords),
                              ('prot_features', prot_features),
                              ('ligand_coords', ligand_coords),
                              ('ligand_features', ligand_features),
                              ('centroid', centroid)):
                group.create_dataset(key, data=data, shape=data.shape, dtype='float32', compression='lzf')


class SiteGenDataset(Dataset):

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
        pdb_id = self.ids[idx]
        if self.transform:
            rot = choice(range(24))
            tr = 5 * np.random.rand(1, 3)
        else:
            rot = 0
            tr = (0, 0, 0)
        rec_grid, lig_grid = self.prepare_complex(pdb_id, rotation=rot, translation=tr)

        return rec_grid, lig_grid

    def prepare_complex(self, pdb_id: str, rotation: int = 0,
                        translation: Tuple[int, int, int] = (0, 0, 0), v_min: int = 0,
                        v_max: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Transform coordinates and features to 3d probability density grid
            Parameters
            ----------
            pdb_id: str,
                PDB ID of complex to transform
            translation: tuple,
                distance for translation of 3d grid
            rotation: int,
                rotation integer that corresponds to certain axis and theta
            v_min: int,
                minimum density in the grid
            v_max: int,
                maximum density in the grid
        """

        prot_coords = self.data_handle[pdb_id]['prot_coords'][:]
        ligand_coords = self.data_handle[pdb_id]['ligand_coords'][:]
        ligand_features = self.data_handle[pdb_id]['ligand_features'][:]
        prot_features = self.data_handle[pdb_id]['prot_features'][:]
        prot_coords = rotate_grid(prot_coords, rotation)
        prot_coords += translation
        ligand_coords = rotate_grid(ligand_coords, rotation)
        ligand_coords += translation
        footprint = ellipsoid(2, 2, 2)
        footprint = footprint.reshape((1, *footprint.shape, 1))
        rec_grid = make_3dgrid(prot_coords, prot_features, max_dist=self.max_dist,
                               grid_resolution=self.grid_resolution)
        lig_grid = make_3dgrid(ligand_coords, ligand_features, max_dist=self.max_dist,
                               grid_resolution=1)
        margin = ndimage.maximum_filter(lig_grid, footprint=footprint)
        lig_grid += margin
        lig_grid = lig_grid.clip(v_min, v_max)

        zoom = rec_grid.shape[1] / lig_grid.shape[1]
        lig_grid = np.stack([ndimage.zoom(lig_grid[0, ..., i],
                                          zoom)
                             for i in range(ligand_features.shape[1])], -1)
        rec_grid = np.squeeze(rec_grid)
        rec_grid = rec_grid.transpose((3, 0, 1, 2))
        lig_grid = lig_grid.transpose((3, 0, 1, 2))
        return rec_grid, lig_grid


class SiteGenDataModule(LightningDataModule):

    def __init__(self, hdf_path: str, max_dist: int, grid_resolution: int, train_ids_path: str, valid_ids_path: str,
                 test_ids_path: str, augment: bool, batch_size: int, num_workers: int, pin_memory: bool):
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
        self.valid_ids_path = valid_ids_path
        self.test_ids_path = test_ids_path
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.test_dataset = None
        self.valid_dataset = None
        self.train_dataset = None

    def setup(self, **kwargs):
        """define train, test and validation datasets """
        self.train_dataset = SiteGenDataset(self.hdf_path, self.max_dist, self.grid_resolution,
                                            self.train_ids_path, self.transform)
        self.valid_dataset = SiteGenDataset(self.hdf_path, self.max_dist, self.grid_resolution,
                                            self.valid_ids_path, False)
        self.test_dataset = SiteGenDataset(self.hdf_path, self.max_dist, self.grid_resolution,
                                           self.test_ids_path, False)

    def train_dataloader(self):
        """returns train dataloader"""
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)
        return loader

    def val_dataloader(self):
        """returns val dataloader"""
        loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)
        return loader

    def test_dataloader(self):
        """returns test dataloader"""
        loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)
        return loader


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="sample_data/affinity_pred/data",
                        help="path where pdb and mol2 files are stored")
    parser.add_argument("--hdf_path", type=str, default="data.h5",
                        help="path where dataset is stored")
    parser.add_argument("--df_path", type=str, default="sample_data/affinity_pred/splits/pdbbind.csv",
                        help="path to csv file containing pdb ids and associated smiles")
    params = parser.parse_args()
    return params


if __name__ == '__main__':
    hparams = parser_args()
    prepare_dataset(hparams.data_path, hparams.hdf_path, hparams.df_path)
