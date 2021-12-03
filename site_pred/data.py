
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


def prepare_dataset(data_path: str, hdf_path: str) -> None:
    """Prepare a HDF5 grouped dataset for I/O from mol2 files

        Parameters
        ----------
        data_path: str,
            Path containing mol2 files
        hdf_path: str,
            Path to save the HDF5 file
    """

    ids = os.listdir(data_path)
    multiple_pockets = {}

    with h5py.File(hdf_path, mode='w') as f:
        for structure_id in tqdm(ids):
            try:
                protein_featurizer = Featurizer(os.path.join(data_path, structure_id, 'protein.mol2'), 'mol2', False,
                                                False, True, True)
                pocket_featurizer = Featurizer(os.path.join(data_path, structure_id, 'cavity6.mol2'), 'mol2', False,
                                               False, False, False)

                pocket_coords = pocket_featurizer.get_coords()
                pocket_features = np.ones((len(pocket_coords), 1))
                prot_coords = protein_featurizer.get_coords()
                prot_features = protein_featurizer.atom_features()
            except:
                print('Rdkit could not parse file skipping %s' % structure_id)
                continue

            centroid = prot_coords.mean(axis=0)
            pocket_coords -= centroid
            prot_coords -= centroid

            group_id = structure_id[:-2]
            if group_id in f:
                group = f[group_id]
                if not np.allclose(centroid, group['centroid'][:], atol=0.5):
                    print('Structures for %s are not aligned, ignoring pocket %s' % (group_id, structure_id))
                    continue

                multiple_pockets[group_id] = multiple_pockets.get(group_id, 1) + 1

                for key, data in (('pocket_coords', pocket_coords),
                                  ('pocket_features', pocket_features)):
                    data = np.concatenate((group[key][:], data))
                    del group[key]
                    group.create_dataset(key, data=data, shape=data.shape, dtype='float32', compression='lzf')
            else:
                group = f.create_group(group_id)
                for key, data in (('coords', prot_coords),
                                  ('features', prot_features),
                                  ('pocket_coords', pocket_coords),
                                  ('pocket_features', pocket_features),
                                  ('centroid', centroid)):
                    group.create_dataset(key, data=data, shape=data.shape, dtype='float32', compression='lzf')


class SitepredDataset(Dataset):
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
            theta = choice(range(1, 4)) * pi / 2.0
            a1 = choice(range(3))
            axis = np.zeros(3)
            axis[a1] = 1
            tr = 5 * np.random.rand(1, 3)
        else:
            theta = 0
            axis = np.zeros(3)
            tr = (0, 0, 0)
        rec_grid, pocket_dens = self.prepare_complex(pdb_id, theta, axis, (0, 0, 0), tr)
        return rec_grid, pocket_dens

    def prepare_complex(self, pdb_id: str, theta: int, axis: np.ndarray, center: Tuple[int, int, int],
                        translation: Tuple[int, int, int], v_min: int = 0,
                        v_max: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Transform coordinates and features to 3d probability density grid
            Parameters
            ----------
            pdb_id: str,
                PDB ID of complex to transform
            center: tuple, optional
                center to rotate the 3d grid
            axis: np.ndarray,
                axis to rotate the 3d grid
            theta: int,
                angle to rotate the grid
            translation: tuple,
                distance for translation of 3d grid
            v_min: int,
                minimum density in the grid
            v_max: int,
                maximum density in the grid
        """

        prot_coords = self.data_handle[pdb_id]['coords'][:]
        poc_coords = self.data_handle[pdb_id]['pocket_coords'][:]
        poc_features = self.data_handle[pdb_id]['pocket_features'][:]
        prot_features = self.data_handle[pdb_id]['features'][:]
        prot_coords = rotate_grid(prot_coords, center, axis, theta)
        prot_coords += translation
        poc_coords = rotate_grid(poc_coords, center, axis, theta)
        poc_coords += translation
        footprint = ellipsoid(2, 2, 2)
        footprint = footprint.reshape((1, *footprint.shape, 1))
        rec_grid = make_3dgrid(prot_coords, prot_features, max_dist=self.max_dist,
                               grid_resolution=self.grid_resolution)
        pocket_dens = make_3dgrid(poc_coords, poc_features, max_dist=self.max_dist,
                                  grid_resolution=1)
        margin = ndimage.maximum_filter(pocket_dens, footprint=footprint)
        pocket_dens += margin
        pocket_dens = pocket_dens.clip(v_min, v_max)

        zoom = rec_grid.shape[1] / pocket_dens.shape[1]
        pocket_dens = np.stack([ndimage.zoom(pocket_dens[0, ..., i],
                                             zoom)
                                for i in range(poc_features.shape[1])], -1)
        rec_grid = np.squeeze(rec_grid)
        rec_grid = rec_grid.transpose(3, 0, 1, 2)
        pocket_dens = pocket_dens.transpose(3, 0, 1, 2)
        return rec_grid, pocket_dens


class SitepredDataModule(LightningDataModule):

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

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, **kwargs):
        """define train, test and validation datasets """
        self.train_dataset = SitepredDataset(self.hdf_path, self.max_dist, self.grid_resolution,
                                             self.train_ids_path, self.transform)
        self.valid_dataset = SitepredDataset(self.hdf_path, self.max_dist, self.grid_resolution,
                                             self.valid_ids_path, False)
        self.test_dataset = SitepredDataset(self.hdf_path, self.max_dist, self.grid_resolution,
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
    parser.add_argument("--data_path", type=str, default="data", help="path where mol2 files are stored")
    parser.add_argument("--hdf_path", type=str, default="data.h5",
                        help="path where dataset is stored")
    hparams = parser.parse_args()
    return hparams


if __name__ == '__main__':
    hparams = parser_args()
    prepare_dataset(hparams.data_path, hparams.hdf_path)
