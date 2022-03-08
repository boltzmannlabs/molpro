import os
from argparse import ArgumentParser
import h5py
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from molpro.utils.preprocess import rotate_grid, make_3dgrid, Featurizer
from molpro.utils.dataset import SiteGenDataset


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
                                               'mol2', named_props=['partialcharge'],
                                               smarts_labels=['hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring'],
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
